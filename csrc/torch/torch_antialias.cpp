// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "torch_common.inl"
#include "torch_types.h"
#include "../common/common.h"
#include "../common/antialias.h"

//------------------------------------------------------------------------
// Kernel prototypes.

void AntialiasFwdMeshKernel         (const AntialiasKernelParams p);
void AntialiasFwdDiscontinuityKernel(const AntialiasKernelParams p);
void AntialiasFwdAnalysisKernel     (const AntialiasKernelParams p);
void AntialiasGradKernel            (const AntialiasKernelParams p);

//------------------------------------------------------------------------
// Topology hash construction.

TopologyHashWrapper antialias_construct_topology_hash(torch::Tensor tri)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(tri));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AntialiasKernelParams p = {}; // Initialize all fields to zero.

    // Check inputs.
    NVDR_CHECK_DEVICE(tri);
    NVDR_CHECK_CONTIGUOUS(tri);
    NVDR_CHECK_I32(tri);
    NVDR_CHECK(tri.sizes().size() == 2 && tri.size(0) > 0 && tri.size(1) == 3, "tri must have shape [>0, 3]");

    // Fill in kernel parameters.
    p.numTriangles = tri.size(0);
    p.numVertices = 0x7fffffff; // Let's not require vertex positions just to enable an error check.
    p.tri = tri.data_ptr<int>();

    // Kernel parameters.
    p.allocTriangles = 64;
    while (p.allocTriangles < p.numTriangles)
        p.allocTriangles <<= 1; // Must be power of two.

    // Construct the hash tensor and get pointer.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor ev_hash = torch::zeros({(uint64_t)p.allocTriangles * AA_HASH_ELEMENTS_PER_TRIANGLE(p.allocTriangles) * 4}, opts);
    p.evHash = (uint4*)(ev_hash.data_ptr<int>());

    // Check alignment.
    NVDR_CHECK(!((uintptr_t)p.evHash & 15), "ev_hash internal tensor not aligned to int4");

    // Populate the hash.
    void* args[] = {&p};
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((void*)AntialiasFwdMeshKernel, (p.numTriangles - 1) / AA_MESH_KERNEL_THREADS_PER_BLOCK + 1, AA_MESH_KERNEL_THREADS_PER_BLOCK, args, 0, stream));

    // Return.
    TopologyHashWrapper hash_wrap;
    hash_wrap.ev_hash = ev_hash;
    return hash_wrap;
}

//------------------------------------------------------------------------
// Forward op.

std::tuple<torch::Tensor, torch::Tensor> antialias_fwd(torch::Tensor color, torch::Tensor rast, torch::Tensor pos, torch::Tensor tri, TopologyHashWrapper topology_hash_wrap)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(color));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AntialiasKernelParams p = {}; // Initialize all fields to zero.
    p.instance_mode = (pos.sizes().size() > 2) ? 1 : 0;
    torch::Tensor& topology_hash = topology_hash_wrap.ev_hash; // Unwrap.

    // Check inputs.
    NVDR_CHECK_DEVICE(color, rast, pos, tri, topology_hash);
    NVDR_CHECK_CONTIGUOUS(color, rast, pos, tri, topology_hash);
    NVDR_CHECK_F32(color, rast, pos);
    NVDR_CHECK_I32(tri, topology_hash);

    // Sanity checks.
    NVDR_CHECK(color.sizes().size() == 4 && color.size(0) > 0 && color.size(1) > 0 && color.size(2) > 0 && color.size(3) > 0, "color must have shape[>0, >0, >0, >0]");
    NVDR_CHECK(rast.sizes().size() == 4 && rast.size(0) > 0 && rast.size(1) > 0 && rast.size(2) > 0 && rast.size(3) == 4, "rast must have shape[>0, >0, >0, 4]");
    NVDR_CHECK(tri.sizes().size() == 2 && tri.size(0) > 0 && tri.size(1) == 3, "tri must have shape [>0, 3]");
    NVDR_CHECK(color.size(1) == rast.size(1) && color.size(2) == rast.size(2), "color and rast inputs must have same spatial dimensions");
    if (p.instance_mode)
    {
        NVDR_CHECK(pos.sizes().size() == 3 && pos.size(0) > 0 && pos.size(1) > 0 && pos.size(2) == 4, "pos must have shape [>0, >0, 4] or [>0, 4]");
        NVDR_CHECK(rast.size(0) == color.size(0) && pos.size(0) == color.size(0), "minibatch size mismatch between inputs color, rast, pos");
    }
    else
    {
        NVDR_CHECK(pos.sizes().size() == 2 && pos.size(0) > 0 && pos.size(1) == 4, "pos must have shape [>0, >0, 4] or [>0, 4]");
        NVDR_CHECK(rast.size(0) == color.size(0), "minibatch size mismatch between inputs color, rast");
    }

    // Extract input dimensions.
    p.numVertices  = pos.size(p.instance_mode ? 1 : 0);
    p.numTriangles = tri.size(0);
    p.n            = color.size(0);
    p.height       = color.size(1);
    p.width        = color.size(2);
    p.channels     = color.size(3);

    // Get input pointers.
    p.color = color.data_ptr<float>();
    p.rasterOut = rast.data_ptr<float>();
    p.tri = tri.data_ptr<int>();
    p.pos = pos.data_ptr<float>();
    p.evHash = (uint4*)(topology_hash.data_ptr<int>());

    // Misc parameters.
    p.xh = .5f * (float)p.width;
    p.yh = .5f * (float)p.height;

    // Determine hash allocation size.
    p.allocTriangles = 64;
    while (p.allocTriangles < p.numTriangles)
        p.allocTriangles <<= 1; // Must be power of two.

    // Allocate output tensors.
    torch::Tensor out = color.detach().clone(); // Use color as base.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor work_buffer = torch::empty({p.n * p.width * p.height * 8 + 4}, opts); // 8 int for a maximum of two work items per pixel.
    p.output = out.data_ptr<float>();
    p.workBuffer = (int4*)(work_buffer.data_ptr<float>());

    // Clear the work counters.
    NVDR_CHECK_CUDA_ERROR(cudaMemsetAsync(p.workBuffer, 0, sizeof(int4), stream));

    // Verify that buffers are aligned to allow float2/float4 operations.
    NVDR_CHECK(!((uintptr_t)p.pos        & 15), "pos input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.rasterOut  &  7), "raster_out input tensor not aligned to float2");
    NVDR_CHECK(!((uintptr_t)p.workBuffer & 15), "work_buffer internal tensor not aligned to int4");
    NVDR_CHECK(!((uintptr_t)p.evHash     & 15), "topology_hash internal tensor not aligned to int4");

    // Choose launch parameters for the discontinuity finder kernel and launch.
    void* args[] = {&p};
    dim3 blockSize(AA_DISCONTINUITY_KERNEL_BLOCK_WIDTH, AA_DISCONTINUITY_KERNEL_BLOCK_HEIGHT, 1);
    dim3 gridSize = getLaunchGridSize(blockSize, p.width, p.height, p.n);
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((void*)AntialiasFwdDiscontinuityKernel, gridSize, blockSize, args, 0, stream));

    // Determine optimum block size for the persistent analysis kernel and launch.
    int device = 0;
    int numCTA = 0;
    int numSM  = 0;
    NVDR_CHECK_CUDA_ERROR(cudaGetDevice(&device));
    NVDR_CHECK_CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numCTA, (void*)AntialiasFwdAnalysisKernel, AA_ANALYSIS_KERNEL_THREADS_PER_BLOCK, 0));
    NVDR_CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, device));
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((void*)AntialiasFwdAnalysisKernel, numCTA * numSM, AA_ANALYSIS_KERNEL_THREADS_PER_BLOCK, args, 0, stream));

    // Return results.
    return std::tuple<torch::Tensor, torch::Tensor>(out, work_buffer);
}

//------------------------------------------------------------------------
// Gradient op.

std::tuple<torch::Tensor, torch::Tensor> antialias_grad(torch::Tensor color, torch::Tensor rast, torch::Tensor pos, torch::Tensor tri, torch::Tensor dy, torch::Tensor work_buffer)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(color));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AntialiasKernelParams p = {}; // Initialize all fields to zero.
    p.instance_mode = (pos.sizes().size() > 2) ? 1 : 0;

    // Check inputs.
    NVDR_CHECK_DEVICE(color, rast, pos, tri, dy, work_buffer);
    NVDR_CHECK_CONTIGUOUS(color, rast, pos, tri, work_buffer);
    NVDR_CHECK_F32(color, rast, pos, dy, work_buffer);
    NVDR_CHECK_I32(tri);

    // Sanity checks.
    NVDR_CHECK(dy.sizes().size() == 4 && dy.size(0) > 0 && dy.size(1) > 0 && dy.size(2) > 0 && dy.size(3) > 0, "dy must have shape[>0, >0, >0, >0]");
    NVDR_CHECK(color.sizes().size() == 4 && color.size(0) > 0 && color.size(1) > 0 && color.size(2) > 0 && color.size(3) > 0, "color must have shape[>0, >0, >0, >0]");
    NVDR_CHECK(rast.sizes().size() == 4 && rast.size(0) > 0 && rast.size(1) > 0 && rast.size(2) > 0 && rast.size(3) == 4, "raster_out must have shape[>0, >0, >0, 4]");
    NVDR_CHECK(tri.sizes().size() == 2 && tri.size(0) > 0 && tri.size(1) == 3, "tri must have shape [>0, 3]");
    NVDR_CHECK(color.size(1) == rast.size(1) && color.size(2) == rast.size(2), "color and raster_out inputs must have same spatial dimensions");
    NVDR_CHECK(color.size(1) == dy.size(1) && color.size(2) == dy.size(2) && color.size(3) == dy.size(3), "color and dy inputs must have same dimensions");
    if (p.instance_mode)
    {
        NVDR_CHECK(pos.sizes().size() == 3 && pos.size(0) > 0 && pos.size(1) > 0 && pos.size(2) == 4, "pos must have shape [>0, >0, 4] or [>0, 4]");
        NVDR_CHECK(rast.size(0) == color.size(0) && pos.size(0) == color.size(0), "minibatch size mismatch between inputs color, raster_out, pos");
        NVDR_CHECK(dy.size(0) == color.size(0) && rast.size(0) == color.size(0) && pos.size(0) ==color.size(0), "minibatch size mismatch between inputs dy, color, raster_out, pos");
    }
    else
    {
        NVDR_CHECK(pos.sizes().size() == 2 && pos.size(0) > 0 && pos.size(1) == 4, "pos must have shape [>0, >0, 4] or [>0, 4]");
        NVDR_CHECK(rast.size(0) == color.size(0), "minibatch size mismatch between inputs color, raster_out");
        NVDR_CHECK(dy.size(0) == color.size(0) && rast.size(0) == color.size(0), "minibatch size mismatch between inputs dy, color, raster_out");
    }

    // Extract input dimensions.
    p.numVertices  = pos.size(p.instance_mode ? 1 : 0);
    p.numTriangles = tri.size(0);
    p.n            = color.size(0);
    p.height       = color.size(1);
    p.width        = color.size(2);
    p.channels     = color.size(3);

    // Ensure dy is contiguous.
    torch::Tensor dy_ = dy.contiguous();

    // Get input pointers.
    p.color = color.data_ptr<float>();
    p.rasterOut = rast.data_ptr<float>();
    p.tri = tri.data_ptr<int>();
    p.pos = pos.data_ptr<float>();
    p.dy = dy_.data_ptr<float>();
    p.workBuffer = (int4*)(work_buffer.data_ptr<float>());

    // Misc parameters.
    p.xh = .5f * (float)p.width;
    p.yh = .5f * (float)p.height;

    // Allocate output tensors.
    torch::Tensor grad_color = dy_.detach().clone(); // Use dy as base.
    torch::Tensor grad_pos = torch::zeros_like(pos);
    p.gradColor = grad_color.data_ptr<float>();
    p.gradPos = grad_pos.data_ptr<float>();

    // Clear gradient kernel work counter.
    NVDR_CHECK_CUDA_ERROR(cudaMemsetAsync(&p.workBuffer[0].y, 0, sizeof(int), stream));

    // Verify that buffers are aligned to allow float2/float4 operations.
    NVDR_CHECK(!((uintptr_t)p.pos        & 15), "pos input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.workBuffer & 15), "work_buffer internal tensor not aligned to int4");

    // Determine optimum block size for the gradient kernel and launch.
    void* args[] = {&p};
    int device = 0;
    int numCTA = 0;
    int numSM  = 0;
    NVDR_CHECK_CUDA_ERROR(cudaGetDevice(&device));
    NVDR_CHECK_CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numCTA, (void*)AntialiasGradKernel, AA_GRAD_KERNEL_THREADS_PER_BLOCK, 0));
    NVDR_CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, device));
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((void*)AntialiasGradKernel, numCTA * numSM, AA_GRAD_KERNEL_THREADS_PER_BLOCK, args, 0, stream));

    // Return results.
    return std::tuple<torch::Tensor, torch::Tensor>(grad_color, grad_pos);
}

//------------------------------------------------------------------------
