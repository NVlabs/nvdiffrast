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
#include "../common/rasterize.h"
#include "../common/cudaraster/CudaRaster.hpp"
#include "../common/cudaraster/impl/Constants.hpp"
#include <tuple>

//------------------------------------------------------------------------
// Kernel prototypes.

void RasterizeCudaFwdShaderKernel(const RasterizeCudaFwdShaderParams p);
void RasterizeGradKernel(const RasterizeGradParams p);
void RasterizeGradKernelDb(const RasterizeGradParams p);

//------------------------------------------------------------------------
// Python CudaRaster state wrapper methods.

RasterizeCRStateWrapper::RasterizeCRStateWrapper(int cudaDeviceIdx_)
{
    const at::cuda::OptionalCUDAGuard device_guard(cudaDeviceIdx_);
    cudaDeviceIdx = cudaDeviceIdx_;
    cr = new CR::CudaRaster();
}

RasterizeCRStateWrapper::~RasterizeCRStateWrapper(void)
{
    const at::cuda::OptionalCUDAGuard device_guard(cudaDeviceIdx);
    delete cr;
}

//------------------------------------------------------------------------
// Forward op (Cuda).

std::tuple<torch::Tensor, torch::Tensor> rasterize_fwd_cuda(RasterizeCRStateWrapper& stateWrapper, torch::Tensor pos, torch::Tensor tri, std::tuple<int, int> resolution, torch::Tensor ranges, int peeling_idx)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(pos));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CR::CudaRaster* cr = stateWrapper.cr;

    // Check inputs.
    NVDR_CHECK_DEVICE(pos, tri);
    NVDR_CHECK_CPU(ranges);
    NVDR_CHECK_CONTIGUOUS(pos, tri, ranges);
    NVDR_CHECK_F32(pos);
    NVDR_CHECK_I32(tri, ranges);

    // Check that CudaRaster context was created for the correct GPU.
    NVDR_CHECK(pos.get_device() == stateWrapper.cudaDeviceIdx, "CudaRaster context must must reside on the same device as input tensors");

    // Determine instance mode and check input dimensions.
    bool instance_mode = pos.sizes().size() > 2;
    if (instance_mode)
        NVDR_CHECK(pos.sizes().size() == 3 && pos.size(0) > 0 && pos.size(1) > 0 && pos.size(2) == 4, "instance mode - pos must have shape [>0, >0, 4]");
    else
    {
        NVDR_CHECK(pos.sizes().size() == 2 && pos.size(0) > 0 && pos.size(1) == 4, "range mode - pos must have shape [>0, 4]");
        NVDR_CHECK(ranges.sizes().size() == 2 && ranges.size(0) > 0 && ranges.size(1) == 2, "range mode - ranges must have shape [>0, 2]");
    }
    NVDR_CHECK(tri.sizes().size() == 2 && tri.size(0) > 0 && tri.size(1) == 3, "tri must have shape [>0, 3]");

    // Get output shape.
    int height_out = std::get<0>(resolution);
    int width_out  = std::get<1>(resolution);
    int depth      = instance_mode ? pos.size(0) : ranges.size(0); // Depth of tensor, not related to depth buffering.
    NVDR_CHECK(height_out > 0 && width_out > 0, "resolution must be [>0, >0]");

    // Round internal resolution up to tile size.
    int height = (height_out + CR_TILE_SIZE - 1) & (-CR_TILE_SIZE);
    int width  = (width_out  + CR_TILE_SIZE - 1) & (-CR_TILE_SIZE);

    // Get position and triangle buffer sizes in vertices / triangles.
    int posCount = instance_mode ? pos.size(1) : pos.size(0);
    int triCount = tri.size(0);

    // Set up CudaRaster buffers.
    const float* posPtr = pos.data_ptr<float>();
    const int32_t* rangesPtr = instance_mode ? 0 : ranges.data_ptr<int32_t>(); // This is in CPU memory.
    const int32_t* triPtr = tri.data_ptr<int32_t>();
    cr->setVertexBuffer((void*)posPtr, posCount);
    cr->setIndexBuffer((void*)triPtr, triCount);
    cr->setBufferSize(width_out, height_out, depth);

    // Enable depth peeling?
    bool enablePeel = (peeling_idx > 0);
    cr->setRenderModeFlags(enablePeel ? CR::CudaRaster::RenderModeFlag_EnableDepthPeeling : 0); // No backface culling.
    if (enablePeel)
        cr->swapDepthAndPeel(); // Use previous depth buffer as peeling depth input.

    // Determine viewport tiling.
    int tileCountX = (width  + CR_MAXVIEWPORT_SIZE - 1) / CR_MAXVIEWPORT_SIZE;
    int tileCountY = (height + CR_MAXVIEWPORT_SIZE - 1) / CR_MAXVIEWPORT_SIZE;
    int tileSizeX = ((width  + tileCountX - 1) / tileCountX + CR_TILE_SIZE - 1) & (-CR_TILE_SIZE);
    int tileSizeY = ((height + tileCountY - 1) / tileCountY + CR_TILE_SIZE - 1) & (-CR_TILE_SIZE);
    TORCH_CHECK(tileCountX > 0 && tileCountY > 0 && tileSizeX > 0 && tileSizeY > 0,             "internal error in tile size calculation: count or size is zero");
    TORCH_CHECK(tileSizeX <= CR_MAXVIEWPORT_SIZE && tileSizeY <= CR_MAXVIEWPORT_SIZE,           "internal error in tile size calculation: tile larger than allowed");
    TORCH_CHECK((tileSizeX & (CR_TILE_SIZE - 1)) == 0 && (tileSizeY & (CR_TILE_SIZE - 1)) == 0, "internal error in tile size calculation: tile not divisible by ", CR_TILE_SIZE);
    TORCH_CHECK(tileCountX * tileSizeX >= width && tileCountY * tileSizeY >= height,            "internal error in tile size calculation: tiles do not cover viewport");

    // Rasterize in tiles.
    for (int tileY = 0; tileY < tileCountY; tileY++)
    for (int tileX = 0; tileX < tileCountX; tileX++)
    {
        // Set CudaRaster viewport according to tile.
        int offsetX = tileX * tileSizeX;
        int offsetY = tileY * tileSizeY;
        int sizeX = (width_out  - offsetX) < tileSizeX ? (width_out  - offsetX) : tileSizeX;
        int sizeY = (height_out - offsetY) < tileSizeY ? (height_out - offsetY) : tileSizeY;
        cr->setViewport(sizeX, sizeY, offsetX, offsetY);

        // Run all triangles in one batch. In case of error, the workload could be split into smaller batches - maybe do that in the future.
        // Only enable peeling-specific optimizations to skip first stages when image fits in one tile. Those are not valid otherwise.
        cr->deferredClear(0u);
        bool success = cr->drawTriangles(rangesPtr, enablePeel && (tileCountX == 1 && tileCountY == 1), stream);
        NVDR_CHECK(success, "subtriangle count overflow");
    }

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({depth, height_out, width_out, 4}, opts);
    torch::Tensor out_db = torch::empty({depth, height_out, width_out, 4}, opts);

    // Populate pixel shader kernel parameters.
    RasterizeCudaFwdShaderParams p;
    p.pos = posPtr;
    p.tri = triPtr;
    p.in_idx = (const int*)cr->getColorBuffer();
    p.out = out.data_ptr<float>();
    p.out_db = out_db.data_ptr<float>();
    p.numTriangles = triCount;
    p.numVertices = posCount;
    p.width_in = width;
    p.height_in = height;
    p.width_out = width_out;
    p.height_out = height_out;
    p.depth  = depth;
    p.instance_mode = (pos.sizes().size() > 2) ? 1 : 0;
    p.xs = 2.f / (float)width_out;
    p.xo = 1.f / (float)width_out - 1.f;
    p.ys = 2.f / (float)height_out;
    p.yo = 1.f / (float)height_out - 1.f;

    // Verify that buffers are aligned to allow float2/float4 operations.
    NVDR_CHECK(!((uintptr_t)p.pos & 15),    "pos input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.out & 15),    "out output tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.out_db & 15), "out_db output tensor not aligned to float4");

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(RAST_CUDA_FWD_SHADER_KERNEL_BLOCK_WIDTH, RAST_CUDA_FWD_SHADER_KERNEL_BLOCK_HEIGHT, p.width_out, p.height_out);
    dim3 gridSize  = getLaunchGridSize(blockSize, p.width_out, p.height_out, p.depth);

    // Launch CUDA kernel.
    void* args[] = {&p};
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((void*)RasterizeCudaFwdShaderKernel, gridSize, blockSize, args, 0, stream));

    // Return.
    return std::tuple<torch::Tensor, torch::Tensor>(out, out_db);
}

//------------------------------------------------------------------------
// Gradient op.

torch::Tensor rasterize_grad_db(torch::Tensor pos, torch::Tensor tri, torch::Tensor out, torch::Tensor dy, torch::Tensor ddb)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(pos));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    RasterizeGradParams p;
    bool enable_db = ddb.defined();

    // Check inputs.
    if (enable_db)
    {
        NVDR_CHECK_DEVICE(pos, tri, out, dy, ddb);
        NVDR_CHECK_CONTIGUOUS(pos, tri, out);
        NVDR_CHECK_F32(pos, out, dy, ddb);
        NVDR_CHECK_I32(tri);
    }
    else
    {
        NVDR_CHECK_DEVICE(pos, tri, out, dy);
        NVDR_CHECK_CONTIGUOUS(pos, tri, out);
        NVDR_CHECK_F32(pos, out, dy);
        NVDR_CHECK_I32(tri);
    }

    // Determine instance mode.
    p.instance_mode = (pos.sizes().size() > 2) ? 1 : 0;

    // Shape is taken from the rasterizer output tensor.
    NVDR_CHECK(out.sizes().size() == 4, "tensor out must be rank-4");
    p.depth  = out.size(0);
    p.height = out.size(1);
    p.width  = out.size(2);
    NVDR_CHECK(p.depth > 0 && p.height > 0 && p.width > 0, "resolution must be [>0, >0, >0]");

    // Check other shapes.
    if (p.instance_mode)
        NVDR_CHECK(pos.sizes().size() == 3 && pos.size(0) == p.depth && pos.size(1) > 0 && pos.size(2) == 4, "pos must have shape [depth, >0, 4]");
    else
        NVDR_CHECK(pos.sizes().size() == 2 && pos.size(0) > 0 && pos.size(1) == 4, "pos must have shape [>0, 4]");
    NVDR_CHECK(tri.sizes().size() == 2 && tri.size(0) > 0 && tri.size(1) == 3, "tri must have shape [>0, 3]");
    NVDR_CHECK(out.sizes().size() == 4 && out.size(0) == p.depth && out.size(1) == p.height && out.size(2) == p.width && out.size(3) == 4, "out must have shape [depth, height, width, 4]");
    NVDR_CHECK( dy.sizes().size() == 4 &&  dy.size(0) == p.depth &&  dy.size(1) == p.height &&  dy.size(2) == p.width &&  dy.size(3) == 4, "dy must have shape [depth, height, width, 4]");
    if (enable_db)
        NVDR_CHECK(ddb.sizes().size() == 4 && ddb.size(0) == p.depth && ddb.size(1) == p.height && ddb.size(2) == p.width && ddb.size(3) == 4, "ddb must have shape [depth, height, width, 4]");

    // Ensure gradients are contiguous.
    torch::Tensor dy_ = dy.contiguous();
    torch::Tensor ddb_;
    if (enable_db)
        ddb_ = ddb.contiguous();

    // Populate parameters.
    p.numTriangles = tri.size(0);
    p.numVertices = p.instance_mode ? pos.size(1) : pos.size(0);
    p.pos = pos.data_ptr<float>();
    p.tri = tri.data_ptr<int>();
    p.out = out.data_ptr<float>();
    p.dy  = dy_.data_ptr<float>();
    p.ddb = enable_db ? ddb_.data_ptr<float>() : NULL;

    // Set up pixel position to clip space x, y transform.
    p.xs = 2.f / (float)p.width;
    p.xo = 1.f / (float)p.width - 1.f;
    p.ys = 2.f / (float)p.height;
    p.yo = 1.f / (float)p.height - 1.f;

    // Allocate output tensor for position gradients.
    torch::Tensor grad = torch::zeros_like(pos);
    p.grad = grad.data_ptr<float>();

    // Verify that buffers are aligned to allow float2/float4 operations.
    NVDR_CHECK(!((uintptr_t)p.pos & 15), "pos input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.dy  &  7), "dy input tensor not aligned to float2");
    NVDR_CHECK(!((uintptr_t)p.ddb & 15), "ddb input tensor not aligned to float4");

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(RAST_GRAD_MAX_KERNEL_BLOCK_WIDTH, RAST_GRAD_MAX_KERNEL_BLOCK_HEIGHT, p.width, p.height);
    dim3 gridSize  = getLaunchGridSize(blockSize, p.width, p.height, p.depth);

    // Launch CUDA kernel.
    void* args[] = {&p};
    void* func = enable_db ? (void*)RasterizeGradKernelDb : (void*)RasterizeGradKernel;
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func, gridSize, blockSize, args, 0, stream));

    // Return the gradients.
    return grad;
}

// Version without derivatives.
torch::Tensor rasterize_grad(torch::Tensor pos, torch::Tensor tri, torch::Tensor out, torch::Tensor dy)
{
    torch::Tensor empty_tensor;
    return rasterize_grad_db(pos, tri, out, dy, empty_tensor);
}

//------------------------------------------------------------------------
