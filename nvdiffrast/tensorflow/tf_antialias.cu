// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

//------------------------------------------------------------------------
// Forward TensorFlow op.

struct AntialiasFwdOp : public OpKernel
{
    AntialiasKernelParams m_attribs;

    AntialiasFwdOp(OpKernelConstruction* ctx): OpKernel(ctx)
    {
        memset(&m_attribs, 0, sizeof(m_attribs));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("tri_const", &m_attribs.tri_const));
    }

    void Compute(OpKernelContext* ctx)
    {
        AntialiasKernelParams& p = m_attribs;
        cudaStream_t stream = ctx->eigen_device<Eigen::GpuDevice>().stream();

        // Get input.
        const Tensor& color     = ctx->input(0);
        const Tensor& rasterOut = ctx->input(1);
        const Tensor& pos       = ctx->input(2);
        const Tensor& tri       = ctx->input(3);

        // Instance rendering mode?
        p.instance_mode = pos.dims() > 2;

        // Extract input dimensions.
        if (p.instance_mode)
            p.numVertices = (pos.dims() > 1) ? pos.dim_size(1) : 0;
        else
            p.numVertices = (pos.dims() > 0) ? pos.dim_size(0) : 0;
        p.numTriangles = (tri.dims() > 0) ? tri.dim_size(0) : 0;
        p.n        = (color.dims() > 0) ? color.dim_size(0) : 0;
        p.height   = (color.dims() > 1) ? color.dim_size(1) : 0;
        p.width    = (color.dims() > 2) ? color.dim_size(2) : 0;
        p.channels = (color.dims() > 3) ? color.dim_size(3) : 0;

        // Sanity checks.
        OP_REQUIRES(ctx, color.dims() == 4 && color.dim_size(0) > 0 && color.dim_size(1) > 0 && color.dim_size(2) > 0 && color.dim_size(3) > 0, errors::InvalidArgument("color must have shape[>0, >0, >0, >0]"));
        OP_REQUIRES(ctx, rasterOut.dims() == 4 && rasterOut.dim_size(0) > 0 && rasterOut.dim_size(1) > 0 && rasterOut.dim_size(2) > 0 && rasterOut.dim_size(3) == 4, errors::InvalidArgument("raster_out must have shape[>0, >0, >0, 4]"));
        OP_REQUIRES(ctx, tri.dims() == 2 && tri.dim_size(0) > 0 && tri.dim_size(1) == 3, errors::InvalidArgument("tri must have shape [>0, 3]"));
        OP_REQUIRES(ctx, color.dim_size(1) == rasterOut.dim_size(1) && color.dim_size(2) == rasterOut.dim_size(2), errors::InvalidArgument("color and raster_out inputs must have same spatial dimensions"));
        if (p.instance_mode)
        {
            OP_REQUIRES(ctx, pos.dims() == 3 && pos.dim_size(0) > 0 && pos.dim_size(1) > 0 && pos.dim_size(2) == 4, errors::InvalidArgument("pos must have shape [>0, >0, 4] or [>0, 4]"));
            OP_REQUIRES(ctx, rasterOut.dim_size(0) == p.n && pos.dim_size(0) == p.n, errors::InvalidArgument("minibatch size mismatch between inputs color, raster_out, pos"));
        }
        else
        {
            OP_REQUIRES(ctx, pos.dims() == 2 && pos.dim_size(0) > 0 && pos.dim_size(1) == 4, errors::InvalidArgument("pos must have shape [>0, >0, 4] or [>0, 4]"));
            OP_REQUIRES(ctx, rasterOut.dim_size(0) == p.n, errors::InvalidArgument("minibatch size mismatch between inputs color, raster_out"));
        }

        // Get input pointers.
        p.color = color.flat<float>().data();
        p.rasterOut = rasterOut.flat<float>().data();
        p.tri = tri.flat<int>().data();
        p.pos = pos.flat<float>().data();

        // Misc parameters.
        p.xh = .5f * (float)p.width;
        p.yh = .5f * (float)p.height;

        // Allocate output tensor.
        Tensor* outputTensor = NULL;
        TensorShape outputShape;
        outputShape.AddDim(p.n);
        outputShape.AddDim(p.height);
        outputShape.AddDim(p.width);
        outputShape.AddDim(p.channels);
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, outputShape, &outputTensor));
        p.output = outputTensor->flat<float>().data();

        // Allocate work buffer. One extra int4 for storing counters.
        Tensor* workTensor = NULL;
        TensorShape workShape;
        workShape.AddDim(p.n * p.width * p.height * 8 + 4); // 8 int for a maximum of two work items per pixel.
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, workShape, &workTensor));
        p.workBuffer = (int4*)(workTensor->flat<int>().data());

        // Clear the work counters.
        OP_CHECK_CUDA_ERROR(ctx, cudaMemsetAsync(p.workBuffer, 0, sizeof(int4), stream));

        // Verify that buffers are aligned to allow float2/float4 operations.
        OP_REQUIRES(ctx, !((uintptr_t)p.pos        & 15), errors::Internal("pos input tensor not aligned to float4"));
        OP_REQUIRES(ctx, !((uintptr_t)p.rasterOut  &  7), errors::Internal("raster_out input tensor not aligned to float2"));
        OP_REQUIRES(ctx, !((uintptr_t)p.workBuffer & 15), errors::Internal("work_buffer internal tensor not aligned to int4"));

        // Kernel parameters.
        void* args[] = {&p};

        // (Re-)calculate opposite vertex hash.
        if (!p.evHash || !p.tri_const)
        {            
            if (p.allocTriangles < p.numTriangles)
            {
                p.allocTriangles = max(p.allocTriangles, 64);
                while (p.allocTriangles < p.numTriangles)
                    p.allocTriangles <<= 1; // Must be power of two.
               
                // (Re-)allocate memory for the hash.
                OP_CHECK_CUDA_ERROR(ctx, cudaFree(p.evHash));
                OP_CHECK_CUDA_ERROR(ctx, cudaMalloc(&p.evHash, p.allocTriangles * AA_HASH_ELEMENTS_PER_TRIANGLE(p.allocTriangles) * sizeof(uint4)));
                LOG(INFO) << "Increasing topology hash size to accommodate " << p.allocTriangles << " triangles";
            }

            // Clear the hash and launch the mesh kernel to populate it.
            OP_CHECK_CUDA_ERROR(ctx, cudaMemsetAsync(p.evHash, 0, p.allocTriangles * AA_HASH_ELEMENTS_PER_TRIANGLE(p.allocTriangles) * sizeof(uint4), stream));
            OP_CHECK_CUDA_ERROR(ctx, cudaLaunchKernel((void*)AntialiasFwdMeshKernel, (p.numTriangles - 1) / AA_MESH_KERNEL_THREADS_PER_BLOCK + 1, AA_MESH_KERNEL_THREADS_PER_BLOCK, args, 0, stream));
        }

        // Copy input to output as a baseline.
        OP_CHECK_CUDA_ERROR(ctx, cudaMemcpyAsync(p.output, p.color, p.n * p.height * p.width * p.channels * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        // Choose launch parameters for the discontinuity finder kernel and launch.
        dim3 blockSize(AA_DISCONTINUITY_KERNEL_BLOCK_WIDTH, AA_DISCONTINUITY_KERNEL_BLOCK_HEIGHT, 1);
        dim3 gridSize = getLaunchGridSize(blockSize, p.width, p.height, p.n);
        OP_CHECK_CUDA_ERROR(ctx, cudaLaunchKernel((void*)AntialiasFwdDiscontinuityKernel, gridSize, blockSize, args, 0, stream));

        // Determine optimum block size for the persistent analysis kernel.
        int device = 0;
        int numCTA = 0;
        int numSM  = 0;
        OP_CHECK_CUDA_ERROR(ctx, cudaGetDevice(&device));
        OP_CHECK_CUDA_ERROR(ctx, cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numCTA, (void*)AntialiasFwdAnalysisKernel, AA_ANALYSIS_KERNEL_THREADS_PER_BLOCK, 0));
        OP_CHECK_CUDA_ERROR(ctx, cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, device));

        // Launch analysis kernel.
        OP_CHECK_CUDA_ERROR(ctx, cudaLaunchKernel((void*)AntialiasFwdAnalysisKernel, numCTA * numSM, AA_ANALYSIS_KERNEL_THREADS_PER_BLOCK, args, 0, stream));
    }
};

REGISTER_OP("AntialiasFwd")
    .Input      ("color: float")
    .Input      ("raster_out: float")
    .Input      ("pos: float")
    .Input      ("tri: int32")
    .Output     ("output: float")
    .Output     ("work_buffer: int32")
    .Attr       ("tri_const: int");

REGISTER_KERNEL_BUILDER(Name("AntialiasFwd").Device(DEVICE_GPU), AntialiasFwdOp);

//------------------------------------------------------------------------
// Gradient TensorFlow op.

struct AntialiasGradOp : public OpKernel
{
    AntialiasKernelParams m_attribs;

    AntialiasGradOp(OpKernelConstruction* ctx): OpKernel(ctx)
    {
        memset(&m_attribs, 0, sizeof(m_attribs));
    }

    void Compute(OpKernelContext* ctx)
    {
        AntialiasKernelParams& p = m_attribs;
        cudaStream_t stream = ctx->eigen_device<Eigen::GpuDevice>().stream();

        // Get input.
        const Tensor& color      = ctx->input(0);
        const Tensor& rasterOut  = ctx->input(1);
        const Tensor& pos        = ctx->input(2);
        const Tensor& tri        = ctx->input(3);
        const Tensor& dy         = ctx->input(4);
        const Tensor& workBuffer = ctx->input(5);

        // Instance rendering mode?
        p.instance_mode = pos.dims() > 2;

        // Extract input dimensions.
        if (p.instance_mode)
            p.numVertices = (pos.dims() > 1) ? pos.dim_size(1) : 0;
        else
            p.numVertices = (pos.dims() > 0) ? pos.dim_size(0) : 0;
        p.numTriangles = (tri.dims() > 0) ? tri.dim_size(0) : 0;
        p.n        = (color.dims() > 0) ? color.dim_size(0) : 0;
        p.height   = (color.dims() > 1) ? color.dim_size(1) : 0;
        p.width    = (color.dims() > 2) ? color.dim_size(2) : 0;
        p.channels = (color.dims() > 3) ? color.dim_size(3) : 0;

        // Sanity checks.
        OP_REQUIRES(ctx, dy.dims() == 4 && dy.dim_size(0) > 0 && dy.dim_size(1) > 0 && dy.dim_size(2) > 0 && dy.dim_size(3) > 0, errors::InvalidArgument("dy must have shape[>0, >0, >0, >0]"));
        OP_REQUIRES(ctx, color.dims() == 4 && color.dim_size(0) > 0 && color.dim_size(1) > 0 && color.dim_size(2) > 0 && color.dim_size(3) > 0, errors::InvalidArgument("color must have shape[>0, >0, >0, >0]"));
        OP_REQUIRES(ctx, rasterOut.dims() == 4 && rasterOut.dim_size(0) > 0 && rasterOut.dim_size(1) > 0 && rasterOut.dim_size(2) > 0 && rasterOut.dim_size(3) == 4, errors::InvalidArgument("raster_out must have shape[>0, >0, >0, 4]"));
        OP_REQUIRES(ctx, tri.dims() == 2 && tri.dim_size(0) > 0 && tri.dim_size(1) == 3, errors::InvalidArgument("tri must have shape [>0, 3]"));
        OP_REQUIRES(ctx, color.dim_size(1) == rasterOut.dim_size(1) && color.dim_size(2) == rasterOut.dim_size(2), errors::InvalidArgument("color and raster_out inputs must have same spatial dimensions"));
        OP_REQUIRES(ctx, color.dim_size(1) == dy.dim_size(1) && color.dim_size(2) == dy.dim_size(2) && color.dim_size(3) == dy.dim_size(3), errors::InvalidArgument("color and dy inputs must have same dimensions"));
        if (p.instance_mode)
        {
            OP_REQUIRES(ctx, pos.dims() == 3 && pos.dim_size(0) > 0 && pos.dim_size(1) > 0 && pos.dim_size(2) == 4, errors::InvalidArgument("pos must have shape [>0, >0, 4] or [>0, 4]"));
            OP_REQUIRES(ctx, rasterOut.dim_size(0) == p.n && pos.dim_size(0) == p.n, errors::InvalidArgument("minibatch size mismatch between inputs color, raster_out, pos"));
            OP_REQUIRES(ctx, dy.dim_size(0) == p.n && rasterOut.dim_size(0) == p.n && pos.dim_size(0) == p.n, errors::InvalidArgument("minibatch size mismatch between inputs dy, color, raster_out, pos"));
        }
        else
        {
            OP_REQUIRES(ctx, pos.dims() == 2 && pos.dim_size(0) > 0 && pos.dim_size(1) == 4, errors::InvalidArgument("pos must have shape [>0, >0, 4] or [>0, 4]"));
            OP_REQUIRES(ctx, rasterOut.dim_size(0) == p.n, errors::InvalidArgument("minibatch size mismatch between inputs color, raster_out"));
            OP_REQUIRES(ctx, dy.dim_size(0) == p.n && rasterOut.dim_size(0) == p.n, errors::InvalidArgument("minibatch size mismatch between inputs dy, color, raster_out"));
        }

        // Get input pointers.
        p.dy = dy.flat<float>().data();
        p.color = color.flat<float>().data();
        p.rasterOut = rasterOut.flat<float>().data();
        p.tri = tri.flat<int>().data();
        p.pos = pos.flat<float>().data();
        p.workBuffer = (int4*)(workBuffer.flat<int>().data());

        // Misc parameters.
        p.xh = .5f * (float)p.width;
        p.yh = .5f * (float)p.height;

        // Allocate color gradient output tensor.
        Tensor* gradColor = NULL;
        TensorShape gradColorShape;
        gradColorShape.AddDim(p.n);
        gradColorShape.AddDim(p.height);
        gradColorShape.AddDim(p.width);
        gradColorShape.AddDim(p.channels);
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, gradColorShape, &gradColor));
        p.gradColor = gradColor->flat<float>().data();

        // Allocate position gradient output tensor.
        Tensor* gradPos = NULL;
        TensorShape gradPosShape;
        if (p.instance_mode)
            gradPosShape.AddDim(p.n);
        gradPosShape.AddDim(p.numVertices);
        gradPosShape.AddDim(4);
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, gradPosShape, &gradPos));
        p.gradPos = gradPos->flat<float>().data();

        // Initialize all the stuff.
        OP_CHECK_CUDA_ERROR(ctx, cudaMemsetAsync(&p.workBuffer[0].y, 0, sizeof(int), stream)); // Gradient kernel work counter.
        OP_CHECK_CUDA_ERROR(ctx, cudaMemcpyAsync(p.gradColor, p.dy, p.n * p.height * p.width * p.channels * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        OP_CHECK_CUDA_ERROR(ctx, cudaMemsetAsync(p.gradPos, 0, (p.instance_mode ? p.n : 1) * p.numVertices * 4 * sizeof(float), stream));

        // Verify that buffers are aligned to allow float2/float4 operations.
        OP_REQUIRES(ctx, !((uintptr_t)p.pos        & 15), errors::Internal("pos input tensor not aligned to float4"));
        OP_REQUIRES(ctx, !((uintptr_t)p.workBuffer & 15), errors::Internal("work_buffer internal tensor not aligned to int4"));

        // Launch the gradient kernel.
        void* args[] = {&p};

        int device = 0;
        int numCTA = 0;
        int numSM  = 0;
        OP_CHECK_CUDA_ERROR(ctx, cudaGetDevice(&device));
        OP_CHECK_CUDA_ERROR(ctx, cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numCTA, (void*)AntialiasGradKernel, AA_GRAD_KERNEL_THREADS_PER_BLOCK, 0));
        OP_CHECK_CUDA_ERROR(ctx, cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, device));
        OP_CHECK_CUDA_ERROR(ctx, cudaLaunchKernel((void*)AntialiasGradKernel, numCTA * numSM, AA_GRAD_KERNEL_THREADS_PER_BLOCK, args, 0, stream));
    }
};

REGISTER_OP("AntialiasGrad")
    .Input      ("color: float")
    .Input      ("raster_out: float")
    .Input      ("pos: float")
    .Input      ("tri: int32")
    .Input      ("dy: float")
    .Input      ("work_buffer: int32")
    .Output     ("grad_color: float")
    .Output     ("grad_pos: float");

REGISTER_KERNEL_BUILDER(Name("AntialiasGrad").Device(DEVICE_GPU), AntialiasGradOp);

//------------------------------------------------------------------------
