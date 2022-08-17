// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

//------------------------------------------------------------------------
// Forward TensorFlow op.

struct RasterizeFwdOp : public OpKernel
{
    RasterizeGLState        m_glState;              // OpenGL-related persistent state.
    int                     m_tri_const;            // 1 if triangle array is known to be constant.

    RasterizeFwdOp(OpKernelConstruction* ctx):
        OpKernel(ctx)
    {
        memset(&m_glState, 0, sizeof(RasterizeGLState));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("enable_db", &m_glState.enableDB));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("tri_const", &m_tri_const));
    }

    void Compute(OpKernelContext* ctx)
    {
        cudaStream_t stream = ctx->eigen_device<Eigen::GpuDevice>().stream();

        // Check that input shapes are correct.
        const Tensor& pos = ctx->input(0);
        const Tensor& tri = ctx->input(1);
        const Tensor& resolution = ctx->input(2);
        const Tensor& ranges = ctx->input(3);

        // Determine number of outputs
        int num_outputs = m_glState.enableDB ? 2 : 1;

        // Determine instance mode and check input dimensions.
        bool instance_mode = pos.dims() > 2;
        if (instance_mode)
        {
            OP_REQUIRES(ctx, pos.dims() == 3 && pos.dim_size(0) > 0 && pos.dim_size(1) > 0 && pos.dim_size(2) == 4, errors::InvalidArgument("instance mode - pos must have shape [>0, >0, 4]"));
            OP_REQUIRES(ctx, tri.dims() == 2 && tri.dim_size(0) > 0 && tri.dim_size(1) == 3, errors::InvalidArgument("tri must have shape [>0, 3]"));
            OP_REQUIRES(ctx, resolution.dims() == 1 && resolution.dim_size(0) == 2, errors::InvalidArgument("resolution must have shape [2]"));
        }
        else
        {
            OP_REQUIRES(ctx, pos.dims() == 2 && pos.dim_size(0) > 0 && pos.dim_size(1) == 4, errors::InvalidArgument("range mode - pos must have shape [>0, 4]"));
            OP_REQUIRES(ctx, tri.dims() == 2 && tri.dim_size(0) > 0 && tri.dim_size(1) == 3, errors::InvalidArgument("tri must have shape [>0, 3]"));
            OP_REQUIRES(ctx, resolution.dims() == 1 && resolution.dim_size(0) == 2, errors::InvalidArgument("resolution must have shape [2]"));
            OP_REQUIRES(ctx, ranges.dims() == 2 && ranges.dim_size(0) > 0 && ranges.dim_size(1) == 2, errors::InvalidArgument("range mode - ranges must have shape [>0, 2]"));
        }

        // Get output shape.
        const int32_t* res_in = resolution.flat<int32_t>().data(); // This is in CPU memory.
        int height = res_in[0];
        int width  = res_in[1];
        int depth  = instance_mode ? pos.dim_size(0) : ranges.dim_size(0);
        OP_REQUIRES(ctx, height > 0 && width > 0, errors::InvalidArgument("resolution must be [>0, >0]"));

        // Get position and triangle buffer sizes in int32/float32.
        int posCount = 4 * pos.dim_size(0) * (instance_mode ? pos.dim_size(1) : 1);
        int triCount = 3 * tri.dim_size(0);

        // Init context and GL?
        bool initCtx = !m_glState.glFBO;
        if (initCtx)
        {
            const DeviceBase::GpuDeviceInfo* g = ctx->device()->tensorflow_gpu_device_info();
            int cudaDeviceIdx = g ? g->gpu_id : -1;
            rasterizeInitGLContext(ctx, m_glState, cudaDeviceIdx); // In common/rasterize.cpp
        }
        else
            setGLContext(m_glState.glctx); // (Re-)Activate GL context.

        // Resize all buffers.
        bool changes = false;
        rasterizeResizeBuffers(ctx, m_glState, changes, posCount, triCount, width, height, depth); // In common/rasterize_gl.cpp
        if (changes)
        {
#ifdef _WIN32
            // Workaround for occasional blank first frame on Windows.
            releaseGLContext();
            setGLContext(m_glState.glctx);
#endif
        }

        // Copy input data to GL and render.
        const float* posPtr = pos.flat<float>().data();
        const int32_t* rangesPtr = instance_mode ? 0 : ranges.flat<int32_t>().data(); // This is in CPU memory.
        const int32_t* triPtr = (initCtx || !m_tri_const) ? tri.flat<int32_t>().data() : NULL; // Copy triangles only if needed.
        int vtxPerInstance = instance_mode ? pos.dim_size(1) : 0;
        rasterizeRender(ctx, m_glState, stream, posPtr, posCount, vtxPerInstance, triPtr, triCount, rangesPtr, width, height, depth, -1);

        // Allocate output tensors.
        TensorShape output_shape;
        output_shape.AddDim(depth);
        output_shape.AddDim(height);
        output_shape.AddDim(width);
        output_shape.AddDim(4);
        float* outputPtr[2];
        for (int i=0; i < 2; i++)
        {
            if (i >= num_outputs)
                output_shape.set_dim(3, 0); // Zero channels for unwanted out_db tensor.
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(i, output_shape, &output_tensor));
            if (i < num_outputs)
                outputPtr[i] = output_tensor->flat<float>().data();
        }

        // Copy rasterized results into CUDA buffers.
        rasterizeCopyResults(ctx, m_glState, stream, outputPtr, width, height, depth);

        // Done. Release GL context.
        releaseGLContext();
    }
};

REGISTER_OP("RasterizeFwd")
    .Input      ("pos: float")
    .Input      ("tri: int32")
    .Input      ("resolution: int32")
    .Input      ("ranges: int32")
    .Output     ("out: float")
    .Output     ("out_db: float")
    .Attr       ("enable_db: int")
    .Attr       ("tri_const: int");

REGISTER_KERNEL_BUILDER(Name("RasterizeFwd").Device(DEVICE_GPU).HostMemory("resolution").HostMemory("ranges"), RasterizeFwdOp);

//------------------------------------------------------------------------
// Gradient TensorFlow op.

template <bool ENABLE_DB>
struct RasterizeGradOp : public OpKernel
{
    RasterizeGradParams m_attribs;

    RasterizeGradOp(OpKernelConstruction* ctx): OpKernel(ctx)
    {
        memset(&m_attribs, 0, sizeof(m_attribs));
    }

    void Compute(OpKernelContext* ctx)
    {
        RasterizeGradParams& p = m_attribs;
        cudaStream_t stream = ctx->eigen_device<Eigen::GpuDevice>().stream();

        // Input tensors.
        const Tensor& pos = ctx->input(0);
        const Tensor& tri = ctx->input(1);
        const Tensor& out = ctx->input(2);
        const Tensor& dy  = ctx->input(3);
        const Tensor& ddb = ctx->input(ENABLE_DB ? 4 : 3);

        // Determine instance mode.
        p.instance_mode = (pos.dims() > 2) ? 1 : 0;

        // Shape is taken from the rasterizer output tensor.
        OP_REQUIRES(ctx, out.dims() == 4, errors::InvalidArgument("out must be rank-4"));
        p.depth  = out.dim_size(0);
        p.height = out.dim_size(1);
        p.width  = out.dim_size(2);
        OP_REQUIRES(ctx, p.depth > 0 && p.height > 0 && p.width > 0, errors::InvalidArgument("resolution must be [>0, >0, >0]"));

        // Check other shapes.
        if (p.instance_mode)
            OP_REQUIRES(ctx, pos.dims() == 3 && pos.dim_size(0) == p.depth && pos.dim_size(1) > 0 && pos.dim_size(2) == 4, errors::InvalidArgument("pos must have shape [depth, >0, 4]"));
        else
            OP_REQUIRES(ctx, pos.dims() == 2 && pos.dim_size(0) > 0 && pos.dim_size(1) == 4, errors::InvalidArgument("pos must have shape [>0, 4]"));
        OP_REQUIRES(ctx, tri.dims() == 2 && tri.dim_size(0) > 0 && tri.dim_size(1) == 3, errors::InvalidArgument("tri must have shape [>0, 3]"));
        OP_REQUIRES(ctx, out.dims() == 4 && out.dim_size(0) == p.depth && out.dim_size(1) == p.height && out.dim_size(2) == p.width && out.dim_size(3) == 4, errors::InvalidArgument("out must have shape [depth, height, width, 4]"));
        OP_REQUIRES(ctx,  dy.dims() == 4 &&  dy.dim_size(0) == p.depth &&  dy.dim_size(1) == p.height &&  dy.dim_size(2) == p.width &&  dy.dim_size(3) == 4, errors::InvalidArgument("dy must have shape [depth, height, width, 4]"));
        if (ENABLE_DB)
            OP_REQUIRES(ctx, ddb.dims() == 4 && ddb.dim_size(0) == p.depth && ddb.dim_size(1) == p.height && ddb.dim_size(2) == p.width && ddb.dim_size(3) == 4, errors::InvalidArgument("ddb must have shape [depth, height, width, 4]"));

        // Populate parameters.
        p.numTriangles = tri.dim_size(0);
        p.numVertices = p.instance_mode ? pos.dim_size(1) : pos.dim_size(0);
        p.pos = pos.flat<float>().data();
        p.tri = tri.flat<int>().data();
        p.out = out.flat<float>().data();
        p.dy  = dy.flat<float>().data();
        p.ddb = ENABLE_DB ? ddb.flat<float>().data() : 0;

        // Set up pixel position to clip space x, y transform.
        p.xs = 2.f / (float)p.width;
        p.xo = 1.f / (float)p.width - 1.f;
        p.ys = 2.f / (float)p.height;
        p.yo = 1.f / (float)p.height - 1.f;

        // Allocate output tensor for position gradients.
        Tensor* grad_tensor = NULL;
        TensorShape grad_shape;
        if (p.instance_mode)
            grad_shape.AddDim(p.depth);
        grad_shape.AddDim(p.numVertices);
        grad_shape.AddDim(4);
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, grad_shape, &grad_tensor));
        p.grad = grad_tensor->flat<float>().data();

        // Clear the output buffers.
        size_t gradBytes = (p.instance_mode ? p.depth : 1) * p.numVertices * 4 * sizeof(float);
        cudaMemsetAsync(p.grad, 0, gradBytes, stream);

        // Verify that buffers are aligned to allow float2/float4 operations.
        OP_REQUIRES(ctx, !((uintptr_t)p.pos & 15), errors::Internal("pos input tensor not aligned to float4"));
        OP_REQUIRES(ctx, !((uintptr_t)p.dy  &  7), errors::Internal("dy input tensor not aligned to float2"));
        if (ENABLE_DB)
            OP_REQUIRES(ctx, !((uintptr_t)p.ddb & 15), errors::Internal("ddb input tensor not aligned to float4"));

        // Choose launch parameters.
        dim3 blockSize = getLaunchBlockSize(RAST_GRAD_MAX_KERNEL_BLOCK_WIDTH, RAST_GRAD_MAX_KERNEL_BLOCK_HEIGHT, p.width, p.height);
        dim3 gridSize  = getLaunchGridSize(blockSize, p.width, p.height, p.depth);

        // Launch CUDA kernel.
        void* args[] = {&p};
        void* func = ENABLE_DB ? (void*)RasterizeGradKernelDb : (void*)RasterizeGradKernel;
        OP_CHECK_CUDA_ERROR(ctx, cudaLaunchKernel(func, gridSize, blockSize, args, 0, stream));
    }
};

REGISTER_OP("RasterizeGrad")
    .Input      ("pos: float")
    .Input      ("tri: int32")
    .Input      ("out: float")
    .Input      ("dy: float")
    .Output     ("grad: float");

REGISTER_OP("RasterizeGradDb")
    .Input      ("pos: float")
    .Input      ("tri: int32")
    .Input      ("out: float")
    .Input      ("dy: float")
    .Input      ("ddb: float")
    .Output     ("grad: float");

REGISTER_KERNEL_BUILDER(Name("RasterizeGrad")  .Device(DEVICE_GPU), RasterizeGradOp<false>);
REGISTER_KERNEL_BUILDER(Name("RasterizeGradDb").Device(DEVICE_GPU), RasterizeGradOp<true>);

//------------------------------------------------------------------------
