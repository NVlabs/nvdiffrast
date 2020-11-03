// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

//------------------------------------------------------------------------
// Common op attribute parser.

static __host__ void interpolateParseOpAttributes(OpKernelConstruction* ctx, InterpolateKernelParams& p, bool enableDA)
{
    if (enableDA)
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("diff_attrs_all", &p.diff_attrs_all));
        if (!p.diff_attrs_all)
        {
            std::vector<int> diff_attrs_vec;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("diff_attrs", &diff_attrs_vec));
            OP_REQUIRES(ctx, diff_attrs_vec.size() > 0, errors::InvalidArgument("differentiation enabled with empty diff_attrs list"));
            OP_REQUIRES(ctx, diff_attrs_vec.size() <= IP_MAX_DIFF_ATTRS, errors::InvalidArgument("too many entries in diff_attrs list (increase IP_MAX_DIFF_ATTRS)"));
            p.numDiffAttr = diff_attrs_vec.size();
            memcpy(p.diffAttrs, &diff_attrs_vec[0], diff_attrs_vec.size()*sizeof(int));
        }
    }
}

//------------------------------------------------------------------------
// Forward TensorFlow op.

template <bool ENABLE_DA>
struct InterpolateFwdOp : public OpKernel
{
    InterpolateKernelParams m_attribs;

    InterpolateFwdOp(OpKernelConstruction* ctx): OpKernel(ctx)
    {
        memset(&m_attribs, 0, sizeof(m_attribs));
        interpolateParseOpAttributes(ctx, m_attribs, ENABLE_DA);
    }

    void Compute(OpKernelContext* ctx)
    {
        InterpolateKernelParams& p = m_attribs;
        cudaStream_t stream = ctx->eigen_device<Eigen::GpuDevice>().stream();

        // Get input.
        const Tensor& attr    = ctx->input(0);
        const Tensor& rast    = ctx->input(1);
        const Tensor& tri     = ctx->input(2);
        const Tensor& rast_db = ctx->input(ENABLE_DA ? 3 : 2);

        // Instance rendering mode?
        p.instance_mode = attr.dims() > 2;

        // Extract input dimensions.
        if (p.instance_mode)
        {
            p.numVertices  = (attr.dims() > 1) ? attr.dim_size(1) : 0;
            p.numAttr      = (attr.dims() > 2) ? attr.dim_size(2) : 0;
        }
        else
        {
            p.numVertices  = (attr.dims() > 0) ? attr.dim_size(0) : 0;
            p.numAttr      = (attr.dims() > 1) ? attr.dim_size(1) : 0;
        }
        p.numTriangles = (tri.dims() > 0) ? tri.dim_size(0) : 0;
        p.height       = (rast.dims() > 1) ? rast.dim_size(1) : 0;
        p.width        = (rast.dims() > 2) ? rast.dim_size(2) : 0;
        p.depth        = (rast.dims() > 0) ? rast.dim_size(0) : 0;

        // Sanity checks.
        OP_REQUIRES(ctx, rast.dims() == 4 && rast.dim_size(0) > 0 && rast.dim_size(1) > 0 && rast.dim_size(2) > 0 && rast.dim_size(3) == 4, errors::InvalidArgument("rast must have shape[>0, >0, >0, 4]"));
        OP_REQUIRES(ctx, tri.dims() == 2 && tri.dim_size(0) > 0 && tri.dim_size(1) == 3, errors::InvalidArgument("tri must have shape [>0, 3]"));
        OP_REQUIRES(ctx, (attr.dims() == 2 || attr.dims() == 3) && attr.dim_size(0) > 0 && attr.dim_size(1) > 0 && (attr.dims() == 2 || attr.dim_size(2) > 0), errors::InvalidArgument("attr must have shape [>0, >0, >0] or [>0, >0]"));
        if (p.instance_mode)
            OP_REQUIRES(ctx, attr.dim_size(0) == p.depth || attr.dim_size(0) == 1, errors::InvalidArgument("minibatch size mismatch between inputs rast, attr"));
        if (ENABLE_DA)
        {
            OP_REQUIRES(ctx, rast_db.dims() == 4 && rast_db.dim_size(0) > 0 && rast_db.dim_size(1) > 0 && rast_db.dim_size(2) > 0 && rast_db.dim_size(3) == 4, errors::InvalidArgument("rast_db must have shape[>0, >0, >0, 4]"));
            OP_REQUIRES(ctx, rast_db.dim_size(1) == rast.dim_size(1) && rast_db.dim_size(2) == rast.dim_size(2), errors::InvalidArgument("spatial size mismatch between inputs rast and rast_db"));
            OP_REQUIRES(ctx, rast_db.dim_size(0) == p.depth, errors::InvalidArgument("minibatch size mismatch between inputs rast, rast_db"));
        }

        // All diff attrs mode.
        if (p.diff_attrs_all)
            p.numDiffAttr = p.numAttr;

        // Get input pointers.
        p.attr = attr.flat<float>().data();
        p.rast = rast.flat<float>().data();
        p.tri = tri.flat<int>().data();
        p.attrBC = (p.instance_mode && attr.dim_size(0) == 1) ? 1 : 0;
        p.rastDB = ENABLE_DA ? rast_db.flat<float>().data() : 0;

        // Allocate main output tensor.
        Tensor* out_tensor = NULL;
        TensorShape out_shape;
        out_shape.AddDim(p.depth);
        out_shape.AddDim(p.height);
        out_shape.AddDim(p.width);
        out_shape.AddDim(p.numAttr);
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out_tensor));
        p.out = out_tensor->flat<float>().data();

        // Allocate pixel differential output tensor.
        Tensor* out_da_tensor = NULL;
        out_shape.set_dim(3, p.numDiffAttr * 2);
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, out_shape, &out_da_tensor));
        p.outDA = ENABLE_DA ? out_da_tensor->flat<float>().data() : 0;

        // Verify that buffers are aligned to allow float2/float4 operations.
        OP_REQUIRES(ctx, !((uintptr_t)p.rast   & 15), errors::Internal("rast input tensor not aligned to float4"));
        OP_REQUIRES(ctx, !((uintptr_t)p.rastDB & 15), errors::Internal("rast_db input tensor not aligned to float4"));        
        if (ENABLE_DA)
            OP_REQUIRES(ctx, !((uintptr_t)p.outDA & 7), errors::Internal("out_da output tensor not aligned to float2"));

        // Choose launch parameters.
        dim3 blockSize = getLaunchBlockSize(IP_FWD_MAX_KERNEL_BLOCK_WIDTH, IP_FWD_MAX_KERNEL_BLOCK_HEIGHT, p.width, p.height);
        dim3 gridSize  = getLaunchGridSize(blockSize, p.width, p.height, p.depth);

        // Launch CUDA kernel.
        void* args[] = {&p};
        void* func = ENABLE_DA ? (void*)InterpolateFwdKernelDa : (void*)InterpolateFwdKernel;
        OP_CHECK_CUDA_ERROR(ctx, cudaLaunchKernel(func, gridSize, blockSize, args, 0, stream));
    }
};

REGISTER_OP("InterpolateFwd")
    .Input      ("attr: float")
    .Input      ("rast: float")
    .Input      ("tri: int32")
    .Output     ("out: float")
    .Output     ("out_da: float");

REGISTER_OP("InterpolateFwdDa")
    .Input      ("attr: float")
    .Input      ("rast: float")
    .Input      ("tri: int32")
    .Input      ("rast_db: float")
    .Output     ("out: float")
    .Output     ("out_da: float")
    .Attr       ("diff_attrs_all: int")
    .Attr       ("diff_attrs: list(int)");

REGISTER_KERNEL_BUILDER(Name("InterpolateFwd")  .Device(DEVICE_GPU), InterpolateFwdOp<false>);
REGISTER_KERNEL_BUILDER(Name("InterpolateFwdDa").Device(DEVICE_GPU), InterpolateFwdOp<true>);

//------------------------------------------------------------------------
// Gradient TensorFlow op.

template <bool ENABLE_DA>
struct InterpolateGradOp : public OpKernel
{
    InterpolateKernelParams m_attribs;

    InterpolateGradOp(OpKernelConstruction* ctx): OpKernel(ctx)
    {
        memset(&m_attribs, 0, sizeof(m_attribs));
        interpolateParseOpAttributes(ctx, m_attribs, ENABLE_DA);      
    }

    void Compute(OpKernelContext* ctx)
    {
        InterpolateKernelParams& p = m_attribs;
        cudaStream_t stream = ctx->eigen_device<Eigen::GpuDevice>().stream();

        // Get input.
        const Tensor& attr    = ctx->input(0);
        const Tensor& rast    = ctx->input(1);
        const Tensor& tri     = ctx->input(2);
        const Tensor& dy      = ctx->input(3);
        const Tensor& rast_db = ctx->input(ENABLE_DA ? 4 : 3);
        const Tensor& dda     = ctx->input(ENABLE_DA ? 5 : 3);

        // Instance rendering mode?
        p.instance_mode = attr.dims() > 2;

        // Extract input dimensions.
        if (p.instance_mode)
        {
            p.numVertices  = (attr.dims() > 1) ? attr.dim_size(1) : 0;
            p.numAttr      = (attr.dims() > 2) ? attr.dim_size(2) : 0;
        }
        else
        {
            p.numVertices  = (attr.dims() > 0) ? attr.dim_size(0) : 0;
            p.numAttr      = (attr.dims() > 1) ? attr.dim_size(1) : 0;
        }
        p.numTriangles = (tri.dims() > 0) ? tri.dim_size(0) : 0;
        p.depth        = (rast.dims() > 0) ? rast.dim_size(0) : 0;
        p.height       = (rast.dims() > 1) ? rast.dim_size(1) : 0;
        p.width        = (rast.dims() > 2) ? rast.dim_size(2) : 0;
        int attr_depth = p.instance_mode ? (attr.dims() > 1 ? attr.dim_size(0) : 0) : 1;

        // Sanity checks.
        OP_REQUIRES(ctx, rast.dims() == 4 && rast.dim_size(0) > 0 && rast.dim_size(1) > 0 && rast.dim_size(2) > 0 && rast.dim_size(3) == 4, errors::InvalidArgument("rast must have shape[>0, >0, >0, 4]"));
        OP_REQUIRES(ctx, tri.dims() == 2 && tri.dim_size(0) > 0 && tri.dim_size(1) == 3, errors::InvalidArgument("tri must have shape [>0, 3]"));
        OP_REQUIRES(ctx, (attr.dims() == 2 || attr.dims() == 3) && attr.dim_size(0) > 0 && attr.dim_size(1) > 0 && (attr.dims() == 2 || attr.dim_size(2) > 0), errors::InvalidArgument("attr must have shape [>0, >0, >0] or [>0, >0]"));
        OP_REQUIRES(ctx, dy.dims() == 4 && dy.dim_size(0) > 0 && dy.dim_size(1) == p.height && dy.dim_size(2) == p.width && dy.dim_size(3) > 0, errors::InvalidArgument("dy must have shape [>0, height, width, >0]"));
        OP_REQUIRES(ctx, dy.dim_size(3) == p.numAttr, errors::InvalidArgument("argument count mismatch between inputs dy, attr"));
        OP_REQUIRES(ctx, (attr_depth == p.depth || attr_depth == 1) && dy.dim_size(0) == p.depth, errors::InvalidArgument("minibatch size mismatch between inputs rast, dy, attr"));
        if (ENABLE_DA)
        {
            OP_REQUIRES(ctx, dda.dims() == 4 && dda.dim_size(0) > 0 && dda.dim_size(1) == p.height && dda.dim_size(2) == p.width, errors::InvalidArgument("dda must have shape [>0, height, width, ?]"));
            OP_REQUIRES(ctx, dda.dim_size(0) == p.depth, errors::InvalidArgument("minibatch size mismatch between rast, dda"));
        }

        // All diff attrs mode.
        if (p.diff_attrs_all)
            p.numDiffAttr = p.numAttr;

        // Get input pointers.
        p.attr   = attr.flat<float>().data();
        p.rast   = rast.flat<float>().data();
        p.tri    = tri.flat<int>().data();
        p.dy     = dy.flat<float>().data();
        p.rastDB = ENABLE_DA ? rast_db.flat<float>().data() : 0;
        p.dda    = ENABLE_DA ? dda.flat<float>().data() : 0;
        p.attrBC = (p.instance_mode && attr_depth < p.depth) ? 1 : 0;

        // Allocate attribute gradient output tensor.
        Tensor* grad_attr_tensor = NULL;
        TensorShape grad_attr_shape;
        if (p.instance_mode)
            grad_attr_shape.AddDim(attr_depth);
        grad_attr_shape.AddDim(p.numVertices);
        grad_attr_shape.AddDim(p.numAttr);
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, grad_attr_shape, &grad_attr_tensor));
        p.gradAttr = grad_attr_tensor->flat<float>().data();

        // Allocate bary gradient output tensor.
        Tensor* grad_rast_tensor = NULL;
        TensorShape grad_rast_shape;
        grad_rast_shape.AddDim(p.depth);
        grad_rast_shape.AddDim(p.height);
        grad_rast_shape.AddDim(p.width);
        grad_rast_shape.AddDim(4);
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, grad_rast_shape, &grad_rast_tensor));
        p.gradRaster = grad_rast_tensor->flat<float>().data();

        // Allocate bary pixel diff gradient output tensor.
        if (ENABLE_DA)
        {
            Tensor* grad_rast_db_tensor = NULL;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(2, grad_rast_shape, &grad_rast_db_tensor));
            p.gradRasterDB = grad_rast_db_tensor->flat<float>().data();
        }
        
        // Clear attribute gradients.
        cudaMemsetAsync(p.gradAttr, 0, attr_depth * p.numVertices * p.numAttr * sizeof(float), stream);

        // Verify that buffers are aligned to allow float2/float4 operations.
        OP_REQUIRES(ctx, !((uintptr_t)p.rast   & 15), errors::Internal("rast input tensor not aligned to float4"));
        OP_REQUIRES(ctx, !((uintptr_t)p.gradRaster & 15), errors::Internal("grad_rast output tensor not aligned to float4"));
        if (ENABLE_DA)
        {
            OP_REQUIRES(ctx, !((uintptr_t)p.dda & 7), errors::Internal("dda input tensor not aligned to float2"));
            OP_REQUIRES(ctx, !((uintptr_t)p.rastDB & 15), errors::Internal("rast_db input tensor not aligned to float4"));        
            OP_REQUIRES(ctx, !((uintptr_t)p.gradRasterDB & 15), errors::Internal("grad_rast_db output tensor not aligned to float4"));
        }
    
        // Choose launch parameters.
        dim3 blockSize = getLaunchBlockSize(IP_GRAD_MAX_KERNEL_BLOCK_WIDTH, IP_GRAD_MAX_KERNEL_BLOCK_HEIGHT, p.width, p.height);
        dim3 gridSize  = getLaunchGridSize(blockSize, p.width, p.height, p.depth);

        // Launch CUDA kernel.
        void* args[] = {&p};
        void* func = ENABLE_DA ? (void*)InterpolateGradKernelDa : (void*)InterpolateGradKernel;
        OP_CHECK_CUDA_ERROR(ctx, cudaLaunchKernel(func, gridSize, blockSize, args, 0, stream));
    }
};

REGISTER_OP("InterpolateGrad")
    .Input      ("attr: float")
    .Input      ("rast: float")
    .Input      ("tri: int32")
    .Input      ("dy: float")
    .Output     ("grad_attr: float")
    .Output     ("grad_rast: float")
    ;

REGISTER_OP("InterpolateGradDa")
    .Input      ("attr: float")
    .Input      ("rast: float")
    .Input      ("tri: int32")
    .Input      ("dy: float")
    .Input      ("rast_db: float")
    .Input      ("dda: float")
    .Output     ("grad_attr: float")
    .Output     ("grad_rast: float")
    .Output     ("grad_rast_db: float")
    .Attr       ("diff_attrs_all: int")
    .Attr       ("diff_attrs: list(int)");
    ;

REGISTER_KERNEL_BUILDER(Name("InterpolateGrad")  .Device(DEVICE_GPU), InterpolateGradOp<false>);
REGISTER_KERNEL_BUILDER(Name("InterpolateGradDa").Device(DEVICE_GPU), InterpolateGradOp<true>);

//------------------------------------------------------------------------
