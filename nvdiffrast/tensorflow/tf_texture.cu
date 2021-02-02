// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

//------------------------------------------------------------------------
// Common op attribute parser.

static __host__ void parseOpAttributes(OpKernelConstruction* ctx, TextureKernelParams& p)
{
    // Mip and filter modes.
    OP_REQUIRES_OK(ctx, ctx->GetAttr("filter_mode", &p.filterMode));
    OP_REQUIRES(ctx, p.filterMode >= 0 && p.filterMode < TEX_MODE_COUNT, errors::InvalidArgument("filter_mode unsupported"));
    p.enableMip = (p.filterMode == TEX_MODE_LINEAR_MIPMAP_NEAREST || p.filterMode == TEX_MODE_LINEAR_MIPMAP_LINEAR);

    // Mip level clamp.
    if (p.enableMip)
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("max_mip_level", &p.mipLevelLimit));
        OP_REQUIRES(ctx, p.mipLevelLimit >= -1, errors::InvalidArgument("invalid max_mip_level"));
        ctx->GetAttr("tex_const", &p.texConst); // Only available in forward op.
    }

    // Boundary mode.
    OP_REQUIRES_OK(ctx, ctx->GetAttr("boundary_mode", &p.boundaryMode));
    OP_REQUIRES(ctx, p.boundaryMode >= 0 && p.boundaryMode < TEX_BOUNDARY_MODE_COUNT, errors::InvalidArgument("boundary_mode unsupported"));
}

//------------------------------------------------------------------------
// Forward TensorFlow op.

struct TextureFwdOp : public OpKernel
{
    TextureKernelParams m_attribs;
    PersistentTensor    m_persistentMipTensor; // Used if texture is constant and mips are enabled.
    bool                m_persistentMipTensorInitialized;

    TextureFwdOp(OpKernelConstruction* ctx): OpKernel(ctx)
    {
        memset(&m_attribs, 0, sizeof(m_attribs));
        m_persistentMipTensorInitialized = false;
        parseOpAttributes(ctx, m_attribs);
    }

    void Compute(OpKernelContext* ctx)
    {
        TextureKernelParams& p = m_attribs;
        cudaStream_t stream = ctx->eigen_device<Eigen::GpuDevice>().stream();
        bool cube_mode = (p.boundaryMode == TEX_BOUNDARY_MODE_CUBE);

        // Get input.
        const Tensor& tex   = ctx->input(0);
        const Tensor& uv    = ctx->input(1);
        const Tensor& uv_da = ctx->input(p.enableMip ? 2 : 1);

        // Extract input dimensions.
        p.n         = (uv.dims() > 0) ? uv.dim_size(0) : 0;
        p.imgHeight = (uv.dims() > 1) ? uv.dim_size(1) : 0;
        p.imgWidth  = (uv.dims() > 2) ? uv.dim_size(2) : 0;
        p.texDepth  = (tex.dims() > 0) ? tex.dim_size(0) : 0;
        if (!cube_mode)
        {
            p.texHeight = (tex.dims() > 1) ? tex.dim_size(1) : 0;
            p.texWidth  = (tex.dims() > 2) ? tex.dim_size(2) : 0;
            p.channels  = (tex.dims() > 3) ? tex.dim_size(3) : 0;
        }
        else
        {
            p.texHeight = (tex.dims() > 2) ? tex.dim_size(2) : 0;
            p.texWidth  = (tex.dims() > 3) ? tex.dim_size(3) : 0;
            p.channels  = (tex.dims() > 4) ? tex.dim_size(4) : 0;
        }

        // Sanity checks.
        if (!cube_mode)
        {
            OP_REQUIRES(ctx, tex.dims() == 4 && tex.dim_size(0) > 0 && tex.dim_size(1) > 0 && tex.dim_size(2) > 0 && tex.dim_size(3) > 0, errors::InvalidArgument("tex must have shape[>0, >0, >0, >0]"));
            OP_REQUIRES(ctx, uv.dims() == 4 && uv.dim_size(0) > 0 && uv.dim_size(1) > 0 && uv.dim_size(2) > 0 && uv.dim_size(3) == 2, errors::InvalidArgument("uv must have shape [>0, >0, >0, 2]"));
        }
        else
        {
            OP_REQUIRES(ctx, tex.dims() == 5 && tex.dim_size(0) > 0 && tex.dim_size(1) == 6 && tex.dim_size(2) > 0 && tex.dim_size(3) > 0 && tex.dim_size(4) > 0, errors::InvalidArgument("tex must have shape[>0, 6, >0, >0, >0] in cube map mode"));
            OP_REQUIRES(ctx, uv.dims() == 4 && uv.dim_size(0) > 0 && uv.dim_size(1) > 0 && uv.dim_size(2) > 0 && uv.dim_size(3) == 3, errors::InvalidArgument("uv must have shape [>0, >0, >0, 3] in cube map mode"));
            OP_REQUIRES(ctx, tex.dim_size(2) == tex.dim_size(3), errors::InvalidArgument("texture shape must be square in cube map mode"));
        }
        OP_REQUIRES(ctx, tex.dim_size(0) == 1 || tex.dim_size(0) == p.n, errors::InvalidArgument("minibatch size mismatch between inputs tex, uv"));
        OP_REQUIRES(ctx, p.texWidth <= (1 << TEX_MAX_MIP_LEVEL) && p.texHeight <= (1 << TEX_MAX_MIP_LEVEL), errors::InvalidArgument("texture size too large"));
        if (p.enableMip)
        {
            if (!cube_mode)
                OP_REQUIRES(ctx, uv_da.dims() == 4 && uv_da.dim_size(0) == p.n && uv_da.dim_size(1) == p.imgHeight && uv_da.dim_size(2) == p.imgWidth && uv_da.dim_size(3) == 4, errors::InvalidArgument("uv_da must have shape [minibatch_size, height, width, 4]"));
            else
                OP_REQUIRES(ctx, uv_da.dims() == 4 && uv_da.dim_size(0) == p.n && uv_da.dim_size(1) == p.imgHeight && uv_da.dim_size(2) == p.imgWidth && uv_da.dim_size(3) == 6, errors::InvalidArgument("uv_da must have shape [minibatch_size, height, width, 6] in cube map mode"));
        }

        // Get input pointers.
        p.tex[0] = tex.flat<float>().data();
        p.uv = uv.flat<float>().data();
        p.uvDA = p.enableMip ? uv_da.flat<float>().data() : 0;

        // Allocate output tensor.
        Tensor* out_tensor = NULL;
        TensorShape out_shape;
        out_shape.AddDim(p.n);
        out_shape.AddDim(p.imgHeight);
        out_shape.AddDim(p.imgWidth);
        out_shape.AddDim(p.channels);
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out_tensor));
        p.out = out_tensor->flat<float>().data();

        // Choose kernel variants based on channel count.
        void* args[] = {&p};
        int channel_div_idx = 0;
        if (!(p.channels & 3))
            channel_div_idx = 2;  // Channel count divisible by 4.
        else if (!(p.channels & 1))
            channel_div_idx = 1;  // Channel count divisible by 2.

        // Mip-related setup.
        float* pmip = 0;
        if (p.enableMip)
        {
            // Generate mip offsets.
            int mipOffsets[TEX_MAX_MIP_LEVEL];
            int mipTotal = calculateMipInfo(ctx, p, mipOffsets);

            // Mip output tensor.
            Tensor* mip_tensor = NULL;
            TensorShape mip_shape;
            mip_shape.AddDim(mipTotal);

            // If texture is constant, calculate mip stack only once.
            bool computeMip = true;
            if (p.texConst)
            {
                // First execution?
                if (!m_persistentMipTensorInitialized)
                {
                    // Allocate a persistent mip tensor.
                    OP_REQUIRES_OK(ctx, ctx->allocate_persistent(DT_FLOAT, mip_shape, &m_persistentMipTensor, &mip_tensor));
                    m_persistentMipTensorInitialized = true;
                }
                else
                {
                    // Reuse the persistent tensor, do not recompute mip levels.
                    mip_tensor = m_persistentMipTensor.AccessTensor(ctx);
                    computeMip = false;
                }

                // Set as output tensor as well.
                ctx->set_output(1, *mip_tensor);
            }
            else
            {
                // Allocate an output tensor as usual.
                OP_REQUIRES_OK(ctx, ctx->allocate_output(1, mip_shape, &mip_tensor));
            }

            pmip = mip_tensor->flat<float>().data(); // Pointer to data.
            for (int i=1; i <= p.mipLevelMax; i++)
                p.tex[i] = pmip + mipOffsets[i]; // Pointers to mip levels.

            // Build mip levels if needed.
            if (computeMip)
            {
                for (int i=1; i <= p.mipLevelMax; i++)
                {
                    int2 ms = mipLevelSize(p, i);
                    int3 sz = make_int3(ms.x, ms.y, p.texDepth);
                    dim3 blockSize = getLaunchBlockSize(TEX_FWD_MAX_MIP_KERNEL_BLOCK_WIDTH, TEX_FWD_MAX_MIP_KERNEL_BLOCK_HEIGHT, sz.x, sz.y);
                    dim3 gridSize  = getLaunchGridSize(blockSize, sz.x, sz.y, sz.z * (cube_mode ? 6 : 1));
                    p.mipLevelOut = i;

                    void* build_func_tbl[3] = { (void*)MipBuildKernel1, (void*)MipBuildKernel2, (void*)MipBuildKernel4 };
                    OP_CHECK_CUDA_ERROR(ctx, cudaLaunchKernel(build_func_tbl[channel_div_idx], gridSize, blockSize, args, 0, stream));
                }
            }
        }

        // Verify that buffers are aligned to allow float2/float4 operations. Unused pointers are zero so always aligned.
        if (!cube_mode)
            OP_REQUIRES(ctx, !((uintptr_t)p.uv & 7), errors::Internal("uv input tensor not aligned to float2"));
        if ((p.channels & 3) == 0)
        {
            OP_REQUIRES(ctx, !((uintptr_t)p.tex[0] & 15), errors::Internal("tex input tensor not aligned to float4"));
            OP_REQUIRES(ctx, !((uintptr_t)p.out    & 15), errors::Internal("out output tensor not aligned to float4"));
            OP_REQUIRES(ctx, !((uintptr_t)pmip     & 15), errors::Internal("mip output tensor not aligned to float4"));
        }
        if ((p.channels & 1) == 0)
        {
            OP_REQUIRES(ctx, !((uintptr_t)p.tex[0] & 7), errors::Internal("tex input tensor not aligned to float2"));
            OP_REQUIRES(ctx, !((uintptr_t)p.out    & 7), errors::Internal("out output tensor not aligned to float2"));
            OP_REQUIRES(ctx, !((uintptr_t)pmip     & 7), errors::Internal("mip output tensor not aligned to float2"));
        }
        if (!cube_mode)
            OP_REQUIRES(ctx, !((uintptr_t)p.uvDA & 15), errors::Internal("uv_da input tensor not aligned to float4"));
        else
            OP_REQUIRES(ctx, !((uintptr_t)p.uvDA & 7), errors::Internal("uv_da input tensor not aligned to float2"));

        // Choose launch parameters for texture lookup kernel.
        dim3 blockSize = getLaunchBlockSize(TEX_FWD_MAX_KERNEL_BLOCK_WIDTH, TEX_FWD_MAX_KERNEL_BLOCK_HEIGHT, p.imgWidth, p.imgHeight);
        dim3 gridSize  = getLaunchGridSize(blockSize, p.imgWidth, p.imgHeight, p.n);

        // Choose kernel based on filter mode, cube mode, and datatype.
        void* func_tbl[TEX_MODE_COUNT * 3 * 2] = {
            (void*)TextureFwdKernelNearest1,
            (void*)TextureFwdKernelNearest2,
            (void*)TextureFwdKernelNearest4,
            (void*)TextureFwdKernelLinear1,
            (void*)TextureFwdKernelLinear2,
            (void*)TextureFwdKernelLinear4,
            (void*)TextureFwdKernelLinearMipmapNearest1,
            (void*)TextureFwdKernelLinearMipmapNearest2,
            (void*)TextureFwdKernelLinearMipmapNearest4,
            (void*)TextureFwdKernelLinearMipmapLinear1,
            (void*)TextureFwdKernelLinearMipmapLinear2,
            (void*)TextureFwdKernelLinearMipmapLinear4,
            (void*)TextureFwdKernelCubeNearest1,
            (void*)TextureFwdKernelCubeNearest2,
            (void*)TextureFwdKernelCubeNearest4,
            (void*)TextureFwdKernelCubeLinear1,
            (void*)TextureFwdKernelCubeLinear2,
            (void*)TextureFwdKernelCubeLinear4,
            (void*)TextureFwdKernelCubeLinearMipmapNearest1,
            (void*)TextureFwdKernelCubeLinearMipmapNearest2,
            (void*)TextureFwdKernelCubeLinearMipmapNearest4,
            (void*)TextureFwdKernelCubeLinearMipmapLinear1,
            (void*)TextureFwdKernelCubeLinearMipmapLinear2,
            (void*)TextureFwdKernelCubeLinearMipmapLinear4,
        };

        // Function index.
        int func_idx = p.filterMode;
        if (cube_mode)
            func_idx += TEX_MODE_COUNT;
        func_idx = func_idx * 3 + channel_div_idx;

        // Launch kernel.
        OP_CHECK_CUDA_ERROR(ctx, cudaLaunchKernel(func_tbl[func_idx], gridSize, blockSize, args, 0, stream));
    }
};

REGISTER_OP("TextureFwd")
    .Input      ("tex: float")
    .Input      ("uv: float")
    .Output     ("out: float")
    .Attr       ("filter_mode: int")
    .Attr       ("boundary_mode: int");

REGISTER_OP("TextureFwdMip")
    .Input      ("tex: float")
    .Input      ("uv: float")
    .Input      ("uv_da: float")
    .Output     ("out: float")
    .Output     ("mip: float")
    .Attr       ("filter_mode: int")
    .Attr       ("boundary_mode: int")
    .Attr       ("tex_const: int")
    .Attr       ("max_mip_level: int");

REGISTER_KERNEL_BUILDER(Name("TextureFwd")   .Device(DEVICE_GPU), TextureFwdOp);
REGISTER_KERNEL_BUILDER(Name("TextureFwdMip").Device(DEVICE_GPU), TextureFwdOp);

//------------------------------------------------------------------------
// Gradient TensorFlow op.

struct TextureGradOp : public OpKernel
{
    TextureKernelParams m_attribs;

    TextureGradOp(OpKernelConstruction* ctx): OpKernel(ctx)
    {
        memset(&m_attribs, 0, sizeof(m_attribs));
        parseOpAttributes(ctx, m_attribs);
    }

    void Compute(OpKernelContext* ctx)
    {
        TextureKernelParams& p = m_attribs;
        cudaStream_t stream = ctx->eigen_device<Eigen::GpuDevice>().stream();
        bool cube_mode = (p.boundaryMode == TEX_BOUNDARY_MODE_CUBE);

        // Get input.
        const Tensor& tex   = ctx->input(0);
        const Tensor& uv    = ctx->input(1);
        const Tensor& dy    = ctx->input(2);
        const Tensor& uv_da = ctx->input(p.enableMip ? 3 : 2);
        const Tensor& mip   = ctx->input(p.enableMip ? 4 : 2);

        // Extract input dimensions.
        p.n         = (uv.dims() > 0) ? uv.dim_size(0) : 0;
        p.imgHeight = (uv.dims() > 1) ? uv.dim_size(1) : 0;
        p.imgWidth  = (uv.dims() > 2) ? uv.dim_size(2) : 0;
        p.texDepth  = (tex.dims() > 0) ? tex.dim_size(0) : 0;
        if (!cube_mode)
        {
            p.texHeight = (tex.dims() > 1) ? tex.dim_size(1) : 0;
            p.texWidth  = (tex.dims() > 2) ? tex.dim_size(2) : 0;
            p.channels  = (tex.dims() > 3) ? tex.dim_size(3) : 0;
        }
        else
        {
            p.texHeight = (tex.dims() > 2) ? tex.dim_size(2) : 0;
            p.texWidth  = (tex.dims() > 3) ? tex.dim_size(3) : 0;
            p.channels  = (tex.dims() > 4) ? tex.dim_size(4) : 0;
        }

        // Sanity checks.
        if (!cube_mode)
        {
            OP_REQUIRES(ctx, tex.dims() == 4 && tex.dim_size(0) > 0 && tex.dim_size(1) > 0 && tex.dim_size(2) > 0 && tex.dim_size(3) > 0, errors::InvalidArgument("tex must have shape[>0, >0, >0, >0]"));
            OP_REQUIRES(ctx, uv.dims() == 4 && uv.dim_size(0) > 0 && uv.dim_size(1) > 0 && uv.dim_size(2) > 0 && uv.dim_size(3) == 2, errors::InvalidArgument("uv must have shape [>0, >0, >0, 2]"));
        }
        else
        {
            OP_REQUIRES(ctx, tex.dims() == 5 && tex.dim_size(0) > 0 && tex.dim_size(1) == 6 && tex.dim_size(2) > 0 && tex.dim_size(3) > 0 && tex.dim_size(4) > 0, errors::InvalidArgument("tex must have shape[>0, 6, >0, >0, >0] in cube map mode"));
            OP_REQUIRES(ctx, uv.dims() == 4 && uv.dim_size(0) > 0 && uv.dim_size(1) > 0 && uv.dim_size(2) > 0 && uv.dim_size(3) == 3, errors::InvalidArgument("uv must have shape [>0, >0, >0, 3] in cube map mode"));
            OP_REQUIRES(ctx, tex.dim_size(2) == tex.dim_size(3), errors::InvalidArgument("texture shape must be square in cube map mode"));
        }
        OP_REQUIRES(ctx, tex.dim_size(0) == 1 || tex.dim_size(0) == p.n, errors::InvalidArgument("minibatch size mismatch between inputs tex, uv"));
        OP_REQUIRES(ctx, dy.dims() == 4 && dy.dim_size(0) == p.n && dy.dim_size(1) == p.imgHeight && dy.dim_size(2) == p.imgWidth && dy.dim_size(3) == p.channels, errors::InvalidArgument("dy must have shape [minibatch_size, height, width, channels]"));
        if (p.enableMip)
        {
            if (!cube_mode)
                OP_REQUIRES(ctx, uv_da.dims() == 4 && uv_da.dim_size(0) == p.n && uv_da.dim_size(1) == p.imgHeight && uv_da.dim_size(2) == p.imgWidth && uv_da.dim_size(3) == 4, errors::InvalidArgument("uv_da must have shape [minibatch_size, height, width, 4]"));
            else
                OP_REQUIRES(ctx, uv_da.dims() == 4 && uv_da.dim_size(0) == p.n && uv_da.dim_size(1) == p.imgHeight && uv_da.dim_size(2) == p.imgWidth && uv_da.dim_size(3) == 6, errors::InvalidArgument("uv_da must have shape [minibatch_size, height, width, 6] in cube map mode"));
        }

        // Get input pointers.
        p.tex[0] = tex.flat<float>().data();
        p.uv = uv.flat<float>().data();
        p.dy = dy.flat<float>().data();
        p.uvDA = p.enableMip ? uv_da.flat<float>().data() : 0;
        float* pmip = p.enableMip ? (float*)mip.flat<float>().data() : 0;

        // Allocate output tensor for tex gradient.
        Tensor* grad_tex_tensor = NULL;
        TensorShape grad_tex_shape;
        grad_tex_shape.AddDim(p.texDepth);
        if (cube_mode)
            grad_tex_shape.AddDim(6);
        grad_tex_shape.AddDim(p.texHeight);
        grad_tex_shape.AddDim(p.texWidth);
        grad_tex_shape.AddDim(p.channels);
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, grad_tex_shape, &grad_tex_tensor));
        p.gradTex[0] = grad_tex_tensor->flat<float>().data();

        // Allocate output tensor for uv gradient.
        if (p.filterMode != TEX_MODE_NEAREST)
        {
            TensorShape grad_uv_shape;
            Tensor* grad_uv_tensor = NULL;
            grad_uv_shape.AddDim(p.n);
            grad_uv_shape.AddDim(p.imgHeight);
            grad_uv_shape.AddDim(p.imgWidth);
            grad_uv_shape.AddDim(uv.dim_size(3));
            OP_REQUIRES_OK(ctx, ctx->allocate_output(1, grad_uv_shape, &grad_uv_tensor));
            p.gradUV = grad_uv_tensor->flat<float>().data();

            // Allocate output tensor for uv_da gradient.
            if (p.filterMode == TEX_MODE_LINEAR_MIPMAP_LINEAR)
            {
                Tensor* grad_uv_da_tensor = NULL;
                grad_uv_shape.set_dim(3, uv_da.dim_size(3));
                OP_REQUIRES_OK(ctx, ctx->allocate_output(2, grad_uv_shape, &grad_uv_da_tensor));
                p.gradUVDA = grad_uv_da_tensor->flat<float>().data();
            }
        }

        // Choose kernel variants based on channel count.
        int channel_div_idx = 0;
        if (!(p.channels & 3))
            channel_div_idx = 2;  // Channel count divisible by 4.
        else if (!(p.channels & 1))
            channel_div_idx = 1;  // Channel count divisible by 2.

        // Mip-related setup.
        Tensor grad_mip_tensor;
        float* pgradMip = 0;
        if (p.enableMip)
        {
            // Generate mip offsets.
            int mipOffsets[TEX_MAX_MIP_LEVEL];
            int mipTotal = calculateMipInfo(ctx, p, mipOffsets);

            // Get space for temporary mip gradients.
            TensorShape grad_mip_shape;
            grad_mip_shape.AddDim(mipTotal);
            ctx->allocate_temp(DT_FLOAT, grad_mip_shape, &grad_mip_tensor);
            pgradMip = grad_mip_tensor.flat<float>().data();
            for (int i=1; i <= p.mipLevelMax; i++)
            {
                p.tex[i] = pmip + mipOffsets[i]; // Pointers to mip levels.
                p.gradTex[i] = pgradMip + mipOffsets[i]; // Pointers to mip gradients.
            }

            // Clear mip gradients.
            OP_CHECK_CUDA_ERROR(ctx, cudaMemsetAsync(pgradMip, 0, mipTotal * sizeof(float), stream));
        }

        // Initialize texture gradients to zero.
        int texBytes = p.texHeight * p.texWidth * p.texDepth * p.channels * sizeof(float);
        if (cube_mode)
            texBytes *= 6;
        OP_CHECK_CUDA_ERROR(ctx, cudaMemsetAsync(p.gradTex[0], 0, texBytes, stream));

        // Verify that buffers are aligned to allow float2/float4 operations. Unused pointers are zero so always aligned.
        if (!cube_mode)
        {
            OP_REQUIRES(ctx, !((uintptr_t)p.uv       & 7), errors::Internal("uv input tensor not aligned to float2"));
            OP_REQUIRES(ctx, !((uintptr_t)p.gradUV   & 7), errors::Internal("grad_uv output tensor not aligned to float2"));
            OP_REQUIRES(ctx, !((uintptr_t)p.uvDA     & 15), errors::Internal("uv_da input tensor not aligned to float4"));
            OP_REQUIRES(ctx, !((uintptr_t)p.gradUVDA & 15), errors::Internal("grad_uv_da output tensor not aligned to float4"));
        }
        else
        {
            OP_REQUIRES(ctx, !((uintptr_t)p.uvDA     & 7), errors::Internal("uv_da input tensor not aligned to float2"));
            OP_REQUIRES(ctx, !((uintptr_t)p.gradUVDA & 7), errors::Internal("grad_uv_da output tensor not aligned to float2"));
        }
        if ((p.channels & 3) == 0)
        {
            OP_REQUIRES(ctx, !((uintptr_t)p.tex[0]     & 15), errors::Internal("tex input tensor not aligned to float4"));
            OP_REQUIRES(ctx, !((uintptr_t)p.gradTex[0] & 15), errors::Internal("grad_tex output tensor not aligned to float4"));
            OP_REQUIRES(ctx, !((uintptr_t)p.dy         & 15), errors::Internal("dy input tensor not aligned to float4"));
            OP_REQUIRES(ctx, !((uintptr_t)pmip         & 15), errors::Internal("mip input tensor not aligned to float4"));
            OP_REQUIRES(ctx, !((uintptr_t)pgradMip     & 15), errors::Internal("internal mip gradient tensor not aligned to float4"));
        }
        if ((p.channels & 1) == 0)
        {
            OP_REQUIRES(ctx, !((uintptr_t)p.tex[0]     & 7), errors::Internal("tex input tensor not aligned to float2"));
            OP_REQUIRES(ctx, !((uintptr_t)p.gradTex[0] & 7), errors::Internal("grad_tex output tensor not aligned to float2"));
            OP_REQUIRES(ctx, !((uintptr_t)p.dy         & 7), errors::Internal("dy output tensor not aligned to float2"));
            OP_REQUIRES(ctx, !((uintptr_t)pmip         & 7), errors::Internal("mip input tensor not aligned to float2"));
            OP_REQUIRES(ctx, !((uintptr_t)pgradMip     & 7), errors::Internal("internal mip gradient tensor not aligned to float2"));
        }

        // Choose launch parameters for main gradient kernel.
        void* args[] = {&p};
        dim3 blockSize = getLaunchBlockSize(TEX_GRAD_MAX_KERNEL_BLOCK_WIDTH, TEX_GRAD_MAX_KERNEL_BLOCK_HEIGHT, p.imgWidth, p.imgHeight);
        dim3 gridSize  = getLaunchGridSize(blockSize, p.imgWidth, p.imgHeight, p.n);

        void* func_tbl[TEX_MODE_COUNT * 2] = {
            (void*)TextureGradKernelNearest,
            (void*)TextureGradKernelLinear,
            (void*)TextureGradKernelLinearMipmapNearest,
            (void*)TextureGradKernelLinearMipmapLinear,
            (void*)TextureGradKernelCubeNearest,
            (void*)TextureGradKernelCubeLinear,
            (void*)TextureGradKernelCubeLinearMipmapNearest,
            (void*)TextureGradKernelCubeLinearMipmapLinear,
        };

        // Function index.
        int func_idx = p.filterMode;
        if (cube_mode)
            func_idx += TEX_MODE_COUNT;

        // Launch main gradient kernel.
        OP_CHECK_CUDA_ERROR(ctx, cudaLaunchKernel(func_tbl[func_idx], gridSize, blockSize, args, 0, stream));

        // Launch kernel to pull gradients from mip levels.
        if (p.enableMip)
        {
            dim3 blockSize = getLaunchBlockSize(TEX_GRAD_MAX_MIP_KERNEL_BLOCK_WIDTH, TEX_GRAD_MAX_MIP_KERNEL_BLOCK_HEIGHT, p.texWidth, p.texHeight);
            dim3 gridSize  = getLaunchGridSize(blockSize, p.texWidth, p.texHeight, p.texDepth * (cube_mode ? 6 : 1));
            int sharedBytes = blockSize.x * blockSize.y * p.channels * sizeof(float);

            void* mip_grad_func_tbl[3] = { (void*)MipGradKernel1, (void*)MipGradKernel2, (void*)MipGradKernel4 };
            OP_CHECK_CUDA_ERROR(ctx, cudaLaunchKernel(mip_grad_func_tbl[channel_div_idx], gridSize, blockSize, args, sharedBytes, stream));
        }
    }
};

REGISTER_OP("TextureGradNearest")
    .Input      ("tex: float")
    .Input      ("uv: float")
    .Input      ("dy: float")
    .Output     ("grad_tex: float")
    .Attr       ("filter_mode: int")
    .Attr       ("boundary_mode: int");

REGISTER_OP("TextureGradLinear")
    .Input      ("tex: float")
    .Input      ("uv: float")
    .Input      ("dy: float")
    .Output     ("grad_tex: float")
    .Output     ("grad_uv: float")
    .Attr       ("filter_mode: int")
    .Attr       ("boundary_mode: int");

REGISTER_OP("TextureGradLinearMipmapNearest")
    .Input      ("tex: float")
    .Input      ("uv: float")
    .Input      ("dy: float")
    .Input      ("uv_da: float")
    .Input      ("mip: float")
    .Output     ("grad_tex: float")
    .Output     ("grad_uv: float")
    .Attr       ("filter_mode: int")
    .Attr       ("boundary_mode: int")
    .Attr       ("max_mip_level: int");
    
REGISTER_OP("TextureGradLinearMipmapLinear")
    .Input      ("tex: float")
    .Input      ("uv: float")
    .Input      ("dy: float")
    .Input      ("uv_da: float")
    .Input      ("mip: float")
    .Output     ("grad_tex: float")
    .Output     ("grad_uv: float")
    .Output     ("grad_uv_da: float")
    .Attr       ("filter_mode: int")
    .Attr       ("boundary_mode: int")
    .Attr       ("max_mip_level: int");
    
REGISTER_KERNEL_BUILDER(Name("TextureGradNearest")            .Device(DEVICE_GPU), TextureGradOp);
REGISTER_KERNEL_BUILDER(Name("TextureGradLinear")             .Device(DEVICE_GPU), TextureGradOp);
REGISTER_KERNEL_BUILDER(Name("TextureGradLinearMipmapNearest").Device(DEVICE_GPU), TextureGradOp);
REGISTER_KERNEL_BUILDER(Name("TextureGradLinearMipmapLinear") .Device(DEVICE_GPU), TextureGradOp);
        
//------------------------------------------------------------------------
