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
#include "../common/texture.h"
#include <cuda_runtime.h>

//------------------------------------------------------------------------
// Kernel prototypes.

void MipBuildKernel1                            (const TextureKernelParams p);
void MipBuildKernel2                            (const TextureKernelParams p);
void MipBuildKernel4                            (const TextureKernelParams p);
void TextureFwdKernelNearest1                   (const TextureKernelParams p);
void TextureFwdKernelNearest2                   (const TextureKernelParams p);
void TextureFwdKernelNearest4                   (const TextureKernelParams p);
void TextureFwdKernelLinear1                    (const TextureKernelParams p);
void TextureFwdKernelLinear2                    (const TextureKernelParams p);
void TextureFwdKernelLinear4                    (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapNearest1       (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapNearest2       (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapNearest4       (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapLinear1        (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapLinear2        (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapLinear4        (const TextureKernelParams p);
void TextureFwdKernelCubeNearest1               (const TextureKernelParams p);
void TextureFwdKernelCubeNearest2               (const TextureKernelParams p);
void TextureFwdKernelCubeNearest4               (const TextureKernelParams p);
void TextureFwdKernelCubeLinear1                (const TextureKernelParams p);
void TextureFwdKernelCubeLinear2                (const TextureKernelParams p);
void TextureFwdKernelCubeLinear4                (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapNearest1   (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapNearest2   (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapNearest4   (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapLinear1    (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapLinear2    (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapLinear4    (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapNearestBO1     (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapNearestBO2     (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapNearestBO4     (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapLinearBO1      (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapLinearBO2      (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapLinearBO4      (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapNearestBO1 (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapNearestBO2 (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapNearestBO4 (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapLinearBO1  (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapLinearBO2  (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapLinearBO4  (const TextureKernelParams p);
void MipGradKernel1                             (const TextureKernelParams p);
void MipGradKernel2                             (const TextureKernelParams p);
void MipGradKernel4                             (const TextureKernelParams p);
void TextureGradKernelNearest                   (const TextureKernelParams p);
void TextureGradKernelLinear                    (const TextureKernelParams p);
void TextureGradKernelLinearMipmapNearest       (const TextureKernelParams p);
void TextureGradKernelLinearMipmapLinear        (const TextureKernelParams p);
void TextureGradKernelCubeNearest               (const TextureKernelParams p);
void TextureGradKernelCubeLinear                (const TextureKernelParams p);
void TextureGradKernelCubeLinearMipmapNearest   (const TextureKernelParams p);
void TextureGradKernelCubeLinearMipmapLinear    (const TextureKernelParams p);
void TextureGradKernelLinearMipmapNearestBO     (const TextureKernelParams p);
void TextureGradKernelLinearMipmapLinearBO      (const TextureKernelParams p);
void TextureGradKernelCubeLinearMipmapNearestBO (const TextureKernelParams p);
void TextureGradKernelCubeLinearMipmapLinearBO  (const TextureKernelParams p);

//------------------------------------------------------------------------
// Modeselektor.

static void set_modes(TextureKernelParams& p, int filter_mode, int boundary_mode, int max_mip_level)
{
    // Mip and filter modes.
    p.filterMode = filter_mode;
    NVDR_CHECK(p.filterMode >= 0 && p.filterMode < TEX_MODE_COUNT, "filter_mode unsupported");
    p.enableMip = (p.filterMode == TEX_MODE_LINEAR_MIPMAP_NEAREST || p.filterMode == TEX_MODE_LINEAR_MIPMAP_LINEAR);

    // Mip level clamp.
    if (p.enableMip)
    {
        p.mipLevelLimit = max_mip_level;
        NVDR_CHECK(p.mipLevelLimit >= -1, "invalid max_mip_level");
    }

    // Boundary mode.
    p.boundaryMode = boundary_mode;
    NVDR_CHECK(p.boundaryMode >= 0 && p.boundaryMode < TEX_BOUNDARY_MODE_COUNT, "boundary_mode unsupported");
}

//------------------------------------------------------------------------
// Mipmap construction.

TextureMipWrapper texture_construct_mip(torch::Tensor tex, int max_mip_level, bool cube_mode)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(tex));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    TextureKernelParams p = {}; // Initialize all fields to zero.
    p.mipLevelLimit = max_mip_level;
    p.boundaryMode = cube_mode ? TEX_BOUNDARY_MODE_CUBE : TEX_BOUNDARY_MODE_WRAP;
    NVDR_CHECK(p.mipLevelLimit >= -1, "invalid max_mip_level");

    // Check inputs.
    NVDR_CHECK_DEVICE(tex);
    NVDR_CHECK_CONTIGUOUS(tex);
    NVDR_CHECK_F32(tex);

    // Populate parameters and sanity check tex shape.
    if (!cube_mode)
    {
        NVDR_CHECK(tex.sizes().size() == 4 && tex.size(0) > 0 && tex.size(1) > 0 && tex.size(2) > 0 && tex.size(3) > 0, "tex must have shape[>0, >0, >0, >0]");
    }
    else
    {
        NVDR_CHECK(tex.sizes().size() == 5 && tex.size(0) > 0 && tex.size(1) == 6 && tex.size(2) > 0 && tex.size(3) > 0 && tex.size(4) > 0, "tex must have shape[>0, 6, >0, >0, >0] in cube map mode");
        NVDR_CHECK(tex.size(2) == tex.size(3), "texture shape must be square in cube map mode");
    }
    p.texDepth  = tex.size(0);
    p.texHeight = tex.size(cube_mode ? 2 : 1);
    p.texWidth  = tex.size(cube_mode ? 3 : 2);
    p.channels  = tex.size(cube_mode ? 4 : 3);

    // Set texture pointer.
    p.tex = tex.data_ptr<float>();

    // Set mip offsets and calculate total size.
    int mipTotal = calculateMipInfo(NVDR_CTX_PARAMS, p);

    // Allocate and set mip tensor.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor mip = torch::empty({mipTotal}, opts);
    p.mip = mip.data_ptr<float>();

    // Choose kernel variants based on channel count.
    void* args[] = {&p};
    int channel_div_idx = 0;
    if (!(p.channels & 3))
        channel_div_idx = 2;  // Channel count divisible by 4.
    else if (!(p.channels & 1))
        channel_div_idx = 1;  // Channel count divisible by 2.

    // Build mip levels.
    for (int i=1; i <= p.mipLevelMax; i++)
    {
        int2 ms = mipLevelSize(p, i);
        int3 sz = make_int3(ms.x, ms.y, p.texDepth);
        dim3 blockSize = getLaunchBlockSize(TEX_FWD_MAX_MIP_KERNEL_BLOCK_WIDTH, TEX_FWD_MAX_MIP_KERNEL_BLOCK_HEIGHT, sz.x, sz.y);
        dim3 gridSize  = getLaunchGridSize(blockSize, sz.x, sz.y, sz.z * (cube_mode ? 6 : 1));
        p.mipLevelOut = i;

        void* build_func_tbl[3] = { (void*)MipBuildKernel1, (void*)MipBuildKernel2, (void*)MipBuildKernel4 };
        NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(build_func_tbl[channel_div_idx], gridSize, blockSize, args, 0, stream));
    }

    // Return the mip tensor in a wrapper.
    TextureMipWrapper mip_wrap;
    mip_wrap.mip = mip;
    mip_wrap.max_mip_level = max_mip_level;
    mip_wrap.texture_size = tex.sizes().vec();
    mip_wrap.cube_mode = cube_mode;
    return mip_wrap;
}

//------------------------------------------------------------------------
// Forward op.

torch::Tensor texture_fwd_mip(torch::Tensor tex, torch::Tensor uv, torch::Tensor uv_da, torch::Tensor mip_level_bias, TextureMipWrapper mip_wrap, int filter_mode, int boundary_mode)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(tex));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    TextureKernelParams p = {}; // Initialize all fields to zero.
    torch::Tensor& mip = mip_wrap.mip; // Unwrap.
    int max_mip_level = mip_wrap.max_mip_level;
    set_modes(p, filter_mode, boundary_mode, max_mip_level);

    // See if we have these tensors or not.
    bool has_uv_da = uv_da.defined() && uv_da.nbytes();
    bool has_mip_level_bias = mip_level_bias.defined() && mip_level_bias.nbytes();

    if (p.enableMip)
    {
        NVDR_CHECK(has_uv_da || has_mip_level_bias, "mipmapping filter mode requires uv_da and/or mip_level_bias input");
        NVDR_CHECK(mip.defined(), "mipmapping filter mode requires mip tensor input");
    }

    // Check inputs.
    NVDR_CHECK_DEVICE(tex, uv);
    NVDR_CHECK_CONTIGUOUS(tex, uv);
    NVDR_CHECK_F32(tex, uv);
    if (p.enableMip)
    {
        NVDR_CHECK_DEVICE(mip);
        NVDR_CHECK_CONTIGUOUS(mip);
        NVDR_CHECK_F32(mip);
        if (has_uv_da)
        {
            NVDR_CHECK_DEVICE(uv_da);
            NVDR_CHECK_CONTIGUOUS(uv_da);
            NVDR_CHECK_F32(uv_da);
        }
        if (has_mip_level_bias)
        {
            NVDR_CHECK_DEVICE(mip_level_bias);
            NVDR_CHECK_CONTIGUOUS(mip_level_bias);
            NVDR_CHECK_F32(mip_level_bias);
        }
    }

    // Sanity checks and state setters.
    bool cube_mode = (boundary_mode == TEX_BOUNDARY_MODE_CUBE);
    if (!cube_mode)
    {
        NVDR_CHECK(tex.sizes().size() == 4 && tex.size(0) > 0 && tex.size(1) > 0 && tex.size(2) > 0 && tex.size(3) > 0, "tex must have shape[>0, >0, >0, >0]");
        NVDR_CHECK(uv.sizes().size() == 4 && uv.size(0) > 0 && uv.size(1) > 0 && uv.size(2) > 0 && uv.size(3) == 2, "uv must have shape [>0, >0, >0, 2]");
        p.texHeight = tex.size(1);
        p.texWidth  = tex.size(2);
        p.channels  = tex.size(3);
    }
    else
    {
        NVDR_CHECK(tex.sizes().size() == 5 && tex.size(0) > 0 && tex.size(1) == 6 && tex.size(2) > 0 && tex.size(3) > 0 && tex.size(4) > 0, "tex must have shape[>0, 6, >0, >0, >0] in cube map mode");
        NVDR_CHECK(uv.sizes().size() == 4 && uv.size(0) > 0 && uv.size(1) > 0 && uv.size(2) > 0 && uv.size(3) == 3, "uv must have shape [>0, >0, >0, 3] in cube map mode");
        NVDR_CHECK(tex.size(2) == tex.size(3), "texture shape must be square in cube map mode");
        p.texHeight = tex.size(2);
        p.texWidth  = tex.size(3);
        p.channels  = tex.size(4);
    }
    NVDR_CHECK(tex.size(0) == 1 || tex.size(0) == uv.size(0), "minibatch size mismatch between inputs tex, uv");
    NVDR_CHECK(p.texWidth <= (1 << TEX_MAX_MIP_LEVEL) && p.texHeight <= (1 << TEX_MAX_MIP_LEVEL), "texture size too large");
    p.n         = uv.size(0);
    p.imgHeight = uv.size(1);
    p.imgWidth  = uv.size(2);
    p.texDepth  = tex.size(0);
    if (p.enableMip)
    {
        if (has_uv_da)
        {
            if (!cube_mode)
                NVDR_CHECK(uv_da.sizes().size() == 4 && uv_da.size(0) == p.n && uv_da.size(1) == p.imgHeight && uv_da.size(2) == p.imgWidth && uv_da.size(3) == 4, "uv_da must have shape [minibatch_size, height, width, 4]");
            else
                NVDR_CHECK(uv_da.sizes().size() == 4 && uv_da.size(0) == p.n && uv_da.size(1) == p.imgHeight && uv_da.size(2) == p.imgWidth && uv_da.size(3) == 6, "uv_da must have shape [minibatch_size, height, width, 6] in cube map mode");
        }
        if (has_mip_level_bias)
            NVDR_CHECK(mip_level_bias.sizes().size() == 3 && mip_level_bias.size(0) == p.n && mip_level_bias.size(1) == p.imgHeight && mip_level_bias.size(2) == p.imgWidth, "mip_level_bias must have shape [minibatch_size, height, width]");
    }

    // Get input pointers.
    p.tex = tex.data_ptr<float>();
    p.uv = uv.data_ptr<float>();
    p.uvDA = (p.enableMip && has_uv_da) ? uv_da.data_ptr<float>() : NULL;
    p.mipLevelBias = (p.enableMip && has_mip_level_bias) ? mip_level_bias.data_ptr<float>() : NULL;

    // Allocate output tensor.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({p.n, p.imgHeight, p.imgWidth, p.channels}, opts);
    p.out = out.data_ptr<float>();

    // Choose kernel variants based on channel count.
    void* args[] = {&p};
    int channel_div_idx = 0;
    if (!(p.channels & 3))
        channel_div_idx = 2;  // Channel count divisible by 4.
    else if (!(p.channels & 1))
        channel_div_idx = 1;  // Channel count divisible by 2.

    // Mip-related setup.
    if (p.enableMip)
    {
        // Generate mip offsets, check mipmap size, and set mip data pointer.
        int mipTotal = calculateMipInfo(NVDR_CTX_PARAMS, p);
        NVDR_CHECK(tex.sizes() == mip_wrap.texture_size && cube_mode == mip_wrap.cube_mode, "mip does not match texture size");
        NVDR_CHECK(mip.sizes().size() == 1 && mip.size(0) == mipTotal, "mip tensor size mismatch");
        p.mip = mip.data_ptr<float>();
    }

    // Verify that buffers are aligned to allow float2/float4 operations. Unused pointers are zero so always aligned.
    if (!cube_mode)
        NVDR_CHECK(!((uintptr_t)p.uv & 7), "uv input tensor not aligned to float2");
    if ((p.channels & 3) == 0)
    {
        NVDR_CHECK(!((uintptr_t)p.tex & 15), "tex input tensor not aligned to float4");
        NVDR_CHECK(!((uintptr_t)p.out & 15), "out output tensor not aligned to float4");
        NVDR_CHECK(!((uintptr_t)p.mip & 15), "mip output tensor not aligned to float4");
    }
    if ((p.channels & 1) == 0)
    {
        NVDR_CHECK(!((uintptr_t)p.tex & 7), "tex input tensor not aligned to float2");
        NVDR_CHECK(!((uintptr_t)p.out & 7), "out output tensor not aligned to float2");
        NVDR_CHECK(!((uintptr_t)p.mip & 7), "mip output tensor not aligned to float2");
    }
    if (!cube_mode)
        NVDR_CHECK(!((uintptr_t)p.uvDA & 15), "uv_da input tensor not aligned to float4");
    else
        NVDR_CHECK(!((uintptr_t)p.uvDA & 7), "uv_da input tensor not aligned to float2");

    // Choose launch parameters for texture lookup kernel.
    dim3 blockSize = getLaunchBlockSize(TEX_FWD_MAX_KERNEL_BLOCK_WIDTH, TEX_FWD_MAX_KERNEL_BLOCK_HEIGHT, p.imgWidth, p.imgHeight);
    dim3 gridSize  = getLaunchGridSize(blockSize, p.imgWidth, p.imgHeight, p.n);

    // Choose kernel based on filter mode, cube mode, bias-only mode, and datatype.
    void* func_tbl[TEX_MODE_COUNT * 2 * 2 * 3] = {
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
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        (void*)TextureFwdKernelLinearMipmapNearestBO1,
        (void*)TextureFwdKernelLinearMipmapNearestBO2,
        (void*)TextureFwdKernelLinearMipmapNearestBO4,
        (void*)TextureFwdKernelLinearMipmapLinearBO1,
        (void*)TextureFwdKernelLinearMipmapLinearBO2,
        (void*)TextureFwdKernelLinearMipmapLinearBO4,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        (void*)TextureFwdKernelCubeLinearMipmapNearestBO1,
        (void*)TextureFwdKernelCubeLinearMipmapNearestBO2,
        (void*)TextureFwdKernelCubeLinearMipmapNearestBO4,
        (void*)TextureFwdKernelCubeLinearMipmapLinearBO1,
        (void*)TextureFwdKernelCubeLinearMipmapLinearBO2,
        (void*)TextureFwdKernelCubeLinearMipmapLinearBO4,
    };

    // Function index.
    int func_idx = p.filterMode;
    if (cube_mode)
        func_idx += TEX_MODE_COUNT; // Cube variant.
    if (p.enableMip && !has_uv_da)
        func_idx += TEX_MODE_COUNT * 2; // Bias-only variant.
    func_idx = func_idx * 3 + channel_div_idx; // Choose vector size.

    // Launch kernel.
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func_tbl[func_idx], gridSize, blockSize, args, 0, stream));

    // Return output tensor.
    return out;
}

// Version without mipmaps.
torch::Tensor texture_fwd(torch::Tensor tex, torch::Tensor uv, int filter_mode, int boundary_mode)
{
    torch::Tensor empty_tensor;
    return texture_fwd_mip(tex, uv, empty_tensor, empty_tensor, TextureMipWrapper(), filter_mode, boundary_mode);
}

//------------------------------------------------------------------------
// Gradient op.

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> texture_grad_linear_mipmap_linear(torch::Tensor tex, torch::Tensor uv, torch::Tensor dy, torch::Tensor uv_da, torch::Tensor mip_level_bias, TextureMipWrapper mip_wrap, int filter_mode, int boundary_mode)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(tex));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    TextureKernelParams p = {}; // Initialize all fields to zero.
    torch::Tensor& mip = mip_wrap.mip; // Unwrap.
    int max_mip_level = mip_wrap.max_mip_level;
    set_modes(p, filter_mode, boundary_mode, max_mip_level);

    // See if we have these tensors or not.
    bool has_uv_da = uv_da.defined() && uv_da.nbytes();
    bool has_mip_level_bias = mip_level_bias.defined() && mip_level_bias.nbytes();

    if (p.enableMip)
    {
        NVDR_CHECK(has_uv_da || has_mip_level_bias, "mipmapping filter mode requires uv_da and/or mip_level_bias input");
        NVDR_CHECK(mip.defined(), "mipmapping filter mode requires mip tensor input");
    }

    // Check inputs.
    NVDR_CHECK_DEVICE(tex, uv);
    NVDR_CHECK_CONTIGUOUS(tex, uv);
    NVDR_CHECK_F32(tex, uv);
    if (p.enableMip)
    {
        NVDR_CHECK_DEVICE(mip);
        NVDR_CHECK_CONTIGUOUS(mip);
        NVDR_CHECK_F32(mip);
        if (has_uv_da)
        {
            NVDR_CHECK_DEVICE(uv_da);
            NVDR_CHECK_CONTIGUOUS(uv_da);
            NVDR_CHECK_F32(uv_da);
        }
        if (has_mip_level_bias)
        {
            NVDR_CHECK_DEVICE(mip_level_bias);
            NVDR_CHECK_CONTIGUOUS(mip_level_bias);
            NVDR_CHECK_F32(mip_level_bias);
        }
    }

    // Sanity checks and state setters.
    bool cube_mode = (boundary_mode == TEX_BOUNDARY_MODE_CUBE);
    if (!cube_mode)
    {
        NVDR_CHECK(tex.sizes().size() == 4 && tex.size(0) > 0 && tex.size(1) > 0 && tex.size(2) > 0 && tex.size(3) > 0, "tex must have shape[>0, >0, >0, >0]");
        NVDR_CHECK(uv.sizes().size() == 4 && uv.size(0) > 0 && uv.size(1) > 0 && uv.size(2) > 0 && uv.size(3) == 2, "uv must have shape [>0, >0, >0, 2]");
        p.texHeight = tex.size(1);
        p.texWidth  = tex.size(2);
        p.channels  = tex.size(3);
    }
    else
    {
        NVDR_CHECK(tex.sizes().size() == 5 && tex.size(0) > 0 && tex.size(1) == 6 && tex.size(2) > 0 && tex.size(3) > 0 && tex.size(4) > 0, "tex must have shape[>0, 6, >0, >0, >0] in cube map mode");
        NVDR_CHECK(uv.sizes().size() == 4 && uv.size(0) > 0 && uv.size(1) > 0 && uv.size(2) > 0 && uv.size(3) == 3, "uv must have shape [>0, >0, >0, 3] in cube map mode");
        NVDR_CHECK(tex.size(2) == tex.size(3), "texture shape must be square in cube map mode");
        p.texHeight = tex.size(2);
        p.texWidth  = tex.size(3);
        p.channels  = tex.size(4);
    }
    NVDR_CHECK(tex.size(0) == 1 || tex.size(0) == uv.size(0), "minibatch size mismatch between inputs tex, uv");
    NVDR_CHECK(p.texWidth <= (1 << TEX_MAX_MIP_LEVEL) && p.texHeight <= (1 << TEX_MAX_MIP_LEVEL), "texture size too large");
    p.n         = uv.size(0);
    p.imgHeight = uv.size(1);
    p.imgWidth  = uv.size(2);
    p.texDepth  = tex.size(0);
    if (p.enableMip)
    {
        if (has_uv_da)
        {
            if (!cube_mode)
                NVDR_CHECK(uv_da.sizes().size() == 4 && uv_da.size(0) == p.n && uv_da.size(1) == p.imgHeight && uv_da.size(2) == p.imgWidth && uv_da.size(3) == 4, "uv_da must have shape [minibatch_size, height, width, 4]");
            else
                NVDR_CHECK(uv_da.sizes().size() == 4 && uv_da.size(0) == p.n && uv_da.size(1) == p.imgHeight && uv_da.size(2) == p.imgWidth && uv_da.size(3) == 6, "uv_da must have shape [minibatch_size, height, width, 6] in cube map mode");
        }
        if (has_mip_level_bias)
            NVDR_CHECK(mip_level_bias.sizes().size() == 3 && mip_level_bias.size(0) == p.n && mip_level_bias.size(1) == p.imgHeight && mip_level_bias.size(2) == p.imgWidth, "mip_level_bias must have shape [minibatch_size, height, width]");
    }
    NVDR_CHECK(dy.sizes().size() == 4 && dy.size(0) == p.n && dy.size(1) == p.imgHeight && dy.size(2) == p.imgWidth && dy.size(3) == p.channels, "dy must have shape [minibatch_size, height, width, channels]");

    // Get contiguous version of dy.
    torch::Tensor dy_ = dy.contiguous();

    // Get input pointers.
    p.tex = tex.data_ptr<float>();
    p.uv = uv.data_ptr<float>();
    p.dy = dy_.data_ptr<float>();
    p.uvDA = (p.enableMip && has_uv_da) ? uv_da.data_ptr<float>() : NULL;
    p.mipLevelBias = (p.enableMip && has_mip_level_bias) ? mip_level_bias.data_ptr<float>() : NULL;
    p.mip = p.enableMip ? (float*)mip.data_ptr<float>() : NULL;

    // Allocate output tensor for tex gradient.
    torch::Tensor grad_tex = torch::zeros_like(tex);
    p.gradTex = grad_tex.data_ptr<float>();

    // Allocate output tensor for uv gradient.
    torch::Tensor grad_uv;
    torch::Tensor grad_uv_da;
    torch::Tensor grad_mip_level_bias;
    if (p.filterMode != TEX_MODE_NEAREST)
    {
        grad_uv = torch::empty_like(uv);
        p.gradUV = grad_uv.data_ptr<float>();

        // Gradients for things affecting mip level.
        if (p.filterMode == TEX_MODE_LINEAR_MIPMAP_LINEAR)
        {
            // Allocate output tensor for uv_da gradient.
            if (has_uv_da)
            {
                grad_uv_da = torch::empty_like(uv_da);
                p.gradUVDA = grad_uv_da.data_ptr<float>();
            }

            // Allocate output tensor for mip_level_bias gradient.
            if (has_mip_level_bias)
            {
                grad_mip_level_bias = torch::empty_like(mip_level_bias);
                p.gradMipLevelBias = grad_mip_level_bias.data_ptr<float>();
            }
        }
    }

    // Choose kernel variants based on channel count.
    int channel_div_idx = 0;
    if (!(p.channels & 3))
        channel_div_idx = 2;  // Channel count divisible by 4.
    else if (!(p.channels & 1))
        channel_div_idx = 1;  // Channel count divisible by 2.

    // Mip-related setup.
    torch::Tensor grad_mip;
    if (p.enableMip)
    {
        // Generate mip offsets and get space for temporary mip gradients.
        int mipTotal = calculateMipInfo(NVDR_CTX_PARAMS, p);
        NVDR_CHECK(tex.sizes() == mip_wrap.texture_size && cube_mode == mip_wrap.cube_mode, "mip does not match texture size");
        NVDR_CHECK(mip.sizes().size() == 1 && mip.size(0) == mipTotal, "mip tensor size mismatch");
        grad_mip = torch::zeros_like(mip);
        p.gradTexMip = grad_mip.data_ptr<float>();
    }

    // Verify that buffers are aligned to allow float2/float4 operations. Unused pointers are zero so always aligned.
    if (!cube_mode)
    {
        NVDR_CHECK(!((uintptr_t)p.uv       & 7), "uv input tensor not aligned to float2");
        NVDR_CHECK(!((uintptr_t)p.gradUV   & 7), "grad_uv output tensor not aligned to float2");
        NVDR_CHECK(!((uintptr_t)p.uvDA     & 15), "uv_da input tensor not aligned to float4");
        NVDR_CHECK(!((uintptr_t)p.gradUVDA & 15), "grad_uv_da output tensor not aligned to float4");
    }
    else
    {
        NVDR_CHECK(!((uintptr_t)p.uvDA     & 7), "uv_da input tensor not aligned to float2");
        NVDR_CHECK(!((uintptr_t)p.gradUVDA & 7), "grad_uv_da output tensor not aligned to float2");
    }
    if ((p.channels & 3) == 0)
    {
        NVDR_CHECK(!((uintptr_t)p.tex     & 15), "tex input tensor not aligned to float4");
        NVDR_CHECK(!((uintptr_t)p.gradTex & 15), "grad_tex output tensor not aligned to float4");
        NVDR_CHECK(!((uintptr_t)p.dy      & 15), "dy input tensor not aligned to float4");
        NVDR_CHECK(!((uintptr_t)p.mip     & 15), "mip input tensor not aligned to float4");
    }
    if ((p.channels & 1) == 0)
    {
        NVDR_CHECK(!((uintptr_t)p.tex     & 7), "tex input tensor not aligned to float2");
        NVDR_CHECK(!((uintptr_t)p.gradTex & 7), "grad_tex output tensor not aligned to float2");
        NVDR_CHECK(!((uintptr_t)p.dy      & 7), "dy output tensor not aligned to float2");
        NVDR_CHECK(!((uintptr_t)p.mip     & 7), "mip input tensor not aligned to float2");
    }

    // Choose launch parameters for main gradient kernel.
    void* args[] = {&p};
    dim3 blockSize = getLaunchBlockSize(TEX_GRAD_MAX_KERNEL_BLOCK_WIDTH, TEX_GRAD_MAX_KERNEL_BLOCK_HEIGHT, p.imgWidth, p.imgHeight);
    dim3 gridSize  = getLaunchGridSize(blockSize, p.imgWidth, p.imgHeight, p.n);

    void* func_tbl[TEX_MODE_COUNT * 2 * 2] = {
        (void*)TextureGradKernelNearest,
        (void*)TextureGradKernelLinear,
        (void*)TextureGradKernelLinearMipmapNearest,
        (void*)TextureGradKernelLinearMipmapLinear,
        (void*)TextureGradKernelCubeNearest,
        (void*)TextureGradKernelCubeLinear,
        (void*)TextureGradKernelCubeLinearMipmapNearest,
        (void*)TextureGradKernelCubeLinearMipmapLinear,
        NULL,
        NULL,
        (void*)TextureGradKernelLinearMipmapNearestBO,
        (void*)TextureGradKernelLinearMipmapLinearBO,
        NULL,
        NULL,
        (void*)TextureGradKernelCubeLinearMipmapNearestBO,
        (void*)TextureGradKernelCubeLinearMipmapLinearBO,
    };

    // Function index.
    int func_idx = p.filterMode;
    if (cube_mode)
        func_idx += TEX_MODE_COUNT; // Cube variant.
    if (p.enableMip && !has_uv_da)
        func_idx += TEX_MODE_COUNT * 2; // Bias-only variant.

    // Launch main gradient kernel.
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func_tbl[func_idx], gridSize, blockSize, args, 0, stream));

    // Launch kernel to pull gradients from mip levels.
    if (p.enableMip)
    {
        dim3 blockSize = getLaunchBlockSize(TEX_GRAD_MAX_MIP_KERNEL_BLOCK_WIDTH, TEX_GRAD_MAX_MIP_KERNEL_BLOCK_HEIGHT, p.texWidth, p.texHeight);
        dim3 gridSize  = getLaunchGridSize(blockSize, p.texWidth, p.texHeight, p.texDepth * (cube_mode ? 6 : 1));
        int sharedBytes = blockSize.x * blockSize.y * p.channels * sizeof(float);

        void* mip_grad_func_tbl[3] = { (void*)MipGradKernel1, (void*)MipGradKernel2, (void*)MipGradKernel4 };
        NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(mip_grad_func_tbl[channel_div_idx], gridSize, blockSize, args, sharedBytes, stream));
    }

    // Return output tensors.
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(grad_tex, grad_uv, grad_uv_da, grad_mip_level_bias);
}

// Version for nearest filter mode.
torch::Tensor texture_grad_nearest(torch::Tensor tex, torch::Tensor uv, torch::Tensor dy, int filter_mode, int boundary_mode)
{
    torch::Tensor empty_tensor;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> result = texture_grad_linear_mipmap_linear(tex, uv, dy, empty_tensor, empty_tensor, TextureMipWrapper(), filter_mode, boundary_mode);
    return std::get<0>(result);
}

// Version for linear filter mode.
std::tuple<torch::Tensor, torch::Tensor> texture_grad_linear(torch::Tensor tex, torch::Tensor uv, torch::Tensor dy, int filter_mode, int boundary_mode)
{
    torch::Tensor empty_tensor;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> result = texture_grad_linear_mipmap_linear(tex, uv, dy, empty_tensor, empty_tensor, TextureMipWrapper(), filter_mode, boundary_mode);
    return std::tuple<torch::Tensor, torch::Tensor>(std::get<0>(result), std::get<1>(result));
}

// Version for linear-mipmap-nearest mode.
std::tuple<torch::Tensor, torch::Tensor> texture_grad_linear_mipmap_nearest(torch::Tensor tex, torch::Tensor uv, torch::Tensor dy, torch::Tensor uv_da, torch::Tensor mip_level_bias, TextureMipWrapper mip, int filter_mode, int boundary_mode)
{
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> result = texture_grad_linear_mipmap_linear(tex, uv, dy, uv_da, mip_level_bias, mip, filter_mode, boundary_mode);
    return std::tuple<torch::Tensor, torch::Tensor>(std::get<0>(result), std::get<1>(result));
}

//------------------------------------------------------------------------
