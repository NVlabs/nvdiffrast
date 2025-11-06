// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once
#include "framework.h"

//------------------------------------------------------------------------
// Constants.

#define TEX_DEBUG_MIP_RETAIN_VARIANCE           0   // For debugging
#define TEX_FWD_MAX_KERNEL_BLOCK_WIDTH          8
#define TEX_FWD_MAX_KERNEL_BLOCK_HEIGHT         8
#define TEX_FWD_MAX_MIP_KERNEL_BLOCK_WIDTH      8
#define TEX_FWD_MAX_MIP_KERNEL_BLOCK_HEIGHT     8
#define TEX_GRAD_MAX_KERNEL_BLOCK_WIDTH         8
#define TEX_GRAD_MAX_KERNEL_BLOCK_HEIGHT        8
#define TEX_GRAD_MAX_MIP_KERNEL_BLOCK_WIDTH     8
#define TEX_GRAD_MAX_MIP_KERNEL_BLOCK_HEIGHT    8
#define TEX_MAX_MIP_LEVEL                       16  // Currently a texture cannot be larger than 2 GB because we use 32-bit indices everywhere.
#define TEX_MODE_NEAREST                        0   // Nearest on base level.
#define TEX_MODE_LINEAR                         1   // Bilinear on base level.
#define TEX_MODE_LINEAR_MIPMAP_NEAREST          2   // Bilinear on nearest mip level.
#define TEX_MODE_LINEAR_MIPMAP_LINEAR           3   // Trilinear.
#define TEX_MODE_COUNT                          4
#define TEX_BOUNDARY_MODE_CUBE                  0   // Cube map mode.
#define TEX_BOUNDARY_MODE_WRAP                  1   // Wrap (u, v).
#define TEX_BOUNDARY_MODE_CLAMP                 2   // Clamp (u, v).
#define TEX_BOUNDARY_MODE_ZERO                  3   // Pad with zeros.
#define TEX_BOUNDARY_MODE_COUNT                 4

//------------------------------------------------------------------------
// CUDA kernel params.

struct TextureKernelParams
{
    const float*    tex[TEX_MAX_MIP_LEVEL];         // Incoming texture buffer with mip levels.
    const float*    uv;                             // Incoming texcoord buffer.
    const float*    uvDA;                           // Incoming uv pixel diffs or NULL.
    const float*    mipLevelBias;                   // Incoming mip level bias or NULL.
    const float*    dy;                             // Incoming output gradient.
    float*          out;                            // Outgoing texture data.
    float*          gradTex[TEX_MAX_MIP_LEVEL];     // Outgoing texture gradients with mip levels.
    float*          gradUV;                         // Outgoing texcoord gradient.
    float*          gradUVDA;                       // Outgoing texcoord pixel differential gradient.
    float*          gradMipLevelBias;               // Outgoing mip level bias gradient.
    int             enableMip;                      // If true, we have uv_da and/or mip_level_bias input(s), and a mip tensor.
    int             filterMode;                     // One of the TEX_MODE_ constants.
    int             boundaryMode;                   // One of the TEX_BOUNDARY_MODE_ contants.
    int             texConst;                       // If true, texture is known to be constant.
    int             mipLevelLimit;                  // Mip level limit coming from the op.
    int             channels;                       // Number of texture channels.
    int             imgWidth;                       // Image width.
    int             imgHeight;                      // Image height.
    int             texWidth;                       // Texture width.
    int             texHeight;                      // Texture height.
    int             texDepth;                       // Texture depth.
    int             n;                              // Minibatch size.
    int             mipLevelMax;                    // Maximum mip level index. Zero if mips disabled.
    int             mipLevelOut;                    // Mip level being calculated in builder kernel.
};

//------------------------------------------------------------------------
// C++ helper function prototypes.

void raiseMipSizeError(NVDR_CTX_ARGS, const TextureKernelParams& p);
int calculateMipInfo(NVDR_CTX_ARGS, TextureKernelParams& p, int* mipOffsets);

//------------------------------------------------------------------------
// Macros.

#define mipLevelSize(p, i) make_int2(((p).texWidth >> (i)) > 1 ? ((p).texWidth >> (i)) : 1, ((p).texHeight >> (i)) > 1 ? ((p).texHeight >> (i)) : 1)

//------------------------------------------------------------------------
