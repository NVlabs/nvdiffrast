// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

//------------------------------------------------------------------------
// Constants and helpers.

#define RAST_CUDA_FWD_SHADER_KERNEL_BLOCK_WIDTH  8
#define RAST_CUDA_FWD_SHADER_KERNEL_BLOCK_HEIGHT 8
#define RAST_GRAD_MAX_KERNEL_BLOCK_WIDTH  8
#define RAST_GRAD_MAX_KERNEL_BLOCK_HEIGHT 8

//------------------------------------------------------------------------
// CUDA forward rasterizer shader kernel params.

struct RasterizeCudaFwdShaderParams
{
    const float*    pos;            // Vertex positions.
    const int*      tri;            // Triangle indices.
    const int*      in_idx;         // Triangle idx buffer from rasterizer.
    float*          out;            // Main output buffer.
    float*          out_db;         // Bary pixel gradient output buffer.
    int             numTriangles;   // Number of triangles.
    int             numVertices;    // Number of vertices.
    int             width;          // Image width.
    int             height;         // Image height.
    int             depth;          // Size of minibatch.
    int             instance_mode;  // 1 if in instance rendering mode.
    float           xs, xo, ys, yo; // Pixel position to clip-space x, y transform.
};

//------------------------------------------------------------------------
// Gradient CUDA kernel params.

struct RasterizeGradParams
{
    const float*    pos;            // Incoming position buffer.
    const int*      tri;            // Incoming triangle buffer.
    const float*    out;            // Rasterizer output buffer.
    const float*    dy;             // Incoming gradients of rasterizer output buffer.
    const float*    ddb;            // Incoming gradients of bary diff output buffer.
    float*          grad;           // Outgoing position gradients.
    int             numTriangles;   // Number of triangles.
    int             numVertices;    // Number of vertices.
    int             width;          // Image width.
    int             height;         // Image height.
    int             depth;          // Size of minibatch.
    int             instance_mode;  // 1 if in instance rendering mode.
    float           xs, xo, ys, yo; // Pixel position to clip-space x, y transform.
};

//------------------------------------------------------------------------
