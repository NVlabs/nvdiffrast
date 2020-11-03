// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once
#include "common.h"

//------------------------------------------------------------------------
// Constants and helpers.

#define AA_DISCONTINUITY_KERNEL_BLOCK_WIDTH     32
#define AA_DISCONTINUITY_KERNEL_BLOCK_HEIGHT    8
#define AA_ANALYSIS_KERNEL_THREADS_PER_BLOCK    256
#define AA_MESH_KERNEL_THREADS_PER_BLOCK        256
#define AA_HASH_ELEMENTS_PER_TRIANGLE           8   // Minimum is 4 but 8 gives fewer collisions. Must be power of two.
#define AA_GRAD_KERNEL_THREADS_PER_BLOCK        256

//------------------------------------------------------------------------
// CUDA kernel params.

struct AntialiasKernelParams
{
    const float*    color;          // Incoming color buffer.
    const float*    rasterOut;      // Incoming rasterizer output buffer.
    const int*      tri;            // Incoming triangle buffer.
    const float*    pos;            // Incoming position buffer.
    float*          output;         // Output buffer of forward kernel.    
    const float*    dy;             // Incoming gradients.
    float*          gradColor;      // Output buffer, color gradient.
    float*          gradPos;        // Output buffer, position gradient.
    int4*           workBuffer;     // Buffer for storing intermediate work items. First item reserved for counters.
    uint4*          evHash;         // Edge-vertex hash.
    int             allocTriangles; // Number of triangles accommodated by evHash. Always power of two.
    int             numTriangles;   // Number of triangles.
    int             numVertices;    // Number of vertices.
    int             width;          // Input width.
    int             height;         // Input height.
    int             n;              // Minibatch size.
    int             channels;       // Channel count in color input.
    float           xh, yh;         // Transfer to pixel space.
    int             instance_mode;  // 0=normal, 1=instance mode.
    int             tri_const;      // 1 if triangle array is known to be constant.
};

//------------------------------------------------------------------------
