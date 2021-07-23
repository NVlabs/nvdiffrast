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

#define RAST_GRAD_MAX_KERNEL_BLOCK_WIDTH  8
#define RAST_GRAD_MAX_KERNEL_BLOCK_HEIGHT 8

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
// Do not try to include OpenGL stuff when compiling CUDA kernels for torch.

#if !(defined(NVDR_TORCH) && defined(__CUDACC__))
#include "framework.h"
#include "glutil.h"

//------------------------------------------------------------------------
// OpenGL-related persistent state for forward op.

struct RasterizeGLState // Must be initializable by memset to zero.
{
    int                     width;              // Allocated frame buffer width.
    int                     height;             // Allocated frame buffer height.
    int                     depth;              // Allocated frame buffer depth.
    int                     posCount;           // Allocated position buffer in floats.
    int                     triCount;           // Allocated triangle buffer in ints.
    GLContext               glctx;
    GLuint                  glFBO;
    GLuint                  glColorBuffer[2];
    GLuint                  glPrevOutBuffer;
    GLuint                  glDepthStencilBuffer;
    GLuint                  glVAO;
    GLuint                  glTriBuffer;
    GLuint                  glPosBuffer;
    GLuint                  glProgram;
    GLuint                  glProgramDP;
    GLuint                  glVertexShader;
    GLuint                  glGeometryShader;
    GLuint                  glFragmentShader;
    GLuint                  glFragmentShaderDP;
    cudaGraphicsResource_t  cudaColorBuffer[2];
    cudaGraphicsResource_t  cudaPrevOutBuffer;
    cudaGraphicsResource_t  cudaPosBuffer;
    cudaGraphicsResource_t  cudaTriBuffer;
    int                     enableDB;
};

//------------------------------------------------------------------------
// Shared C++ code prototypes.

void rasterizeInitGLContext(NVDR_CTX_ARGS, RasterizeGLState& s, int cudaDeviceIdx);
void rasterizeResizeBuffers(NVDR_CTX_ARGS, RasterizeGLState& s, int posCount, int triCount, int width, int height, int depth);
void rasterizeRender(NVDR_CTX_ARGS, RasterizeGLState& s, cudaStream_t stream, const float* posPtr, int posCount, int vtxPerInstance, const int32_t* triPtr, int triCount, const int32_t* rangesPtr, int width, int height, int depth, int peeling_idx);
void rasterizeCopyResults(NVDR_CTX_ARGS, RasterizeGLState& s, cudaStream_t stream, float** outputPtr, int width, int height, int depth);
void rasterizeReleaseBuffers(NVDR_CTX_ARGS, RasterizeGLState& s);

//------------------------------------------------------------------------
#endif // !(defined(NVDR_TORCH) && defined(__CUDACC__))
