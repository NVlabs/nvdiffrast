// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

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
    int                     enableZModify;      // Modify depth in shader, workaround for a rasterization issue on A100.
};

//------------------------------------------------------------------------
// Shared C++ code prototypes.

void rasterizeInitGLContext(NVDR_CTX_ARGS, RasterizeGLState& s, int cudaDeviceIdx);
void rasterizeResizeBuffers(NVDR_CTX_ARGS, RasterizeGLState& s, bool& changes, int posCount, int triCount, int width, int height, int depth);
void rasterizeRender(NVDR_CTX_ARGS, RasterizeGLState& s, cudaStream_t stream, const float* posPtr, int posCount, int vtxPerInstance, const int32_t* triPtr, int triCount, const int32_t* rangesPtr, int width, int height, int depth, int peeling_idx);
void rasterizeCopyResults(NVDR_CTX_ARGS, RasterizeGLState& s, cudaStream_t stream, float** outputPtr, int width, int height, int depth);
void rasterizeReleaseBuffers(NVDR_CTX_ARGS, RasterizeGLState& s);

//------------------------------------------------------------------------
#endif // !(defined(NVDR_TORCH) && defined(__CUDACC__))
