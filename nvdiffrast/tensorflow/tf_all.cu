// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

// TF-specific helpers.

#define OP_CHECK_CUDA_ERROR(CTX, CUDA_CALL) do { cudaError_t err = CUDA_CALL; OP_REQUIRES(CTX, err == cudaSuccess, errors::Internal("Cuda error: ", cudaGetErrorName(err), "[", #CUDA_CALL, ";]")); } while (0)
#define OP_CHECK_GL_ERROR(CTX, GL_CALL) do { GL_CALL; GLenum err = glGetError(); OP_REQUIRES(CTX, err == GL_NO_ERROR, errors::Internal("OpenGL error: ", getGLErrorString(err), "[", #GL_CALL, ";]")); } while (0)

// Cuda kernels and CPP all together. What an absolute compilation unit.

#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#include "../common/framework.h"
#include "../common/glutil.cpp"

#include "../common/common.h"
#include "../common/common.cpp"

#include "../common/rasterize.h"
#include "../common/rasterize_gl.cpp"
#include "../common/rasterize.cu"
#include "tf_rasterize.cu"

#include "../common/interpolate.cu"
#include "tf_interpolate.cu"

#include "../common/texture.cpp"
#include "../common/texture.cu"
#include "tf_texture.cu"

#include "../common/antialias.cu"
#include "tf_antialias.cu"
