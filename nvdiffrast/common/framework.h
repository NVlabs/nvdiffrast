// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

// Framework-specific macros to enable code sharing.

//------------------------------------------------------------------------
// Tensorflow.

#ifdef NVDR_TENSORFLOW
#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/default/logging.h"
using namespace tensorflow;
using namespace tensorflow::shape_inference;
#define NVDR_CTX_ARGS OpKernelContext* _nvdr_ctx
#define NVDR_CTX_PARAMS _nvdr_ctx
#define NVDR_CHECK(COND, ERR) OP_REQUIRES(_nvdr_ctx, COND, errors::Internal(ERR))
#define NVDR_CHECK_CUDA_ERROR(CUDA_CALL) OP_CHECK_CUDA_ERROR(_nvdr_ctx, CUDA_CALL)
#define NVDR_CHECK_GL_ERROR(GL_CALL) OP_CHECK_GL_ERROR(_nvdr_ctx, GL_CALL)
#endif

//------------------------------------------------------------------------
// PyTorch.

#ifdef NVDR_TORCH
#ifndef __CUDACC__
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/numpy.h>
#endif
#define NVDR_CTX_ARGS int _nvdr_ctx_dummy
#define NVDR_CTX_PARAMS 0
#define NVDR_CHECK(COND, ERR) do { TORCH_CHECK(COND, ERR) } while(0)
#define NVDR_CHECK_CUDA_ERROR(CUDA_CALL) do { cudaError_t err = CUDA_CALL; TORCH_CHECK(!err, "Cuda error: ", cudaGetLastError(), "[", #CUDA_CALL, ";]"); } while(0)
#define NVDR_CHECK_GL_ERROR(GL_CALL) do { GL_CALL; GLenum err = glGetError(); TORCH_CHECK(err == GL_NO_ERROR, "OpenGL error: ", getGLErrorString(err), "[", #GL_CALL, ";]"); } while(0)
#endif

//------------------------------------------------------------------------
