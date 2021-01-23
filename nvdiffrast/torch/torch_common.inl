// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once
#include "../common/framework.h"

//------------------------------------------------------------------------
// Input check helpers.
//------------------------------------------------------------------------

#ifdef _MSC_VER
#define __func__ __FUNCTION__
#endif

#define NVDR_CHECK_DEVICE(...) do { TORCH_CHECK(at::cuda::check_device({__VA_ARGS__}), __func__, "(): Inputs " #__VA_ARGS__ " must reside on the same GPU device") } while(0)
#define NVDR_CHECK_CPU(...) do { nvdr_check_cpu({__VA_ARGS__}, __func__, "(): Inputs " #__VA_ARGS__ " must reside on CPU"); } while(0)
#define NVDR_CHECK_CONTIGUOUS(...) do { nvdr_check_contiguous({__VA_ARGS__}, __func__, "(): Inputs " #__VA_ARGS__ " must be contiguous tensors"); } while(0)
#define NVDR_CHECK_F32(...) do { nvdr_check_f32({__VA_ARGS__}, __func__, "(): Inputs " #__VA_ARGS__ " must be float32 tensors"); } while(0)
#define NVDR_CHECK_I32(...) do { nvdr_check_i32({__VA_ARGS__}, __func__, "(): Inputs " #__VA_ARGS__ " must be int32 tensors"); } while(0)
inline void nvdr_check_cpu(at::ArrayRef<at::Tensor> ts,        const char* func, const char* err_msg) { for (const at::Tensor& t : ts) TORCH_CHECK(t.device().type() == c10::DeviceType::CPU, func, err_msg); }
inline void nvdr_check_contiguous(at::ArrayRef<at::Tensor> ts, const char* func, const char* err_msg) { for (const at::Tensor& t : ts) TORCH_CHECK(t.is_contiguous(), func, err_msg); }
inline void nvdr_check_f32(at::ArrayRef<at::Tensor> ts,        const char* func, const char* err_msg) { for (const at::Tensor& t : ts) TORCH_CHECK(t.dtype() == torch::kFloat32, func, err_msg); }
inline void nvdr_check_i32(at::ArrayRef<at::Tensor> ts,        const char* func, const char* err_msg) { for (const at::Tensor& t : ts) TORCH_CHECK(t.dtype() == torch::kInt32, func, err_msg); }
//------------------------------------------------------------------------
