// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "torch_common.inl"
#include "torch_types.h"
#include <tuple>

//------------------------------------------------------------------------
// Op prototypes.

std::tuple<torch::Tensor, torch::Tensor> rasterize_fwd_gl(RasterizeGLStateWrapper& stateWrapper, torch::Tensor pos, torch::Tensor tri, std::tuple<int, int> resolution, torch::Tensor ranges, int peeling_idx);

//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // State classes.
    pybind11::class_<RasterizeGLStateWrapper>(m, "RasterizeGLStateWrapper").def(pybind11::init<bool, bool, int>())
        .def("set_context",     &RasterizeGLStateWrapper::setContext)
        .def("release_context", &RasterizeGLStateWrapper::releaseContext);

    // Ops.
    m.def("rasterize_fwd_gl", &rasterize_fwd_gl, "rasterize forward op (opengl)");
}

//------------------------------------------------------------------------
