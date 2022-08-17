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
// Op prototypes. Return type macros for readability.

#define OP_RETURN_T     torch::Tensor
#define OP_RETURN_TT    std::tuple<torch::Tensor, torch::Tensor>
#define OP_RETURN_TTT   std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
#define OP_RETURN_TTTT  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
#define OP_RETURN_TTV   std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor> >
#define OP_RETURN_TTTTV std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<torch::Tensor> >

OP_RETURN_TT        rasterize_fwd_cuda                  (RasterizeCRStateWrapper& stateWrapper, torch::Tensor pos, torch::Tensor tri, std::tuple<int, int> resolution, torch::Tensor ranges, int peeling_idx);
OP_RETURN_T         rasterize_grad                      (torch::Tensor pos, torch::Tensor tri, torch::Tensor out, torch::Tensor dy);
OP_RETURN_T         rasterize_grad_db                   (torch::Tensor pos, torch::Tensor tri, torch::Tensor out, torch::Tensor dy, torch::Tensor ddb);
OP_RETURN_TT        interpolate_fwd                     (torch::Tensor attr, torch::Tensor rast, torch::Tensor tri);
OP_RETURN_TT        interpolate_fwd_da                  (torch::Tensor attr, torch::Tensor rast, torch::Tensor tri, torch::Tensor rast_db, bool diff_attrs_all, std::vector<int>& diff_attrs_vec);
OP_RETURN_TT        interpolate_grad                    (torch::Tensor attr, torch::Tensor rast, torch::Tensor tri, torch::Tensor dy);
OP_RETURN_TTT       interpolate_grad_da                 (torch::Tensor attr, torch::Tensor rast, torch::Tensor tri, torch::Tensor dy, torch::Tensor rast_db, torch::Tensor dda, bool diff_attrs_all, std::vector<int>& diff_attrs_vec);
TextureMipWrapper   texture_construct_mip               (torch::Tensor tex, int max_mip_level, bool cube_mode);
OP_RETURN_T         texture_fwd                         (torch::Tensor tex, torch::Tensor uv, int filter_mode, int boundary_mode);
OP_RETURN_T         texture_fwd_mip                     (torch::Tensor tex, torch::Tensor uv, torch::Tensor uv_da, torch::Tensor mip_level_bias, TextureMipWrapper mip_wrapper, std::vector<torch::Tensor> mip_stack, int filter_mode, int boundary_mode);
OP_RETURN_T         texture_grad_nearest                (torch::Tensor tex, torch::Tensor uv, torch::Tensor dy, int filter_mode, int boundary_mode);
OP_RETURN_TT        texture_grad_linear                 (torch::Tensor tex, torch::Tensor uv, torch::Tensor dy, int filter_mode, int boundary_mode);
OP_RETURN_TTV       texture_grad_linear_mipmap_nearest  (torch::Tensor tex, torch::Tensor uv, torch::Tensor dy, torch::Tensor uv_da, torch::Tensor mip_level_bias, TextureMipWrapper mip_wrapper, std::vector<torch::Tensor> mip_stack, int filter_mode, int boundary_mode);
OP_RETURN_TTTTV     texture_grad_linear_mipmap_linear   (torch::Tensor tex, torch::Tensor uv, torch::Tensor dy, torch::Tensor uv_da, torch::Tensor mip_level_bias, TextureMipWrapper mip_wrapper, std::vector<torch::Tensor> mip_stack, int filter_mode, int boundary_mode);
TopologyHashWrapper antialias_construct_topology_hash   (torch::Tensor tri);
OP_RETURN_TT        antialias_fwd                       (torch::Tensor color, torch::Tensor rast, torch::Tensor pos, torch::Tensor tri, TopologyHashWrapper topology_hash);
OP_RETURN_TT        antialias_grad                      (torch::Tensor color, torch::Tensor rast, torch::Tensor pos, torch::Tensor tri, torch::Tensor dy, torch::Tensor work_buffer);

//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // State classes.
    pybind11::class_<RasterizeCRStateWrapper>(m, "RasterizeCRStateWrapper").def(pybind11::init<int>());
    pybind11::class_<TextureMipWrapper>(m, "TextureMipWrapper").def(pybind11::init<>());
    pybind11::class_<TopologyHashWrapper>(m, "TopologyHashWrapper");

    // Plumbing to torch/c10 logging system.
    m.def("get_log_level", [](void)     { return FLAGS_caffe2_log_level;  }, "get log level");
    m.def("set_log_level", [](int level){ FLAGS_caffe2_log_level = level; }, "set log level");

    // Ops.
    m.def("rasterize_fwd_cuda",                 &rasterize_fwd_cuda,                    "rasterize forward op (cuda)");
    m.def("rasterize_grad",                     &rasterize_grad,                        "rasterize gradient op ignoring db gradients");
    m.def("rasterize_grad_db",                  &rasterize_grad_db,                     "rasterize gradient op with db gradients");
    m.def("interpolate_fwd",                    &interpolate_fwd,                       "interpolate forward op with attribute derivatives");
    m.def("interpolate_fwd_da",                 &interpolate_fwd_da,                    "interpolate forward op without attribute derivatives");
    m.def("interpolate_grad",                   &interpolate_grad,                      "interpolate gradient op with attribute derivatives");
    m.def("interpolate_grad_da",                &interpolate_grad_da,                   "interpolate gradient op without attribute derivatives");
    m.def("texture_construct_mip",              &texture_construct_mip,                 "texture mipmap construction");
    m.def("texture_fwd",                        &texture_fwd,                           "texture forward op without mipmapping");
    m.def("texture_fwd_mip",                    &texture_fwd_mip,                       "texture forward op with mipmapping");
    m.def("texture_grad_nearest",               &texture_grad_nearest,                  "texture gradient op in nearest mode");
    m.def("texture_grad_linear",                &texture_grad_linear,                   "texture gradient op in linear mode");
    m.def("texture_grad_linear_mipmap_nearest", &texture_grad_linear_mipmap_nearest,    "texture gradient op in linear-mipmap-nearest mode");
    m.def("texture_grad_linear_mipmap_linear",  &texture_grad_linear_mipmap_linear,     "texture gradient op in linear-mipmap-linear mode");
    m.def("antialias_construct_topology_hash",  &antialias_construct_topology_hash,     "antialias topology hash construction");
    m.def("antialias_fwd",                      &antialias_fwd,                         "antialias forward op");
    m.def("antialias_grad",                     &antialias_grad,                        "antialias gradient op");
}

//------------------------------------------------------------------------
