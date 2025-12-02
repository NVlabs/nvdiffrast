# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import setuptools
import os

# Print an error message if there's no PyTorch installed.
try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    # This happens if the user runs 'pip install' with default build isolation
    # OR if they simply don't have torch installed at all.
    print("\n\n" + "*" * 70)
    print("ERROR! Cannot compile nvdiffrast CUDA extension. Please ensure that:\n")
    print("1. You have PyTorch installed")
    print("2. You run 'pip install' with --no-build-isolation flag")
    print("*" * 70 + "\n\n")
    exit(1)

setuptools.setup(
    ext_modules=[
        CUDAExtension(
            "_nvdiffrast_c",
            sources=[
                "csrc/common/antialias.cu",
                "csrc/common/common.cpp",
                "csrc/common/cudaraster/impl/Buffer.cpp",
                "csrc/common/cudaraster/impl/CudaRaster.cpp",
                "csrc/common/cudaraster/impl/RasterImpl.cpp",
                "csrc/common/cudaraster/impl/RasterImpl_kernel.cu",
                "csrc/common/interpolate.cu",
                "csrc/common/rasterize.cu",
                "csrc/common/texture.cpp",
                "csrc/common/texture_kernel.cu",
                "csrc/torch/torch_antialias.cpp",
                "csrc/torch/torch_bindings.cpp",
                "csrc/torch/torch_interpolate.cpp",
                "csrc/torch/torch_rasterize.cpp",
                "csrc/torch/torch_texture.cpp",
            ],
            extra_compile_args={
                "cxx": ["-DNVDR_TORCH"]
                # Disable warnings in torch headers.
                + (["/wd4067", "/wd4624", "/wd4996"] if os.name == "nt" else []),
                "nvcc": ["-DNVDR_TORCH", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
