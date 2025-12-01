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
    # This happens if the user runs 'pip install .' (default build isolation)
    # OR if they simply don't have torch installed at all.
    print("\n\n" + "*" * 70)
    print("ERROR: PyTorch is not installed in the build environment!")
    print("To fix this, you have two options:\n")
    print("1. Install PyTorch in your environment first, then run:")
    print("   pip install . --no-build-isolation\n")
    print("2. If you are building a wheel for distribution, ensure")
    print("   PyTorch is installed in your build environment.")
    print("*" * 70 + "\n\n")
    raise

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
