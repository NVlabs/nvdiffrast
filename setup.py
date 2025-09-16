# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import nvdiffrast
import setuptools
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nvdiffrast",
    version=nvdiffrast.__version__,
    author="Samuli Laine",
    author_email="slaine@nvidia.com",
    description="nvdiffrast - modular primitives for high-performance differentiable rendering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NVlabs/nvdiffrast",
    packages=setuptools.find_packages(),
    package_data={
        'nvdiffrast': [
            'common/*.h',
            'common/*.inl',
            'common/*.cu',
            'common/*.cpp',
            'common/cudaraster/*.hpp',
            'common/cudaraster/impl/*.cpp',
            'common/cudaraster/impl/*.hpp',
            'common/cudaraster/impl/*.inl',
            'common/cudaraster/impl/*.cu',
            'torch/*.h',
            'torch/*.inl',
            'torch/*.cpp',
        ]
    },
    include_package_data=True,
    ext_modules=[
        CUDAExtension(
            "_nvdiffrast_c",
            sources=[
                "nvdiffrast/common/antialias.cu",
                "nvdiffrast/common/common.cpp",
                "nvdiffrast/common/cudaraster/impl/Buffer.cpp",
                "nvdiffrast/common/cudaraster/impl/CudaRaster.cpp",
                "nvdiffrast/common/cudaraster/impl/RasterImpl.cpp",
                "nvdiffrast/common/cudaraster/impl/RasterImpl_kernel.cu",
                "nvdiffrast/common/interpolate.cu",
                "nvdiffrast/common/rasterize.cu",
                "nvdiffrast/common/texture.cpp",
                "nvdiffrast/common/texture_kernel.cu",
                "nvdiffrast/torch/torch_antialias.cpp",
                "nvdiffrast/torch/torch_bindings.cpp",
                "nvdiffrast/torch/torch_interpolate.cpp",
                "nvdiffrast/torch/torch_rasterize.cpp",
                "nvdiffrast/torch/torch_texture.cpp",
            ],
            extra_compile_args={
                "cxx": ["-DNVDR_TORCH"]
                # Disable warnings in torch headers.
                + (["/wd4067", "/wd4624"] if os.name == "nt" else []),
                "nvcc": ["-DNVDR_TORCH", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=['numpy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
