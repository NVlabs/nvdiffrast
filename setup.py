# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Modified setup.py for building BINARY WHEELS with pre-compiled CUDA extensions
# This version compiles CUDA code during wheel build instead of JIT at runtime

import os
import sys
from setuptools import setup, find_packages

# Import version from __init__.py
import nvdiffrast
__version__ = nvdiffrast.__version__

# Read README
with open("README.md", "r") as fh:
    long_description = fh.read()

# Check if we should build binary wheels (requires PyTorch)
BUILD_BINARY = os.environ.get('BUILD_BINARY_WHEEL', '0') == '1'

if BUILD_BINARY:
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        print("[BUILD] Building BINARY wheel with pre-compiled CUDA extensions")
    except ImportError:
        print("[WARNING] PyTorch not found, falling back to source wheel")
        BUILD_BINARY = False

# Get repository root
nvdr_root = os.path.dirname(os.path.abspath(__file__))

# Setup configuration
setup_kwargs = {
    "name": "nvdiffrast",
    "version": __version__,
    "author": "Samuli Laine",
    "author_email": "slaine@nvidia.com",
    "description": "nvdiffrast - modular primitives for high-performance differentiable rendering",
    "long_description": long_description,
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/NVlabs/nvdiffrast",
    "packages": find_packages(),
    "install_requires": ['numpy'],
    "classifiers": [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    "python_requires": '>=3.6',
}

if BUILD_BINARY:
    # Binary wheel configuration with pre-compiled CUDA extensions
    print("[BUILD] Configuring CUDA extension compilation...")

    # Define source files for CUDA extension
    cuda_sources = [
        'nvdiffrast/common/cudaraster/impl/Buffer.cpp',
        'nvdiffrast/common/cudaraster/impl/CudaRaster.cpp',
        'nvdiffrast/common/cudaraster/impl/RasterImpl.cu',
        'nvdiffrast/common/cudaraster/impl/RasterImpl.cpp',
        'nvdiffrast/common/common.cpp',
        'nvdiffrast/common/rasterize.cu',
        'nvdiffrast/common/interpolate.cu',
        'nvdiffrast/common/texture.cu',
        'nvdiffrast/common/texture.cpp',
        'nvdiffrast/common/antialias.cu',
        'nvdiffrast/torch/torch_bindings.cpp',
        'nvdiffrast/torch/torch_rasterize.cpp',
        'nvdiffrast/torch/torch_interpolate.cpp',
        'nvdiffrast/torch/torch_texture.cpp',
        'nvdiffrast/torch/torch_antialias.cpp',
    ]

    # Convert to absolute paths
    cuda_sources = [os.path.join(nvdr_root, src) for src in cuda_sources]

    # Include directories
    include_dirs = [
        os.path.join(nvdr_root, 'nvdiffrast', 'common'),
        os.path.join(nvdr_root, 'nvdiffrast', 'torch'),
    ]

    # Compiler flags
    # Windows-specific flags for CUDA 11.8 + VS 2022 compatibility
    if os.name == 'nt':
        cxx_flags = [
            '/wd4067',  # Disable warning: unexpected tokens following preprocessor directive
            '/wd4624',  # Disable warning: destructor was implicitly defined as deleted
            '/DNVDR_TORCH',  # Define NVDR_TORCH for C++ files (enables CUDA macros)
            '/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH',  # Allow CUDA 11.8 with latest VS 2022
            '/D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS',  # Silence C++17 deprecation warnings
            '/bigobj',  # Support large object files (needed for CUDA compilation)
        ]
        nvcc_flags = [
            '-DNVDR_TORCH',
            '--use_fast_math',
            '-allow-unsupported-compiler',  # Allow CUDA 11.8 with VS 2022
            '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH',  # Bypass STL version check
            '--expt-relaxed-constexpr',  # Allow constexpr extensions
            '--expt-extended-lambda',  # Allow extended lambda features
        ]
    else:
        cxx_flags = []
        nvcc_flags = ['-DNVDR_TORCH', '--use_fast_math']

    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': nvcc_flags
    }

    # Linker flags (Windows)
    extra_link_args = []
    if os.name == 'nt':
        lib_dir = os.path.join(nvdr_root, 'nvdiffrast', 'lib')
        extra_link_args = [f'/LIBPATH:{lib_dir}', '/DEFAULTLIB:setgpu']

    # Create the CUDA extension
    ext_modules = [
        CUDAExtension(
            name='nvdiffrast_plugin',
            sources=cuda_sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    setup_kwargs.update({
        "ext_modules": ext_modules,
        "cmdclass": {'build_ext': BuildExtension.with_options(use_ninja=False)},
        "zip_safe": False,
    })

    print(f"[BUILD] CUDA sources: {len(cuda_sources)} files")
    print(f"[BUILD] Include dirs: {include_dirs}")

else:
    # Source wheel configuration (fallback - original behavior)
    print("[BUILD] Building SOURCE wheel (requires JIT compilation at runtime)")

    setup_kwargs.update({
        "package_data": {
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
                'lib/*.h',
                'torch/*.h',
                'torch/*.inl',
                'torch/*.cpp',
                'tensorflow/*.cu',
            ] + (['lib/*.lib'] if os.name == 'nt' else [])
        },
        "include_package_data": True,
    })

# Run setup
setup(**setup_kwargs)
