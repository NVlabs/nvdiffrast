# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import setuptools
import os
import logging
import subprocess

# Print an error message if there's no PyTorch installed.
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
except ImportError:
    # This happens if the user runs 'pip install' with default build isolation
    # OR if they simply don't have torch installed at all.
    print("\n\n" + "*" * 70)
    print("ERROR! Cannot compile nvdiffrast CUDA extension. Please ensure that:\n")
    print("1. You have PyTorch installed")
    print("2. You run 'pip install' with --no-build-isolation flag")
    print("*" * 70 + "\n\n")
    exit(1)

def get_cuda_bare_metal_version(cuda_dir):
    """Get CUDA version from nvcc."""
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    return raw_output, release[0], release[1][0]

# Handle CUDA availability
if not torch.cuda.is_available() and os.getenv('NVDIFFRAST_CROSS_COMPILE_ALL', '0') == '1':
    logging.warning(
        "Torch did not find available GPUs.\n"
        "Assuming cross-compilation on all the supported architecture (by the torch's CUDA).\n"
        "Set TORCH_CUDA_ARCH_LIST for specific architectures."
    )
    if os.getenv("TORCH_CUDA_ARCH_LIST") is None:
        _, major, minor = get_cuda_bare_metal_version(CUDA_HOME)
        major, minor = int(major), int(minor)
        if major == 11:
            if minor == 0:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0"
            elif minor < 8:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6"
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.9"
        elif major == 12:
            if minor <= 6:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0"
            elif minor == 8:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0;12.0"
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0;12.0;12.1"
        elif major == 13:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6;8.9;9.0;12.0;12.1"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"
        print(f'TORCH_CUDA_ARCH_LIST: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
elif not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available, to cross-compile Set TORCH_CUDA_ARCH_LIST for specific architectures,\n"
                       "or NVDIFFRAST_CROSS_COMPILE_ALL=1 to cross-compile across all the supported architectures.")

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
