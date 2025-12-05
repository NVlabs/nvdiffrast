# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Note: you should be able to use a newer image here:
FROM nvcr.io/nvidia/pytorch:25.06-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install ninja imageio imageio-ffmpeg

COPY csrc /tmp/nvdiffrast/csrc
COPY nvdiffrast /tmp/nvdiffrast/nvdiffrast
COPY *.py *.toml *.md *.txt /tmp/nvdiffrast
RUN TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" pip install /tmp/nvdiffrast --no-build-isolation
