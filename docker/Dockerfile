# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Note: Should also work with NVIDIA's Docker image builds such as
#
# nvcr.io/nvidia/pytorch:20.09-py3
#
# This file defaults to pytorch/pytorch as it works on slightly older
# driver versions.
FROM nvcr.io/nvidia/pytorch:23.03-py3

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# for GLEW
ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Default pyopengl to EGL for good headless rendering support
ENV PYOPENGL_PLATFORM egl

COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN pip install --upgrade pip
RUN pip install ninja imageio imageio-ffmpeg

COPY nvdiffrast /tmp/pip/nvdiffrast/
COPY README.md setup.py /tmp/pip/
RUN cd /tmp/pip && pip install .
