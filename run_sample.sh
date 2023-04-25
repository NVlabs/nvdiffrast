#!/bin/bash

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

function print_help {
    echo "Usage: `basename $0` [--build-container] <python_file>"
    echo ""
    echo "Option --build-container will build the Docker container based on"
    echo "docker/Dockerfile and tag the image with gltorch:latest."
    echo ""
    echo "Example: `basename $0` samples/torch/envphong.py"
}

build_container=0
sample=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --build-container) build_container=1;;
        -h|--help) print_help; exit 0 ;;
        --*) echo "Unknown parameter passed: $1"; exit 1 ;;
        *) sample="$1"; shift; break;
    esac
    shift
done

rest=$@

# Build the docker container
if [ "$build_container" = "1" ]; then
    docker build --tag gltorch:latest -f docker/Dockerfile .
fi

if [ ! -f "$sample" ]; then
    echo
    echo "No python sample given or file '$sample' not found.  Exiting."
    exit 1
fi

image="gltorch:latest"

echo "Using container image: $image"
echo "Running command: $sample $rest"

# Run a sample with docker
docker run --rm -it --gpus all --user $(id -u):$(id -g) \
    -v `pwd`:/app --workdir /app -e TORCH_EXTENSIONS_DIR=/app/tmp $image python3 $sample $rest
