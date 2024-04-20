# Facial Expression Recognition

This repository is used for academic purposes and was forked from [nvdiffrast](https://github.com/NVlabs/nvdiffrast).

## Setup

* We performed the work of this project using the following AWS AMI: Deep Learning Base OSS NVIDIA Driver GPU AMI (Ubuntu 20.04) 20240410 and used a g5.xlarge instance type.
* Create the image:
```shell
$ docker build -t avafer:latest -f docker/Dockerfile .
```
* Create the container:
```shell
docker run -dit --gpus all --name avafer avafer:latest
```
* Prepare models used for inference by following the instructions from the [Deep3DFaceRecon_pytorch readme](https://github.com/sicxu/Deep3DFaceRecon_pytorch?tab=readme-ov-file#inference-with-a-pre-trained-model). 


