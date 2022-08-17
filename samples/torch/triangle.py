# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import imageio
import numpy as np
import torch
import nvdiffrast.torch as dr
import sys

def tensor(*args, **kwargs):
    return torch.tensor(*args, device='cuda', **kwargs)

if sys.argv[1:] == ['--cuda']:
    glctx = dr.RasterizeCudaContext()
elif sys.argv[1:] == ['--opengl']:
    glctx = dr.RasterizeGLContext()
else:
    print("Specify either --cuda or --opengl")
    exit(1)

pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32)
col = tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.float32)
tri = tensor([[0, 1, 2]], dtype=torch.int32)

rast, _ = dr.rasterize(glctx, pos, tri, resolution=[256, 256])
out, _ = dr.interpolate(col, rast, tri)

img = out.cpu().numpy()[0, ::-1, :, :] # Flip vertically.
img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8

print("Saving to 'tri.png'.")
imageio.imsave('tri.png', img)
