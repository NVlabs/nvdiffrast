# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import imageio
import logging
import os
import numpy as np
import tensorflow as tf
import nvdiffrast.tensorflow as dr

# Silence deprecation warnings and debug level logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')

pos = tf.convert_to_tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=tf.float32)
col = tf.convert_to_tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=tf.float32)
tri = tf.convert_to_tensor([[0, 1, 2]], dtype=tf.int32)

rast, _ = dr.rasterize(pos, tri, resolution=[256, 256])
out, _ = dr.interpolate(col, rast, tri)

with tf.Session() as sess:
    img = sess.run(out)
    
img = img[0, ::-1, :, :] # Flip vertically.
img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8

print("Saving to 'tri.png'.")
imageio.imsave('tri.png', img)
