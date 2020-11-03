# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import tensorflow as tf
import os
import sys
import pathlib

import util

sys.path.insert(0, os.path.join(sys.path[0], '../..')) # for nvdiffrast
import nvdiffrast.tensorflow as dr

#----------------------------------------------------------------------------
# Environment map and Phong BRDF learning.
#----------------------------------------------------------------------------

def fit_env_phong(max_iter          = 1000,
                  log_interval      = 10,
                  display_interval  = None,
                  display_res       = 1024,
                  res               = 1024,
                  lr_base           = 1e-2,
                  lr_ramp           = 1.0,
                  out_dir           = '.',
                  log_fn            = None,
                  imgsave_interval  = None,
                  imgsave_fn        = None):

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Texture adapted from https://github.com/WaveEngine/Samples/tree/master/Materials/EnvironmentMap/Content/Assets/CubeMap.cubemap
    datadir = f'{pathlib.Path(__file__).absolute().parents[1]}/data'
    with np.load(f'{datadir}/envphong.npz') as f:
        pos_idx, pos, normals, env = f.values()
    env = env.astype(np.float32)/255.0
    print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], pos.shape[0]))

    # Target Phong parameters.
    phong_rgb = np.asarray([1.0, 0.8, 0.6], np.float32)
    phong_exp = 25.0

    # Inputs to TF graph.
    mtx_in = tf.placeholder(tf.float32, [4, 4])
    invmtx_in = tf.placeholder(tf.float32, [4, 4]) # Inverse.
    campos_in = tf.placeholder(tf.float32, [3]) # Camera position in world space.
    lightdir_in = tf.placeholder(tf.float32, [3]) # Light direction.

    # Learned variables: environment maps, phong color, phong exponent.
    env_var = tf.get_variable('env_var', initializer=tf.constant_initializer(0.5), shape=env.shape)
    phong_var_raw = tf.get_variable('phong_var', initializer=tf.random_uniform_initializer(0.0, 1.0), shape=[4]) # R, G, B, exp.
    phong_var = phong_var_raw * [1.0, 1.0, 1.0, 10.0] # Faster learning rate for the exponent.

    # Transform and rasterize.
    viewvec = pos[..., :3] - campos_in[np.newaxis, np.newaxis, :] # View vectors at vertices.
    reflvec = viewvec - 2.0 * normals[tf.newaxis, ...] * tf.reduce_sum(normals[tf.newaxis, ...] * viewvec, axis=-1, keepdims=True) # Reflection vectors at vertices.
    reflvec = reflvec / tf.reduce_sum(reflvec**2, axis=-1, keepdims=True)**0.5 # Normalize.
    pos_clip = tf.matmul(pos, mtx_in, transpose_b=True)[tf.newaxis, ...]
    rast_out, rast_out_db = dr.rasterize(pos_clip, pos_idx, [res, res])
    refl, refld = dr.interpolate(reflvec, rast_out, pos_idx, rast_db=rast_out_db, diff_attrs='all') # Interpolated reflection vectors.
    
    # Phong light.
    refl = refl / tf.reduce_sum(refl**2, axis=-1, keepdims=True)**0.5  # Normalize.
    ldotr = tf.reduce_sum(-lightdir_in * refl, axis=-1, keepdims=True) # L dot R.

    # Reference color. No need for AA because we are not learning geometry.
    env = np.stack(env)[:, ::-1]
    color = dr.texture(env[np.newaxis, ...], refl, refld, filter_mode='linear-mipmap-linear', boundary_mode='cube')
    color = tf.reduce_sum(tf.stack(color), axis=0)
    color = color + phong_rgb * tf.maximum(0.0, ldotr) ** phong_exp # Phong.
    color = tf.maximum(color, 1.0 - tf.clip_by_value(rast_out[..., -1:], 0, 1)) # White background.

    # Candidate rendering same up to this point, but uses learned texture and Phong parameters instead.
    color_opt = dr.texture(env_var[tf.newaxis, ...], refl, uv_da=refld, filter_mode='linear-mipmap-linear', boundary_mode='cube')
    color_opt = tf.reduce_sum(tf.stack(color_opt), axis=0)
    color_opt = color_opt + phong_var[:3] * tf.maximum(0.0, ldotr) ** phong_var[3] # Phong.
    color_opt = tf.maximum(color_opt, 1.0 - tf.clip_by_value(rast_out[..., -1:], 0, 1)) # White background.

    # Training.
    loss = tf.reduce_mean((color - color_opt)**2) # L2 pixel loss.
    lr_in = tf.placeholder(tf.float32, [])
    train_op = tf.train.AdamOptimizer(lr_in, 0.9, 0.99).minimize(loss, var_list=[env_var, phong_var_raw])

    # Open log file.
    log_file = open(out_dir + '/' + log_fn, 'wt') if log_fn else None

    # Render.
    ang = 0.0
    util.init_uninitialized_vars()
    imgloss_avg, phong_avg = [], []
    for it in range(max_iter + 1):
        lr = lr_base * lr_ramp**(float(it)/float(max_iter))

        # Random rotation/translation matrix for optimization.
        r_rot = util.random_rotation_translation(0.25)

        # Smooth rotation for display.
        ang = ang + 0.01
        a_rot = np.matmul(util.rotate_x(-0.4), util.rotate_y(ang))

        # Modelview and modelview + projection matrices.
        proj  = util.projection(x=0.4, n=1.0, f=200.0)
        r_mv  = np.matmul(util.translate(0, 0, -3.5), r_rot)
        r_mvp = np.matmul(proj, r_mv).astype(np.float32)
        a_mv  = np.matmul(util.translate(0, 0, -3.5), a_rot)
        a_mvp = np.matmul(proj, a_mv).astype(np.float32)
    
        # Solve camera positions.
        a_campos = np.linalg.inv(a_mv)[:3, 3]
        r_campos = np.linalg.inv(r_mv)[:3, 3]

        # Random light direction.        
        lightdir = np.random.normal(size=[3])
        lightdir /= np.linalg.norm(lightdir) + 1e-8

        # Run training and measure image-space RMSE loss.
        imgloss_val, phong_val, _ = util.run([loss, phong_var, train_op], {mtx_in: r_mvp, invmtx_in: np.linalg.inv(r_mvp), campos_in: r_campos, lightdir_in: lightdir, lr_in: lr})
        imgloss_avg.append(imgloss_val**0.5)
        phong_avg.append(phong_val)

        # Print/save log.
        if log_interval and (it % log_interval == 0):
            imgloss_val, imgloss_avg = np.mean(np.asarray(imgloss_avg, np.float32)), []
            phong_val, phong_avg = np.mean(np.asarray(phong_avg, np.float32), axis=0), []
            phong_rgb_rmse = np.mean((phong_val[:3] - phong_rgb)**2)**0.5
            phong_exp_rel_err = np.abs(phong_val[3] - phong_exp)/phong_exp
            s = "iter=%d,phong_rgb_rmse=%f,phong_exp_rel_err=%f,img_rmse=%f" % (it, phong_rgb_rmse, phong_exp_rel_err, imgloss_val)
            print(s)
            if log_file:
                log_file.write(s + '\n')

        # Show/save result image.        
        display_image = display_interval and (it % display_interval == 0)
        save_image = imgsave_interval and (it % imgsave_interval == 0)

        if display_image or save_image:
            result_image = util.run(color_opt, {mtx_in: a_mvp, invmtx_in: np.linalg.inv(a_mvp), campos_in: a_campos, lightdir_in: lightdir})[0]
        if display_image:
            util.display_image(result_image, size=display_res, title='%d / %d' % (it, max_iter))
        if save_image:
            util.save_image(out_dir + '/' + (imgsave_fn % it), result_image)

    # Done.
    if log_file:
        log_file.close()

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

def main():
    display_interval = 0
    for a in sys.argv[1:]:
        if a == '-v':
            display_interval = 10
        else:
            print("Usage: python envphong.py [-v]")
            exit()

    # Initialize TensorFlow.        
    util.init_tf()

    # Run.
    fit_env_phong(max_iter=1500, log_interval=10, display_interval=display_interval, out_dir='out/env_phong', log_fn='log.txt', imgsave_interval=100, imgsave_fn='img_%06d.png')

    # Done.
    print("Done.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
