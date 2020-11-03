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
# Texture learning with/without mipmaps.
#----------------------------------------------------------------------------

def fit_earth(max_iter          = 20000,
              log_interval      = 10,
              display_interval  = None,
              display_res       = 1024,
              enable_mip        = True,
              res               = 512,
              ref_res           = 4096,
              lr_base           = 1e-2,
              lr_ramp           = 0.1,
              out_dir           = '.',
              log_fn            = None,
              texsave_interval  = None,
              texsave_fn        = None,
              imgsave_interval  = None,
              imgsave_fn        = None):

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Mesh and texture adapted from "3D Earth Photorealistic 2K" model at
    # https://www.turbosquid.com/3d-models/3d-realistic-earth-photorealistic-2k-1279125
    datadir = f'{pathlib.Path(__file__).absolute().parents[1]}/data'
    with np.load(f'{datadir}/earth.npz') as f:
        pos_idx, pos, uv_idx, uv, tex = f.values()
    tex = tex.astype(np.float32)/255.0
    max_mip_level = 9 # Texture is a 4x3 atlas of 512x512 maps.
    print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], pos.shape[0]))

    # Transformation matrix input to TF graph.
    mtx_in = tf.placeholder(tf.float32, [4, 4])

    # Learned texture.
    tex_var = tf.get_variable('tex', initializer=tf.constant_initializer(0.2), shape=tex.shape)

    # Setup TF graph for reference rendering in high resolution.
    pos_clip = tf.matmul(pos, mtx_in, transpose_b=True)[tf.newaxis, ...]
    rast_out, rast_out_db = dr.rasterize(pos_clip, pos_idx, [ref_res, ref_res])
    texc, texd = dr.interpolate(uv[tf.newaxis, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
    color = dr.texture(tex[np.newaxis], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    color = color * tf.clip_by_value(rast_out[..., -1:], 0, 1) # Mask out background.
    
    # Reduce the reference to correct size.
    while color.shape[1] > res:
        color = util.bilinear_downsample(color)

    # TF Graph for rendered candidate.
    if enable_mip:
        # With mipmaps.
        rast_out_opt, rast_out_db_opt = dr.rasterize(pos_clip, pos_idx, [res, res])
        texc_opt, texd_opt = dr.interpolate(uv[tf.newaxis, ...], rast_out_opt, uv_idx, rast_db=rast_out_db_opt, diff_attrs='all')
        color_opt = dr.texture(tex_var[np.newaxis], texc_opt, texd_opt, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        # No mipmaps: no image-space derivatives anywhere.
        rast_out_opt, _ = dr.rasterize(pos_clip, pos_idx, [res, res], output_db=False)
        texc_opt, _ = dr.interpolate(uv[tf.newaxis, ...], rast_out_opt, uv_idx)
        color_opt = dr.texture(tex_var[np.newaxis], texc_opt, filter_mode='linear')    
    color_opt = color_opt * tf.clip_by_value(rast_out_opt[..., -1:], 0, 1) # Mask out background.

    # Measure only relevant portions of texture when calculating texture PSNR.
    loss = tf.reduce_mean((color - color_opt)**2)
    texmask = np.zeros_like(tex)
    tr = tex.shape[1]//4
    texmask[tr+13:2*tr-13, 25:-25, :] += 1.0
    texmask[25:-25, tr+13:2*tr-13, :] += 1.0
    texloss = (tf.reduce_sum(texmask * (tex - tex_var)**2)/np.sum(texmask))**0.5 # RMSE within masked area.

    # Training driven by image-space loss.
    lr_in = tf.placeholder(tf.float32, [])
    train_op = tf.train.AdamOptimizer(lr_in, 0.9, 0.99).minimize(loss, var_list=[tex_var])

    # Open log file.
    log_file = open(out_dir + '/' + log_fn, 'wt') if log_fn else None

    # Render.
    ang = 0.0
    util.init_uninitialized_vars()
    texloss_avg = []
    for it in range(max_iter + 1):
        lr = lr_base * lr_ramp**(float(it)/float(max_iter))

        # Random rotation/translation matrix for optimization.
        r_rot = util.random_rotation_translation(0.25)

        # Smooth rotation for display.
        ang = ang + 0.01
        a_rot = np.matmul(util.rotate_x(-0.4), util.rotate_y(ang))
        dist = np.random.uniform(0.0, 48.5)

        # Modelview and modelview + projection matrices.
        proj  = util.projection(x=0.4, n=1.0, f=200.0)
        r_mv  = np.matmul(util.translate(0, 0, -1.5-dist), r_rot)
        r_mvp = np.matmul(proj, r_mv).astype(np.float32)
        a_mv  = np.matmul(util.translate(0, 0, -3.5), a_rot)
        a_mvp = np.matmul(proj, a_mv).astype(np.float32)
    
        # Run training and measure texture-space RMSE loss.
        texloss_val, _ = util.run([texloss, train_op], {mtx_in: r_mvp, lr_in: lr})
        texloss_avg.append(texloss_val)

        # Print/save log.
        if log_interval and (it % log_interval == 0):
            texloss_val, texloss_avg = np.mean(np.asarray(texloss_avg)), []
            psnr = -10.0 * np.log10(texloss_val**2) # PSNR based on average RMSE.
            s = "iter=%d,loss=%f,psnr=%f" % (it, texloss_val, psnr)
            print(s)
            if log_file:
                log_file.write(s + '\n')

        # Show/save result images/textures.
        display_image = display_interval and (it % display_interval) == 0
        save_image = imgsave_interval and (it % imgsave_interval) == 0
        save_texture = texsave_interval and (it % texsave_interval) == 0

        if display_image or save_image:
            result_image = util.run(color_opt, {mtx_in: a_mvp})[0]
        if display_image:
            util.display_image(result_image, size=display_res, title='%d / %d' % (it, max_iter))
        if save_image:
            util.save_image(out_dir + '/' + (imgsave_fn % it), result_image)
        if save_texture:
            util.save_image(out_dir + '/' + (texsave_fn % it), util.run(tex_var)[::-1])

    # Done.
    if log_file:
        log_file.close()

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

def main():
    display_interval = 0
    enable_mip = None

    def usage():
        print("Usage: python earth.py [-v] [-mip|-nomip]")
        exit()

    for a in sys.argv[1:]:
        if   a == '-v':     display_interval = 10
        elif a == '-mip':   enable_mip = True
        elif a == '-nomip': enable_mip = False
        else:               usage()

    if enable_mip is None:
        usage()

    # Initialize TensorFlow.        
    util.init_tf()

    # Run.
    out_dir = 'out/earth_mip' if enable_mip else 'out/earth_nomip'
    fit_earth(max_iter=20000, log_interval=10, display_interval=display_interval, enable_mip=enable_mip, out_dir=out_dir, log_fn='log.txt', texsave_interval=1000, texsave_fn='tex_%06d.png', imgsave_interval=1000, imgsave_fn='img_%06d.png')

    # Done.
    print("Done.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
