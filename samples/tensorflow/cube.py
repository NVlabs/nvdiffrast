# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
import sys
import pathlib

import util
import tensorflow as tf

sys.path.insert(0, os.path.join(sys.path[0], '../..')) # for nvdiffrast
import nvdiffrast.tensorflow as dr

#----------------------------------------------------------------------------
# Cube shape fitter.
#----------------------------------------------------------------------------

def fit_cube(max_iter          = 5000,
             resolution        = 4, 
             discontinuous     = False,
             repeats           = 1,
             log_interval      = 10, 
             display_interval  = None,
             display_res       = 512,
             out_dir           = '.',
             log_fn            = None,
             imgsave_interval  = None,
             imgsave_fn        = None):

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    datadir = f'{pathlib.Path(__file__).absolute().parents[1]}/data'
    fn = 'cube_%s.npz' % ('d' if discontinuous else 'c')
    with np.load(f'{datadir}/{fn}') as f:
        pos_idx, vtxp, col_idx, vtxc = f.values()
    print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], vtxp.shape[0]))
        
    # Transformation matrix input to TF graph.
    mtx_in = tf.placeholder(tf.float32, [4, 4])

    # Setup TF graph for reference.
    vtxw = np.concatenate([vtxp, np.ones([vtxp.shape[0], 1])], axis=1).astype(np.float32)
    pos_clip = tf.matmul(vtxw, mtx_in, transpose_b=True)[tf.newaxis, ...]
    rast_out, _ = dr.rasterize(pos_clip, pos_idx, resolution=[resolution, resolution], output_db=False)
    color, _ = dr.interpolate(vtxc[tf.newaxis, ...], rast_out, col_idx)
    color = dr.antialias(color, rast_out, pos_clip, pos_idx)

    # Optimized variables.
    vtxc_opt = tf.get_variable('vtxc', initializer=tf.zeros_initializer(), shape=vtxc.shape)
    vtxp_opt = tf.get_variable('vtxp', initializer=tf.zeros_initializer(), shape=vtxp.shape)

    # Optimization variable setters for initialization.
    vtxc_opt_in = tf.placeholder(tf.float32, vtxc.shape)
    vtxp_opt_in = tf.placeholder(tf.float32, vtxp.shape)
    opt_set = tf.group(tf.assign(vtxc_opt, vtxc_opt_in), tf.assign(vtxp_opt, vtxp_opt_in))

    # Setup TF graph for what we optimize result.
    vtxw_opt = tf.concat([vtxp_opt, tf.ones([vtxp.shape[0], 1], tf.float32)], axis=1)
    pos_clip_opt = tf.matmul(vtxw_opt, mtx_in, transpose_b=True)[tf.newaxis, ...]
    rast_out_opt, _ = dr.rasterize(pos_clip_opt, pos_idx, resolution=[resolution, resolution], output_db=False)
    color_opt, _ = dr.interpolate(vtxc_opt[tf.newaxis, ...], rast_out_opt, col_idx)
    color_opt = dr.antialias(color_opt, rast_out_opt, pos_clip_opt, pos_idx)

    # Image-space loss and optimizer.
    loss = tf.reduce_mean((color_opt - color)**2)
    lr_in = tf.placeholder(tf.float32, [])
    train_op = tf.train.AdamOptimizer(lr_in, 0.9, 0.999).minimize(loss, var_list=[vtxp_opt, vtxc_opt])

    # Setup TF graph for display.
    rast_out_disp, _ = dr.rasterize(pos_clip_opt, pos_idx, resolution=[display_res, display_res], output_db=False)
    color_disp, _ = dr.interpolate(vtxc_opt[tf.newaxis, ...], rast_out_disp, col_idx)
    color_disp = dr.antialias(color_disp, rast_out_disp, pos_clip_opt, pos_idx)
    rast_out_disp_ref, _ = dr.rasterize(pos_clip, pos_idx, resolution=[display_res, display_res], output_db=False)
    color_disp_ref, _ = dr.interpolate(vtxc[tf.newaxis, ...], rast_out_disp_ref, col_idx)
    color_disp_ref = dr.antialias(color_disp_ref, rast_out_disp_ref, pos_clip, pos_idx)

    # Geometric error calculation
    geom_loss = tf.reduce_mean(tf.reduce_sum((tf.abs(vtxp_opt) - .5)**2, axis=1)**0.5)

    # Open log file.
    log_file = open(out_dir + '/' + log_fn, 'wt') if log_fn else None

    # Repeats.
    for rep in range(repeats):

        # Optimize.
        ang = 0.0
        gl_avg = []
        util.init_uninitialized_vars()
        for it in range(max_iter + 1):
            # Initialize optimization.
            if it == 0:
                vtxp_init = np.random.uniform(-0.5, 0.5, size=vtxp.shape) + vtxp
                vtxc_init = np.random.uniform(0.0, 1.0, size=vtxc.shape)
                util.run(opt_set, {vtxc_opt_in: vtxc_init.astype(np.float32), vtxp_opt_in: vtxp_init.astype(np.float32)})

            # Learning rate ramp.
            lr = 1e-2
            lr = lr * max(0.01, 10**(-it*0.0005))

            # Random rotation/translation matrix for optimization.
            r_rot = util.random_rotation_translation(0.25)

            # Smooth rotation for display.
            a_rot = np.matmul(util.rotate_x(-0.4), util.rotate_y(ang))

            # Modelview and modelview + projection matrices.
            proj  = util.projection(x=0.4)
            r_mv  = np.matmul(util.translate(0, 0, -3.5), r_rot)
            r_mvp = np.matmul(proj, r_mv).astype(np.float32)
            a_mv  = np.matmul(util.translate(0, 0, -3.5), a_rot)
            a_mvp = np.matmul(proj, a_mv).astype(np.float32)
        
            # Run training and measure geometric error.
            gl_val, _ = util.run([geom_loss, train_op], {mtx_in: r_mvp, lr_in: lr})
            gl_avg.append(gl_val)

            # Print/save log.
            if log_interval and (it % log_interval == 0):
                gl_val, gl_avg = np.mean(np.asarray(gl_avg)), []
                s = ("rep=%d," % rep) if repeats > 1 else ""
                s += "iter=%d,err=%f" % (it, gl_val)
                print(s)
                if log_file:
                    log_file.write(s + "\n")

            # Show/save image.
            display_image = display_interval and (it % display_interval == 0)
            save_image = imgsave_interval and (it % imgsave_interval == 0)

            if display_image or save_image:
                ang = ang + 0.1
                img_o = util.run(color_opt,      {mtx_in: r_mvp})[0]
                img_b = util.run(color,          {mtx_in: r_mvp})[0]
                img_d = util.run(color_disp,     {mtx_in: a_mvp})[0]
                img_r = util.run(color_disp_ref, {mtx_in: a_mvp})[0]

                scl = display_res // img_o.shape[0]
                img_b = np.repeat(np.repeat(img_b, scl, axis=0), scl, axis=1)
                img_o = np.repeat(np.repeat(img_o, scl, axis=0), scl, axis=1)
                result_image = np.concatenate([img_o, img_b, img_d, img_r], axis=1)

            if display_image:
                util.display_image(result_image, size=display_res, title='%d / %d' % (it, max_iter))
            if save_image:
                util.save_image(out_dir + '/' + (imgsave_fn % it), result_image)

    # All repeats done.
    if log_file:
        log_file.close()

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

def main():
    display_interval = 0
    discontinuous = False
    resolution = 0

    def usage():
        print("Usage: python cube.py [-v] [-discontinuous] resolution")
        exit()

    for a in sys.argv[1:]:
        if a == '-v':
            display_interval = 100
        elif a == '-discontinuous':
            discontinuous = True
        elif a.isdecimal():
            resolution = int(a)
        else:
            usage()

    if resolution <= 0:
        usage()

    # Initialize TensorFlow.
    util.init_tf()

    # Run.
    out_dir = 'out/cube_%s_%d' % (('d' if discontinuous else 'c'), resolution)
    fit_cube(max_iter=5000, resolution=resolution, discontinuous=discontinuous, log_interval=10, display_interval=display_interval, out_dir=out_dir, log_fn='log.txt', imgsave_interval=1000, imgsave_fn='img_%06d.png')

    # Done.
    print("Done.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
