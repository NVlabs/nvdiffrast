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
import util
import pathlib

sys.path.insert(0, os.path.join(sys.path[0], '../..')) # for nvdiffrast
import nvdiffrast.tensorflow as dr

#----------------------------------------------------------------------------
# Quaternion math.
#----------------------------------------------------------------------------

# Unit quaternion.
def q_unit():
    return np.asarray([1, 0, 0, 0], np.float32)

# Get a random normalized quaternion.
def q_rnd():
    u, v, w = np.random.uniform(0.0, 1.0, size=[3])
    v *= 2.0 * np.pi
    w *= 2.0 * np.pi
    return np.asarray([(1.0-u)**0.5 * np.sin(v), (1.0-u)**0.5 * np.cos(v), u**0.5 * np.sin(w), u**0.5 * np.cos(w)], np.float32)

# Get a random quaternion from the octahedral symmetric group S_4.
_r2 = 0.5**0.5
_q_S4 = [[ 1.0, 0.0, 0.0, 0.0], [ 0.0, 1.0, 0.0, 0.0], [ 0.0, 0.0, 1.0, 0.0], [ 0.0, 0.0, 0.0, 1.0],
         [-0.5, 0.5, 0.5, 0.5], [-0.5,-0.5,-0.5, 0.5], [ 0.5,-0.5, 0.5, 0.5], [ 0.5, 0.5,-0.5, 0.5],
         [ 0.5, 0.5, 0.5, 0.5], [-0.5, 0.5,-0.5, 0.5], [ 0.5,-0.5,-0.5, 0.5], [-0.5,-0.5, 0.5, 0.5],
         [ _r2,-_r2, 0.0, 0.0], [ _r2, _r2, 0.0, 0.0], [ 0.0, 0.0, _r2, _r2], [ 0.0, 0.0,-_r2, _r2],
         [ 0.0, _r2, _r2, 0.0], [ _r2, 0.0, 0.0,-_r2], [ _r2, 0.0, 0.0, _r2], [ 0.0,-_r2, _r2, 0.0],
         [ _r2, 0.0, _r2, 0.0], [ 0.0, _r2, 0.0, _r2], [ _r2, 0.0,-_r2, 0.0], [ 0.0,-_r2, 0.0, _r2]]
def q_rnd_S4():
    return np.asarray(_q_S4[np.random.randint(24)], np.float32)

# Quaternion slerp.
def q_slerp(p, q, t):
    d = np.dot(p, q)
    if d < 0.0:
        q = -q
        d = -d
    if d > 0.999:
        a = p + t * (q-p)
        return a / np.linalg.norm(a)
    t0 = np.arccos(d)
    tt = t0 * t
    st = np.sin(tt)
    st0 = np.sin(t0)
    s1 = st / st0
    s0 = np.cos(tt) - d*s1
    return s0*p + s1*q

# Quaterion scale (slerp vs. identity quaternion).
def q_scale(q, scl):
    return q_slerp(q_unit(), q, scl)

# Quaternion product.
def q_mul(p, q):
    s1, V1 = p[0], p[1:]
    s2, V2 = q[0], q[1:]
    s = s1*s2 - np.dot(V1, V2)
    V = s1*V2 + s2*V1 + np.cross(V1, V2)
    return np.asarray([s, V[0], V[1], V[2]], np.float32)

# Angular difference between two quaternions in degrees.
def q_angle_deg(p, q):
    d = np.abs(np.dot(p, q))
    d = min(d, 1.0)
    return np.degrees(2.0 * np.arccos(d))

# Quaternion product in TensorFlow.
def q_mul_tf(p, q):
    a = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    b = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    c = p[0]*q[2] + p[2]*q[0] + p[3]*q[1] - p[1]*q[3]
    d = p[0]*q[3] + p[3]*q[0] + p[1]*q[2] - p[2]*q[1]
    return tf.stack([a, b, c, d])

# Convert quaternion to 4x4 rotation matrix. TensorFlow.
def q_to_mtx_tf(q):
    r0 = tf.stack([1.0-2.0*q[1]**2 - 2.0*q[2]**2, 2.0*q[0]*q[1] - 2.0*q[2]*q[3], 2.0*q[0]*q[2] + 2.0*q[1]*q[3]])
    r1 = tf.stack([2.0*q[0]*q[1] + 2.0*q[2]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[2]**2, 2.0*q[1]*q[2] - 2.0*q[0]*q[3]])
    r2 = tf.stack([2.0*q[0]*q[2] - 2.0*q[1]*q[3], 2.0*q[1]*q[2] + 2.0*q[0]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[1]**2])
    rr = tf.transpose(tf.stack([r0, r1, r2]), [1, 0])
    rr = tf.concat([rr, tf.convert_to_tensor([[0], [0], [0]], tf.float32)], axis=1) # Pad right column.
    rr = tf.concat([rr, tf.convert_to_tensor([[0, 0, 0, 1]], tf.float32)], axis=0)  # Pad bottom row.
    return rr

#----------------------------------------------------------------------------
# Cube pose fitter.
#----------------------------------------------------------------------------

def fit_pose(max_iter           = 10000,
             repeats            = 1,
             log_interval       = 10,
             display_interval   = None,
             display_res        = 512,
             lr_base            = 0.01,
             lr_falloff         = 1.0,
             nr_base            = 1.0,
             nr_falloff         = 1e-4,
             grad_phase_start   = 0.5,
             resolution         = 256,
             out_dir            = '.',
             log_fn             = None,
             imgsave_interval   = None,
             imgsave_fn         = None):

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    datadir = f'{pathlib.Path(__file__).absolute().parents[1]}/data'
    with np.load(f'{datadir}/cube_p.npz') as f:
        pos_idx, pos, col_idx, col = f.values()
    print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], pos.shape[0]))

    # Transformation matrix input to TF graph.
    mtx_in = tf.placeholder(tf.float32, [4, 4])

    # Pose matrix input to TF graph.    
    pose_in = tf.placeholder(tf.float32, [4]) # Quaternion.
    noise_in = tf.placeholder(tf.float32, [4]) # Mollification noise.
    
    # Setup TF graph for reference.
    mtx_total = tf.matmul(mtx_in, q_to_mtx_tf(pose_in))
    pos_clip = tf.matmul(pos, mtx_total, transpose_b=True)[tf.newaxis, ...]
    rast_out, _ = dr.rasterize(pos_clip, pos_idx, resolution=[resolution, resolution], output_db=False)
    color, _ = dr.interpolate(col[tf.newaxis, ...], rast_out, col_idx)
    color = dr.antialias(color, rast_out, pos_clip, pos_idx)

    # Setup TF graph for optimization candidate.
    pose_var = tf.get_variable('pose', initializer=tf.zeros_initializer(), shape=[4])
    pose_var_in = tf.placeholder(tf.float32, [4])
    pose_set = tf.assign(pose_var, pose_var_in)
    pose_norm_op = tf.assign(pose_var, pose_var / tf.reduce_sum(pose_var**2)**0.5) # Normalization operation.
    pose_total = q_mul_tf(pose_var, noise_in)
    mtx_total_opt = tf.matmul(mtx_in, q_to_mtx_tf(pose_total))
    pos_clip_opt = tf.matmul(pos, mtx_total_opt, transpose_b=True)[tf.newaxis, ...]
    rast_out_opt, _ = dr.rasterize(pos_clip_opt, pos_idx, resolution=[resolution, resolution], output_db=False)
    color_opt, _ = dr.interpolate(col[tf.newaxis, ...], rast_out_opt, col_idx)
    color_opt = dr.antialias(color_opt, rast_out_opt, pos_clip_opt, pos_idx)

    # Image-space loss.
    diff = (color_opt - color)**2 # L2 norm.
    diff = tf.tanh(5.0 * tf.reduce_max(diff, axis=-1)) # Add some oomph to the loss.
    loss = tf.reduce_mean(diff)
    lr_in = tf.placeholder(tf.float32, [])
    train_op = tf.train.AdamOptimizer(lr_in, 0.9, 0.999).minimize(loss, var_list=[pose_var])

    # Open log file.
    log_file = open(out_dir + '/' + log_fn, 'wt') if log_fn else None

    # Repeats.
    for rep in range(repeats):

        # Optimize.
        util.init_uninitialized_vars()
        loss_best = np.inf
        pose_best = None
        for it in range(max_iter + 1):
            # Modelview + projection matrix.
            mvp = np.matmul(util.projection(x=0.4), util.translate(0, 0, -3.5)).astype(np.float32)

            # Learning and noise rate scheduling.
            itf = 1.0 * it / max_iter
            lr = lr_base * lr_falloff**itf
            nr = nr_base * nr_falloff**itf

            # Noise input.
            if itf >= grad_phase_start:
                noise = q_unit()
            else:
                noise = q_scale(q_rnd(), nr)
                noise = q_mul(noise, q_rnd_S4()) # Orientation noise.

            # Initialize optimization.
            if it == 0:
                pose_target = q_rnd()                
                util.run(pose_set, {pose_var_in: q_rnd()})
                util.run(pose_norm_op)
                util.run(loss, {mtx_in: mvp, pose_in: pose_target, noise_in: noise}) # Pipecleaning pass.

            # Run gradient training step.
            if itf >= grad_phase_start:
                util.run(train_op, {mtx_in: mvp, pose_in: pose_target, noise_in: noise, lr_in: lr})
                util.run(pose_norm_op)

            # Measure image-space loss and update best found pose.
            loss_val = util.run(loss, {mtx_in: mvp, pose_in: pose_target, noise_in: noise, lr_in: lr})
            if loss_val < loss_best:
                pose_best = util.run(pose_total, {noise_in: noise})
                if loss_val > 0.0:
                    loss_best = loss_val
            else:
                # Return to best pose in the greedy phase.
                if itf < grad_phase_start:
                    util.run(pose_set, {pose_var_in: pose_best})

            # Print/save log.
            if log_interval and (it % log_interval == 0):
                err = q_angle_deg(util.run(pose_var), pose_target)
                ebest = q_angle_deg(pose_best, pose_target)
                s = "rep=%d,iter=%d,err=%f,err_best=%f,loss=%f,loss_best=%f,lr=%f,nr=%f" % (rep, it, err, ebest, loss_val, loss_best, lr, nr)
                print(s)
                if log_file:
                    log_file.write(s + "\n")

            # Show/save image.
            display_image = display_interval and (it % display_interval == 0)
            save_image = imgsave_interval and (it % imgsave_interval == 0)

            if display_image or save_image:
                img_ref, img_opt = util.run([color, color_opt], {mtx_in: mvp, pose_in: pose_target, noise_in: noise})
                img_best, = util.run([color_opt], {mtx_in: mvp, pose_in: pose_best, noise_in: q_unit()})
                img_ref = img_ref[0]
                img_opt = img_opt[0]
                img_best = img_best[0]
                result_image = np.concatenate([img_ref, img_best, img_opt], axis=1)

            if display_image:
                util.display_image(result_image, size=display_res, title='(%d) %d / %d' % (rep, it, max_iter))
            if save_image:
                util.save_image(out_dir + '/' + (imgsave_fn % (rep, it)), result_image)

    # All repeats done.
    if log_file:
        log_file.close()

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

def main():
    display_interval = 0
    repeats = 1

    def usage():
        print("Usage: python pose.py [-v] [repeats]")
        exit()

    for a in sys.argv[1:]:
        if a == '-v':
            display_interval = 10
        elif a.isascii() and a.isdecimal():
            repeats = int(a)
        else:
            usage()

    if repeats <= 0:
        usage()

    # Initialize TensorFlow.
    util.init_tf()

    # Run.
    fit_pose(max_iter=1000, repeats=repeats, log_interval=100, display_interval=display_interval, out_dir='out/pose', log_fn='log.txt', imgsave_interval=1000, imgsave_fn='img_%03d_%06d.png')

    # Done.
    print("Done.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
