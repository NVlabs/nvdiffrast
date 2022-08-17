# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import pathlib
import sys
import numpy as np
import torch
import imageio

import util

import nvdiffrast.torch as dr

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
    p = p.detach().cpu().numpy()
    q = q.detach().cpu().numpy()
    d = np.abs(np.dot(p, q))
    d = min(d, 1.0)
    return np.degrees(2.0 * np.arccos(d))

# Quaternion product
def q_mul_torch(p, q):
    a = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    b = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    c = p[0]*q[2] + p[2]*q[0] + p[3]*q[1] - p[1]*q[3]
    d = p[0]*q[3] + p[3]*q[0] + p[1]*q[2] - p[2]*q[1]
    return torch.stack([a, b, c, d])

# Convert quaternion to 4x4 rotation matrix.
def q_to_mtx(q):
    r0 = torch.stack([1.0-2.0*q[1]**2 - 2.0*q[2]**2, 2.0*q[0]*q[1] - 2.0*q[2]*q[3], 2.0*q[0]*q[2] + 2.0*q[1]*q[3]])
    r1 = torch.stack([2.0*q[0]*q[1] + 2.0*q[2]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[2]**2, 2.0*q[1]*q[2] - 2.0*q[0]*q[3]])
    r2 = torch.stack([2.0*q[0]*q[2] - 2.0*q[1]*q[3], 2.0*q[1]*q[2] + 2.0*q[0]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[1]**2])
    rr = torch.transpose(torch.stack([r0, r1, r2]), 1, 0)
    rr = torch.cat([rr, torch.tensor([[0], [0], [0]], dtype=torch.float32).cuda()], dim=1) # Pad right column.
    rr = torch.cat([rr, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).cuda()], dim=0)  # Pad bottom row.
    return rr

# Transform vertex positions to clip space
def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def render(glctx, mtx, pos, pos_idx, col, col_idx, resolution: int):
    # Setup TF graph for reference.
    pos_clip    = transform_pos(mtx, pos)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    color   , _ = dr.interpolate(col[None, ...], rast_out, col_idx)
    color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
    return color

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
             out_dir            = None,
             log_fn             = None,
             mp4save_interval   = None,
             mp4save_fn         = None,
             use_opengl         = False):

    log_file = None
    writer = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if log_fn:
            log_file = open(out_dir + '/' + log_fn, 'wt')
        if mp4save_interval != 0:
            writer = imageio.get_writer(f'{out_dir}/{mp4save_fn}', mode='I', fps=30, codec='libx264', bitrate='16M')
    else:
        mp4save_interval = None

    datadir = f'{pathlib.Path(__file__).absolute().parents[1]}/data'
    with np.load(f'{datadir}/cube_p.npz') as f:
        pos_idx, pos, col_idx, col = f.values()
    print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], pos.shape[0]))

    # Some input geometry contains vertex positions in (N, 4) (with v[:,3]==1).  Drop
    # the last column in that case.
    if pos.shape[1] == 4: pos = pos[:, 0:3]

    # Create position/triangle index tensors
    pos_idx = torch.from_numpy(pos_idx.astype(np.int32)).cuda()
    vtx_pos = torch.from_numpy(pos.astype(np.float32)).cuda()
    col_idx = torch.from_numpy(col_idx.astype(np.int32)).cuda()
    vtx_col = torch.from_numpy(col.astype(np.float32)).cuda()

    glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()

    for rep in range(repeats):
        pose_target = torch.tensor(q_rnd(), device='cuda')
        pose_init   = q_rnd()
        pose_opt    = torch.tensor(pose_init / np.sum(pose_init**2)**0.5, dtype=torch.float32, device='cuda', requires_grad=True)

        loss_best   = np.inf
        pose_best   = pose_opt.detach().clone()

        # Modelview + projection matrix.
        mvp = torch.tensor(np.matmul(util.projection(x=0.4), util.translate(0, 0, -3.5)).astype(np.float32), device='cuda')

        # Adam optimizer for texture with a learning rate ramp.
        optimizer = torch.optim.Adam([pose_opt], betas=(0.9, 0.999), lr=lr_base)
        # Render.
        for it in range(max_iter + 1):
            # Set learning rate.
            itf = 1.0 * it / max_iter
            nr = nr_base * nr_falloff**itf
            lr = lr_base * lr_falloff**itf
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Noise input.
            if itf >= grad_phase_start:
                noise = q_unit()
            else:
                noise = q_scale(q_rnd(), nr)
                noise = q_mul(noise, q_rnd_S4()) # Orientation noise.

            # Render.
            color          = render(glctx, torch.matmul(mvp, q_to_mtx(pose_target)), vtx_pos, pos_idx, vtx_col, col_idx, resolution)
            pose_total_opt = q_mul_torch(pose_opt, noise)
            mtx_total_opt  = torch.matmul(mvp, q_to_mtx(pose_total_opt))
            color_opt      = render(glctx, mtx_total_opt, vtx_pos, pos_idx, vtx_col, col_idx, resolution)

            # Image-space loss.
            diff = (color_opt - color)**2 # L2 norm.
            diff = torch.tanh(5.0 * torch.max(diff, dim=-1)[0])
            loss = torch.mean(diff)

            # Measure image-space loss and update best found pose.
            loss_val = float(loss)
            if (loss_val < loss_best) and (loss_val > 0.0):
                pose_best = pose_total_opt.detach().clone()
                loss_best = loss_val
                if itf < grad_phase_start:
                    with torch.no_grad(): pose_opt[:] = pose_best

            # Print/save log.
            if log_interval and (it % log_interval == 0):
                err = q_angle_deg(pose_opt, pose_target)
                ebest = q_angle_deg(pose_best, pose_target)
                s = "rep=%d,iter=%d,err=%f,err_best=%f,loss=%f,loss_best=%f,lr=%f,nr=%f" % (rep, it, err, ebest, loss_val, loss_best, lr, nr)
                print(s)
                if log_file:
                    log_file.write(s + "\n")

            # Run gradient training step.
            if itf >= grad_phase_start:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                pose_opt /= torch.sum(pose_opt**2)**0.5

            # Show/save image.
            display_image = display_interval and (it % display_interval == 0)
            save_mp4      = mp4save_interval and (it % mp4save_interval == 0)

            if display_image or save_mp4:
                img_ref  = color[0].detach().cpu().numpy()
                img_opt  = color_opt[0].detach().cpu().numpy()
                img_best = render(glctx, torch.matmul(mvp, q_to_mtx(pose_best)), vtx_pos, pos_idx, vtx_col, col_idx, resolution)[0].detach().cpu().numpy()
                result_image = np.concatenate([img_ref, img_best, img_opt], axis=1)[::-1]

                if display_image:
                    util.display_image(result_image, size=display_res, title='(%d) %d / %d' % (rep, it, max_iter))
                if save_mp4:
                    writer.append_data(np.clip(np.rint(result_image*255.0), 0, 255).astype(np.uint8))

    # Done.
    if writer is not None:
        writer.close()
    if log_file:
        log_file.close()

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Cube pose fitting example')
    parser.add_argument('--opengl', help='enable OpenGL rendering', action='store_true', default=False)
    parser.add_argument('--outdir', help='specify output directory', default='')
    parser.add_argument('--display-interval', type=int, default=0)
    parser.add_argument('--mp4save-interval', type=int, default=10)
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()

    # Set up logging.
    if args.outdir:
        out_dir = f'{args.outdir}/pose'
        print (f'Saving results under {out_dir}')
    else:
        out_dir = None
        print ('No output directory specified, not saving log or images')

    # Run.
    fit_pose(
        max_iter=args.max_iter,
        repeats=args.repeats,
        log_interval=100,
        display_interval=args.display_interval,
        out_dir=out_dir,
        log_fn='log.txt',
        mp4save_interval=args.mp4save_interval,
        mp4save_fn='progress.mp4',
        use_opengl=args.opengl
    )

    # Done.
    print("Done.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
