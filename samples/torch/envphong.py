# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import numpy as np
import torch
import os
import sys
import pathlib
import imageio

import util

import nvdiffrast.torch as dr

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
                  out_dir           = None,
                  log_fn            = None,
                  mp4save_interval  = None,
                  mp4save_fn        = None,
                  use_opengl        = False):

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

    # Texture adapted from https://github.com/WaveEngine/Samples/tree/master/Materials/EnvironmentMap/Content/Assets/CubeMap.cubemap
    datadir = f'{pathlib.Path(__file__).absolute().parents[1]}/data'
    with np.load(f'{datadir}/envphong.npz') as f:
        pos_idx, pos, normals, env = f.values()
    env = env.astype(np.float32)/255.0
    env = np.stack(env)[:, ::-1].copy()
    print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], pos.shape[0]))

    # Move all the stuff to GPU.
    pos_idx = torch.as_tensor(pos_idx, dtype=torch.int32, device='cuda')
    pos = torch.as_tensor(pos, dtype=torch.float32, device='cuda')
    normals = torch.as_tensor(normals, dtype=torch.float32, device='cuda')
    env = torch.as_tensor(env, dtype=torch.float32, device='cuda')

    # Target Phong parameters.
    phong_rgb = np.asarray([1.0, 0.8, 0.6], np.float32)
    phong_exp = 25.0
    phong_rgb_t = torch.as_tensor(phong_rgb, dtype=torch.float32, device='cuda')

    # Learned variables: environment maps, phong color, phong exponent.
    env_var = torch.ones_like(env) * .5
    env_var.requires_grad_()
    phong_var_raw = torch.as_tensor(np.random.uniform(size=[4]), dtype=torch.float32, device='cuda')
    phong_var_raw.requires_grad_()
    phong_var_mul = torch.as_tensor([1.0, 1.0, 1.0, 10.0], dtype=torch.float32, device='cuda')

    # Render.
    ang = 0.0
    imgloss_avg, phong_avg = [], []
    glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
    zero_tensor = torch.as_tensor(0.0, dtype=torch.float32, device='cuda')
    one_tensor = torch.as_tensor(1.0, dtype=torch.float32, device='cuda')

    # Adam optimizer for environment map and phong with a learning rate ramp.
    optimizer = torch.optim.Adam([env_var, phong_var_raw], lr=lr_base)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_ramp**(float(x)/float(max_iter)))

    for it in range(max_iter + 1):
        phong_var = phong_var_raw * phong_var_mul

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
        a_mvc = a_mvp
        r_mvp = torch.as_tensor(r_mvp, dtype=torch.float32, device='cuda')
        a_mvp = torch.as_tensor(a_mvp, dtype=torch.float32, device='cuda')

        # Solve camera positions.
        a_campos = torch.as_tensor(np.linalg.inv(a_mv)[:3, 3], dtype=torch.float32, device='cuda')
        r_campos = torch.as_tensor(np.linalg.inv(r_mv)[:3, 3], dtype=torch.float32, device='cuda')

        # Random light direction.        
        lightdir = np.random.normal(size=[3])
        lightdir /= np.linalg.norm(lightdir) + 1e-8
        lightdir = torch.as_tensor(lightdir, dtype=torch.float32, device='cuda')

        def render_refl(ldir, cpos, mvp):
            # Transform and rasterize.
            viewvec = pos[..., :3] - cpos[np.newaxis, np.newaxis, :] # View vectors at vertices.
            reflvec = viewvec - 2.0 * normals[np.newaxis, ...] * torch.sum(normals[np.newaxis, ...] * viewvec, -1, keepdim=True) # Reflection vectors at vertices.
            reflvec = reflvec / torch.sum(reflvec**2, -1, keepdim=True)**0.5 # Normalize.
            pos_clip = torch.matmul(pos, mvp.t())[np.newaxis, ...]
            rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, [res, res])
            refl, refld = dr.interpolate(reflvec, rast_out, pos_idx, rast_db=rast_out_db, diff_attrs='all') # Interpolated reflection vectors.

            # Phong light.
            refl = refl / (torch.sum(refl**2, -1, keepdim=True) + 1e-8)**0.5  # Normalize.
            ldotr = torch.sum(-ldir * refl, -1, keepdim=True) # L dot R.

            # Return
            return refl, refld, ldotr, (rast_out[..., -1:] == 0)

        # Render the reflections.
        refl, refld, ldotr, mask = render_refl(lightdir, r_campos, r_mvp)

        # Reference color. No need for AA because we are not learning geometry.
        color = dr.texture(env[np.newaxis, ...], refl, uv_da=refld, filter_mode='linear-mipmap-linear', boundary_mode='cube')
        color = color + phong_rgb_t * torch.max(zero_tensor, ldotr) ** phong_exp # Phong.
        color = torch.where(mask, one_tensor, color) # White background.

        # Candidate rendering same up to this point, but uses learned texture and Phong parameters instead.
        color_opt = dr.texture(env_var[np.newaxis, ...], refl, uv_da=refld, filter_mode='linear-mipmap-linear', boundary_mode='cube')
        color_opt = color_opt + phong_var[:3] * torch.max(zero_tensor, ldotr) ** phong_var[3] # Phong.
        color_opt = torch.where(mask, one_tensor, color_opt) # White background.

        # Compute loss and train.
        loss = torch.mean((color - color_opt)**2) # L2 pixel loss.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Collect losses.
        imgloss_avg.append(loss.detach().cpu().numpy())
        phong_avg.append(phong_var.detach().cpu().numpy())

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
        save_mp4 = mp4save_interval and (it % mp4save_interval == 0)

        if display_image or save_mp4:
            lightdir = np.asarray([.8, -1., .5, 0.0])
            lightdir = np.matmul(a_mvc, lightdir)[:3]
            lightdir /= np.linalg.norm(lightdir)
            lightdir = torch.as_tensor(lightdir, dtype=torch.float32, device='cuda')
            refl, refld, ldotr, mask = render_refl(lightdir, a_campos, a_mvp)
            color_opt = dr.texture(env_var[np.newaxis, ...], refl, uv_da=refld, filter_mode='linear-mipmap-linear', boundary_mode='cube')
            color_opt = color_opt + phong_var[:3] * torch.max(zero_tensor, ldotr) ** phong_var[3]
            color_opt = torch.where(mask, one_tensor, color_opt)
            result_image = color_opt.detach()[0].cpu().numpy()[::-1]
            if display_image:
                util.display_image(result_image, size=display_res, title='%d / %d' % (it, max_iter))
            if save_mp4:
                writer.append_data(np.clip(np.rint(result_image*255.0), 0, 255).astype(np.uint8))

    # Done.
    if writer is not None:
        writer.close()
    if log_file:
        log_file.close()

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Environment map fitting example')
    parser.add_argument('--opengl', help='enable OpenGL rendering', action='store_true', default=False)
    parser.add_argument('--outdir', help='specify output directory', default='')
    parser.add_argument('--display-interval', type=int, default=0)
    parser.add_argument('--mp4save-interval', type=int, default=10)
    parser.add_argument('--max-iter', type=int, default=5000)
    args = parser.parse_args()

    # Set up logging.
    if args.outdir:
        out_dir = f'{args.outdir}/env_phong'
        print (f'Saving results under {out_dir}')
    else:
        out_dir = None
        print ('No output directory specified, not saving log or images')

    # Run.
    fit_env_phong(
        max_iter=args.max_iter,
        log_interval=100,
        display_interval=args.display_interval,
        out_dir=out_dir,
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
