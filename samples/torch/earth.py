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

import util

import nvdiffrast.torch as dr

#----------------------------------------------------------------------------
# Helpers.

def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def render(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution, enable_mip, max_mip_level):
    pos_clip = transform_pos(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])

    if enable_mip:
        texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        color = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        color = dr.texture(tex[None, ...], texc, filter_mode='linear')

    color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background.
    return color

#----------------------------------------------------------------------------

def fit_earth(max_iter          = 20000,
              log_interval      = 10,
              display_interval  = None,
              display_res       = 1024,
              enable_mip        = True,
              res               = 512,
              ref_res           = 2048,  # Dropped from 4096 to 2048 to allow using the Cuda rasterizer.
              lr_base           = 1e-2,
              lr_ramp           = 0.1,
              out_dir           = None,
              log_fn            = None,
              texsave_interval  = None,
              texsave_fn        = None,
              imgsave_interval  = None,
              imgsave_fn        = None,
              use_opengl        = False):

    log_file = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if log_fn:
            log_file = open(out_dir + '/' + log_fn, 'wt')
    else:
        imgsave_interval, texsave_interval = None, None

    # Mesh and texture adapted from "3D Earth Photorealistic 2K" model at
    # https://www.turbosquid.com/3d-models/3d-realistic-earth-photorealistic-2k-1279125
    datadir = f'{pathlib.Path(__file__).absolute().parents[1]}/data'
    with np.load(f'{datadir}/earth.npz') as f:
        pos_idx, pos, uv_idx, uv, tex = f.values()
    tex = tex.astype(np.float32)/255.0
    max_mip_level = 9 # Texture is a 4x3 atlas of 512x512 maps.
    print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], pos.shape[0]))

    # Some input geometry contains vertex positions in (N, 4) (with v[:,3]==1).  Drop
    # the last column in that case.
    if pos.shape[1] == 4: pos = pos[:, 0:3]

    # Create position/triangle index tensors
    pos_idx = torch.from_numpy(pos_idx.astype(np.int32)).cuda()
    vtx_pos = torch.from_numpy(pos.astype(np.float32)).cuda()
    uv_idx  = torch.from_numpy(uv_idx.astype(np.int32)).cuda()
    vtx_uv  = torch.from_numpy(uv.astype(np.float32)).cuda()

    tex     = torch.from_numpy(tex.astype(np.float32)).cuda()
    tex_opt = torch.full(tex.shape, 0.2, device='cuda', requires_grad=True)
    glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()

    ang = 0.0

    # Adam optimizer for texture with a learning rate ramp.
    optimizer    = torch.optim.Adam([tex_opt], lr=lr_base)
    scheduler    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_ramp**(float(x)/float(max_iter)))

    # Render.
    ang = 0.0
    texloss_avg = []
    for it in range(max_iter + 1):
        # Random rotation/translation matrix for optimization.
        r_rot = util.random_rotation_translation(0.25)

        # Smooth rotation for display.
        a_rot = np.matmul(util.rotate_x(-0.4), util.rotate_y(ang))
        dist = np.random.uniform(0.0, 48.5)

        # Modelview and modelview + projection matrices.
        proj  = util.projection(x=0.4, n=1.0, f=200.0)
        r_mv  = np.matmul(util.translate(0, 0, -1.5-dist), r_rot)
        r_mvp = np.matmul(proj, r_mv).astype(np.float32)
        a_mv  = np.matmul(util.translate(0, 0, -3.5), a_rot)
        a_mvp = np.matmul(proj, a_mv).astype(np.float32)

        # Measure texture-space RMSE loss
        with torch.no_grad():
            texmask = torch.zeros_like(tex)
            tr = tex.shape[1]//4
            texmask[tr+13:2*tr-13, 25:-25, :] += 1.0
            texmask[25:-25, tr+13:2*tr-13, :] += 1.0
            # Measure only relevant portions of texture when calculating texture
            # PSNR.
            texloss = (torch.sum(texmask * (tex - tex_opt)**2)/torch.sum(texmask))**0.5 # RMSE within masked area.
            texloss_avg.append(float(texloss))

        # Render reference and optimized frames. Always enable mipmapping for reference.
        color = render(glctx, r_mvp, vtx_pos, pos_idx, vtx_uv, uv_idx, tex, ref_res, True, max_mip_level)
        color_opt = render(glctx, r_mvp, vtx_pos, pos_idx, vtx_uv, uv_idx, tex_opt, res, enable_mip, max_mip_level)

        # Reduce the reference to correct size.
        while color.shape[1] > res:
            color = util.bilinear_downsample(color)

        # Compute loss and perform a training step.
        loss = torch.mean((color - color_opt)**2) # L2 pixel loss.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Print/save log.
        if log_interval and (it % log_interval == 0):
            texloss_val = np.mean(np.asarray(texloss_avg))
            texloss_avg = []
            psnr = -10.0 * np.log10(texloss_val**2) # PSNR based on average RMSE.
            s = "iter=%d,loss=%f,psnr=%f" % (it, texloss_val, psnr)
            print(s)
            if log_file:
                log_file.write(s + '\n')

        # Show/save image.
        display_image = display_interval and (it % display_interval == 0)
        save_image = imgsave_interval and (it % imgsave_interval == 0)
        save_texture = texsave_interval and (it % texsave_interval) == 0

        if display_image or save_image:
            ang = ang + 0.1

            with torch.no_grad():
                result_image = render(glctx, a_mvp, vtx_pos, pos_idx, vtx_uv, uv_idx, tex_opt, res, enable_mip, max_mip_level)[0].cpu().numpy()[::-1]

                if display_image:
                    util.display_image(result_image, size=display_res, title='%d / %d' % (it, max_iter))
                if save_image:
                    util.save_image(out_dir + '/' + (imgsave_fn % it), result_image)

                if save_texture:
                    texture = tex_opt.cpu().numpy()[::-1]
                    util.save_image(out_dir + '/' + (texsave_fn % it), texture)


    # Done.
    if log_file:
        log_file.close()

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Earth texture fitting example')
    parser.add_argument('--opengl', help='enable OpenGL rendering', action='store_true', default=False)
    parser.add_argument('--outdir', help='specify output directory', default='')
    parser.add_argument('--mip', help='enable mipmapping', action='store_true', default=False)
    parser.add_argument('--display-interval', type=int, default=0)
    parser.add_argument('--max-iter', type=int, default=10000)
    args = parser.parse_args()

    # Set up logging.
    if args.outdir:
        ms = 'mip' if args.mip else 'nomip'
        out_dir = f'{args.outdir}/earth_{ms}'
        print (f'Saving results under {out_dir}')
    else:
        out_dir = None
        print ('No output directory specified, not saving log or images')

    # Run.
    fit_earth(max_iter=args.max_iter, log_interval=10, display_interval=args.display_interval, enable_mip=args.mip, out_dir=out_dir, log_fn='log.txt', texsave_interval=1000, texsave_fn='tex_%06d.png', imgsave_interval=1000, imgsave_fn='img_%06d.png', use_opengl=args.opengl)

    # Done.
    print("Done.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
