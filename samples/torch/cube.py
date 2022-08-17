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

# Transform vertex positions to clip space
def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def render(glctx, mtx, pos, pos_idx, vtx_col, col_idx, resolution: int):
    pos_clip    = transform_pos(mtx, pos)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    color, _    = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
    color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
    return color

def make_grid(arr, ncols=2):
    n, height, width, nc = arr.shape
    nrows = n//ncols
    assert n == nrows*ncols
    return arr.reshape(nrows, ncols, height, width, nc).swapaxes(1,2).reshape(height*nrows, width*ncols, nc)

def fit_cube(max_iter          = 5000,
             resolution        = 4,
             discontinuous     = False,
             repeats           = 1,
             log_interval      = 10,
             display_interval  = None,
             display_res       = 512,
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
            log_file = open(f'{out_dir}/{log_fn}', 'wt')
        if mp4save_interval != 0:
            writer = imageio.get_writer(f'{out_dir}/{mp4save_fn}', mode='I', fps=30, codec='libx264', bitrate='16M')
    else:
        mp4save_interval = None

    datadir = f'{pathlib.Path(__file__).absolute().parents[1]}/data'
    fn = 'cube_%s.npz' % ('d' if discontinuous else 'c')
    with np.load(f'{datadir}/{fn}') as f:
        pos_idx, vtxp, col_idx, vtxc = f.values()
    print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], vtxp.shape[0]))

    # Create position/triangle index tensors
    pos_idx = torch.from_numpy(pos_idx.astype(np.int32)).cuda()
    col_idx = torch.from_numpy(col_idx.astype(np.int32)).cuda()
    vtx_pos = torch.from_numpy(vtxp.astype(np.float32)).cuda()
    vtx_col = torch.from_numpy(vtxc.astype(np.float32)).cuda()

    # Rasterizer context
    glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()

    # Repeats.
    for rep in range(repeats):

        ang = 0.0
        gl_avg = []

        vtx_pos_rand = np.random.uniform(-0.5, 0.5, size=vtxp.shape) + vtxp
        vtx_col_rand = np.random.uniform(0.0, 1.0, size=vtxc.shape)
        vtx_pos_opt  = torch.tensor(vtx_pos_rand, dtype=torch.float32, device='cuda', requires_grad=True)
        vtx_col_opt  = torch.tensor(vtx_col_rand, dtype=torch.float32, device='cuda', requires_grad=True)

        # Adam optimizer for vertex position and color with a learning rate ramp.
        optimizer    = torch.optim.Adam([vtx_pos_opt, vtx_col_opt], lr=1e-2)
        scheduler    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.01, 10**(-x*0.0005)))

        for it in range(max_iter + 1):
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

            # Compute geometric error for logging.
            with torch.no_grad():
                geom_loss = torch.mean(torch.sum((torch.abs(vtx_pos_opt) - .5)**2, dim=1)**0.5)
                gl_avg.append(float(geom_loss))

            # Print/save log.
            if log_interval and (it % log_interval == 0):
                gl_val = np.mean(np.asarray(gl_avg))
                gl_avg = []
                s = ("rep=%d," % rep) if repeats > 1 else ""
                s += "iter=%d,err=%f" % (it, gl_val)
                print(s)
                if log_file:
                    log_file.write(s + "\n")

            color     = render(glctx, r_mvp, vtx_pos, pos_idx, vtx_col, col_idx, resolution)
            color_opt = render(glctx, r_mvp, vtx_pos_opt, pos_idx, vtx_col_opt, col_idx, resolution)

            # Compute loss and train.
            loss = torch.mean((color - color_opt)**2) # L2 pixel loss.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Show/save image.
            display_image = display_interval and (it % display_interval == 0)
            save_mp4      = mp4save_interval and (it % mp4save_interval == 0)

            if display_image or save_mp4:
                ang = ang + 0.01

                img_b = color[0].cpu().numpy()[::-1]
                img_o = color_opt[0].detach().cpu().numpy()[::-1]
                img_d = render(glctx, a_mvp, vtx_pos_opt, pos_idx, vtx_col_opt, col_idx, display_res)[0]
                img_r = render(glctx, a_mvp, vtx_pos, pos_idx, vtx_col, col_idx, display_res)[0]

                scl = display_res // img_o.shape[0]
                img_b = np.repeat(np.repeat(img_b, scl, axis=0), scl, axis=1)
                img_o = np.repeat(np.repeat(img_o, scl, axis=0), scl, axis=1)
                result_image = make_grid(np.stack([img_o, img_b, img_d.detach().cpu().numpy()[::-1], img_r.cpu().numpy()[::-1]]))

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

def main():
    parser = argparse.ArgumentParser(description='Cube fit example')
    parser.add_argument('--opengl', help='enable OpenGL rendering', action='store_true', default=False)
    parser.add_argument('--outdir', help='specify output directory', default='')
    parser.add_argument('--discontinuous', action='store_true', default=False)
    parser.add_argument('--resolution', type=int, default=0, required=True)
    parser.add_argument('--display-interval', type=int, default=0)
    parser.add_argument('--mp4save-interval', type=int, default=100)
    parser.add_argument('--max-iter', type=int, default=1000)
    args = parser.parse_args()

    # Set up logging.
    if args.outdir:
        ds = 'd' if args.discontinuous else 'c'
        out_dir = f'{args.outdir}/cube_{ds}_{args.resolution}'
        print (f'Saving results under {out_dir}')
    else:
        out_dir = None
        print ('No output directory specified, not saving log or images')

    # Run.
    fit_cube(
        max_iter=args.max_iter,
        resolution=args.resolution,
        discontinuous=args.discontinuous,
        log_interval=10,
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
