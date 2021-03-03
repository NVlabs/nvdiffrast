# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import tensorflow as tf
import numpy as np
import os
from . import plugin_loader

#----------------------------------------------------------------------------
# Helpers.
#----------------------------------------------------------------------------

# OpenGL-related linker options depending on platform.
def _get_gl_opts():
    libs = {
        'posix': ['GL', 'EGL'],
        'nt':    ['gdi32', 'opengl32', 'user32', 'setgpu'],
    }
    return ['-l' + x for x in libs[os.name]]

# Load the cpp plugin.
def _get_plugin():
    fn = os.path.join(os.path.dirname(__file__), 'tf_all.cu')
    return plugin_loader.get_plugin(fn, extra_nvcc_options=_get_gl_opts() + ['-DNVDR_TENSORFLOW'])

# Convert parameter to a numpy array if possible.
def _get_constant(x, dtype):
    try:
        return np.asarray(x, dtype=dtype)
    except (TypeError, ValueError):
        return None

# Tests for a construction-time constantness instead of tf.constant node because
# the latter can be overridden in Session.run() feed_dict at evaluation time.
def _is_constant(x, dtype):
    if isinstance(x, np.ndarray):
        return np.can_cast(x.dtype, dtype, 'unsafe')
    else:
        return _get_constant(x, dtype) is not None

#----------------------------------------------------------------------------
# Rasterize.
#----------------------------------------------------------------------------

def rasterize(pos, tri, resolution, ranges=None, tri_const=False, output_db=True, grad_db=True):
    assert tri_const is True or tri_const is False
    assert output_db is True or output_db is False

    # Known constant resolution?
    resolution_c = _get_constant(resolution, np.int32)

    # Known constant triangles?
    tri_const = tri_const or _is_constant(tri, np.int32)

    # Convert all inputs to tensors / base types.
    tri_const = 1 if tri_const else 0
    tri = tf.convert_to_tensor(tri, dtype=tf.int32)
    pos = tf.convert_to_tensor(pos, dtype=tf.float32)
    resolution = tf.convert_to_tensor(resolution, dtype=tf.int32)
    if ranges is None:
        ranges = tf.convert_to_tensor(np.zeros(shape=[0, 2], dtype=np.int32)) # Empty tensor.
    else:
        ranges = tf.convert_to_tensor(ranges, dtype=tf.int32) # Convert input to tensor.

    # Infer as much about the output shape as possible.
    out_shape = [None, None, None, 4]
    if pos.shape.rank == 3: # Instanced mode.
        out_shape[0] = pos.shape[0].value
    elif pos.shape.rank == 2: # Range mode.
        if ranges.shape.rank not in [None, 0]:
            out_shape[0] = ranges.shape[0].value
    if resolution_c is not None:
        assert resolution_c.shape == (2,)
        out_shape[1], out_shape[2] = resolution_c

    # Output pixel differentials.
    @tf.custom_gradient
    def func_db(pos):
        out, out_db = _get_plugin().rasterize_fwd(pos, tri, resolution, ranges, 1, tri_const)
        out.set_shape(out_shape)
        out_db.set_shape(out_shape)
        def grad(dy, ddb):
            if grad_db:
                return _get_plugin().rasterize_grad_db(pos, tri, out, dy, ddb)
            else:
                return _get_plugin().rasterize_grad(pos, tri, out, dy)
        return (out, out_db), grad

    # Do not output pixel differentials.
    @tf.custom_gradient
    def func(pos):
        out, out_db = _get_plugin().rasterize_fwd(pos, tri, resolution, ranges, 0, tri_const)
        out.set_shape(out_shape)
        out_db.set_shape(out_shape[:-1] + [0]) # Zero channels in out_db.
        def grad(dy, _):
            return _get_plugin().rasterize_grad(pos, tri, out, dy)
        return (out, out_db), grad

    # Choose stub.
    if output_db:
        return func_db(pos)
    else:
        return func(pos)

#----------------------------------------------------------------------------
# Interpolate.
#----------------------------------------------------------------------------

def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    # Sanitize the list of pixel differential attributes.
    if diff_attrs is None:
        diff_attrs = []
    elif diff_attrs != 'all':
        diff_attrs = _get_constant(diff_attrs, np.int32)
        assert (diff_attrs is not None) and len(diff_attrs.shape) == 1
        diff_attrs = diff_attrs.tolist()

    # Convert all inputs to tensors.
    attr = tf.convert_to_tensor(attr, dtype=tf.float32)
    rast = tf.convert_to_tensor(rast, dtype=tf.float32)
    tri = tf.convert_to_tensor(tri, dtype=tf.int32)
    if diff_attrs:
        rast_db = tf.convert_to_tensor(rast_db, dtype=tf.float32)

    # Infer output shape.
    out_shape = [None, None, None, None]
    if rast.shape.rank is not None:
        out_shape = [rast.shape[0].value, rast.shape[1].value, rast.shape[2].value, None]
    if attr.shape.rank in [2, 3]:
        out_shape[3] = attr.shape[-1].value

    # Output pixel differentials for at least some attributes.
    @tf.custom_gradient
    def func_da(attr, rast, rast_db):
        diff_attrs_all = int(diff_attrs == 'all')
        diff_attrs_list = [] if diff_attrs_all else diff_attrs
        out, out_da = _get_plugin().interpolate_fwd_da(attr, rast, tri, rast_db, diff_attrs_all, diff_attrs_list)

        # Infer number of channels in out_da.
        if not diff_attrs_all:
            da_channels = 2 * len(diff_attrs)
        if (attr.shape.rank in [2, 3]) and (attr.shape[-1].value is not None):
            da_channels = 2 * attr.shape[-1].value
        else:
            da_channels = None

        # Set output shapes.
        out.set_shape(out_shape)
        out_da.set_shape([out_shape[0], out_shape[1], out_shape[2], da_channels])

        def grad(dy, dda):
            return _get_plugin().interpolate_grad_da(attr, rast, tri, dy, rast_db, dda, diff_attrs_all, diff_attrs_list)
        return (out, out_da), grad

    # No pixel differentials for any attribute.
    @tf.custom_gradient
    def func(attr, rast):
        out, out_da = _get_plugin().interpolate_fwd(attr, rast, tri)
        out.set_shape(out_shape)
        out_da.set_shape(out_shape[:-1] + [0]) # Zero channels in out_da.
        def grad(dy, _):
            return _get_plugin().interpolate_grad(attr, rast, tri, dy)
        return (out, out_da), grad

    # Choose stub.
    if diff_attrs:
        return func_da(attr, rast, rast_db)
    else:
        return func(attr, rast)

#----------------------------------------------------------------------------
# Texture.
#----------------------------------------------------------------------------

def texture(tex, uv, uv_da=None, filter_mode='auto', boundary_mode='wrap', tex_const=False, max_mip_level=None):
    assert tex_const is True or tex_const is False

    # Default filter mode.
    if filter_mode == 'auto':
        filter_mode = 'linear-mipmap-linear' if (uv_da is not None) else 'linear'

    # Known constant texture?
    tex_const = tex_const or _is_constant(tex, np.float32)

    # Sanitize inputs.
    tex_const = 1 if tex_const else 0
    if max_mip_level is None:
        max_mip_level = -1
    else:
        max_mip_level = int(max_mip_level)
        assert max_mip_level >= 0

    # Convert inputs to tensors.
    tex = tf.convert_to_tensor(tex, dtype=tf.float32)
    uv = tf.convert_to_tensor(uv, dtype=tf.float32)
    if 'mipmap' in filter_mode:
        uv_da = tf.convert_to_tensor(uv_da, dtype=tf.float32)

    # Infer output shape.
    out_shape = [None, None, None, None]
    if uv.shape.rank is not None:
        assert uv.shape.rank == 4
        out_shape = [uv.shape[0].value, uv.shape[1].value, uv.shape[2].value, None]
    if tex.shape.rank is not None:
        assert tex.shape.rank == (5 if boundary_mode == 'cube' else 4)
        out_shape[-1] = tex.shape[-1].value

    # If mipping disabled via max level=0, we may as well use simpler filtering internally.
    if max_mip_level == 0 and filter_mode in ['linear-mipmap-nearest', 'linear-mipmap-linear']:
        filter_mode = 'linear'

    # Convert filter mode to internal enumeration.
    filter_mode_dict = {'nearest': 0, 'linear': 1, 'linear-mipmap-nearest': 2, 'linear-mipmap-linear': 3}
    filter_mode_enum = filter_mode_dict[filter_mode]

    # Convert boundary mode to internal enumeration.
    boundary_mode_dict = {'cube': 0, 'wrap': 1, 'clamp': 2, 'zero': 3}
    boundary_mode_enum = boundary_mode_dict[boundary_mode]

    # Linear-mipmap-linear: Mipmaps enabled, all gradients active.
    @tf.custom_gradient
    def func_linear_mipmap_linear(tex, uv, uv_da):
        out, mip = _get_plugin().texture_fwd_mip(tex, uv, uv_da, filter_mode_enum, boundary_mode_enum, tex_const, max_mip_level)
        out.set_shape(out_shape)
        def grad(dy):
            return _get_plugin().texture_grad_linear_mipmap_linear(tex, uv, dy, uv_da, mip, filter_mode_enum, boundary_mode_enum, max_mip_level)
        return out, grad

    # Linear-mipmap-nearest: Mipmaps enabled, no gradients to uv_da.
    @tf.custom_gradient
    def func_linear_mipmap_nearest(tex, uv):
        out, mip = _get_plugin().texture_fwd_mip(tex, uv, uv_da, filter_mode_enum, boundary_mode_enum, tex_const, max_mip_level)
        out.set_shape(out_shape)
        def grad(dy):
            return _get_plugin().texture_grad_linear_mipmap_nearest(tex, uv, dy, uv_da, mip, filter_mode_enum, boundary_mode_enum, max_mip_level)
        return out, grad

    # Linear: Mipmaps disabled, no uv_da, no gradients to uv_da.
    @tf.custom_gradient
    def func_linear(tex, uv):
        out = _get_plugin().texture_fwd(tex, uv, filter_mode_enum, boundary_mode_enum)
        out.set_shape(out_shape)
        def grad(dy):
            return _get_plugin().texture_grad_linear(tex, uv, dy, filter_mode_enum, boundary_mode_enum)
        return out, grad

    # Nearest: Mipmaps disabled, no uv_da, no gradients to uv_da or uv.
    @tf.custom_gradient
    def func_nearest(tex):
        out = _get_plugin().texture_fwd(tex, uv, filter_mode_enum, boundary_mode_enum)
        out.set_shape(out_shape)
        def grad(dy):
            return _get_plugin().texture_grad_nearest(tex, uv, dy, filter_mode_enum, boundary_mode_enum)
        return out, grad

    # Choose stub.
    if filter_mode == 'linear-mipmap-linear':
        return func_linear_mipmap_linear(tex, uv, uv_da)
    elif filter_mode == 'linear-mipmap-nearest':
        return func_linear_mipmap_nearest(tex, uv)
    elif filter_mode == 'linear':
        return func_linear(tex, uv)
    elif filter_mode == 'nearest':
        return func_nearest(tex)

#----------------------------------------------------------------------------
# Antialias.
#----------------------------------------------------------------------------

def antialias(color, rast, pos, tri, tri_const=False, pos_gradient_boost=1.0):
    assert tri_const is True or tri_const is False

    # Known constant triangles?
    tri_const = tri_const or _is_constant(tri, np.int32)

    # Convert inputs to tensors.
    color = tf.convert_to_tensor(color, dtype=tf.float32)
    rast = tf.convert_to_tensor(rast, dtype=tf.float32)
    pos = tf.convert_to_tensor(pos, dtype=tf.float32)
    tri = tf.convert_to_tensor(tri, dtype=tf.int32)

    # Sanitize inputs.
    tri_const = 1 if tri_const else 0

    @tf.custom_gradient
    def func(color, pos):
        color_out, work_buffer = _get_plugin().antialias_fwd(color, rast, pos, tri, tri_const)
        color_out.set_shape(color.shape)
        def grad(dy):
            grad_color, grad_pos = _get_plugin().antialias_grad(color, rast, pos, tri, dy, work_buffer)
            if pos_gradient_boost != 1.0:
                grad_pos = grad_pos * pos_gradient_boost
            return grad_color, grad_pos
        return color_out, grad

    return func(color, pos)

#----------------------------------------------------------------------------
