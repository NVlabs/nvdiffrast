# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import numpy as np
import tensorflow as tf

# Silence deprecation warnings from TensorFlow 1.13 onwards
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from typing import Any, List

#----------------------------------------------------------------------------
# Projection and transformation matrix helpers.
#----------------------------------------------------------------------------

def projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n/x,    0,            0,              0], 
                     [  0, n/-x,            0,              0], 
                     [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                     [  0,    0,           -1,              0]]).astype(np.float32)
                    
def translate(x, y, z):
    return np.array([[1, 0, 0, x], 
                     [0, 1, 0, y], 
                     [0, 0, 1, z], 
                     [0, 0, 0, 1]]).astype(np.float32)

def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1,  0, 0, 0], 
                     [0,  c, s, 0], 
                     [0, -s, c, 0], 
                     [0,  0, 0, 1]]).astype(np.float32)

def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[ c, 0, s, 0], 
                     [ 0, 1, 0, 0], 
                     [-s, 0, c, 0], 
                     [ 0, 0, 0, 1]]).astype(np.float32)

def random_rotation_translation(t):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return m

#----------------------------------------------------------------------------
# Bilinear downsample by 2x.
#----------------------------------------------------------------------------

def bilinear_downsample(x):
    w = tf.constant([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype=tf.float32) / 64.0
    w = w[..., tf.newaxis, tf.newaxis] * tf.eye(x.shape[-1].value, batch_shape=[1, 1])
    x = tf.nn.conv2d(x, w, strides=2, padding='SAME')
    return x

#----------------------------------------------------------------------------
# Image display function using OpenGL.
#----------------------------------------------------------------------------

_glfw_window = None
def display_image(image, zoom=None, size=None, title=None): # HWC
    # Import OpenGL and glfw.
    import OpenGL.GL as gl
    import glfw

    # Zoom image if requested.
    image = np.asarray(image)
    if size is not None:
        assert zoom is None
        zoom = max(1, size // image.shape[0])
    if zoom is not None:
        image = image.repeat(zoom, axis=0).repeat(zoom, axis=1)
    height, width, channels = image.shape

    # Initialize window.
    if title is None:
        title = 'Debug window'
    global _glfw_window
    if _glfw_window is None:
        glfw.init()
        _glfw_window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(_glfw_window)
        glfw.show_window(_glfw_window)
        glfw.swap_interval(0)
    else:
        glfw.make_context_current(_glfw_window)
        glfw.set_window_title(_glfw_window, title)
        glfw.set_window_size(_glfw_window, width, height)

    # Update window.
    glfw.poll_events()
    gl.glClearColor(0, 0, 0, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glWindowPos2f(0, 0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl_format = {3: gl.GL_RGB, 2: gl.GL_RG, 1: gl.GL_LUMINANCE}[channels]
    gl_dtype = {'uint8': gl.GL_UNSIGNED_BYTE, 'float32': gl.GL_FLOAT}[image.dtype.name]
    gl.glDrawPixels(width, height, gl_format, gl_dtype, image[::-1])
    glfw.swap_buffers(_glfw_window)
    if glfw.window_should_close(_glfw_window):
        return False
    return True

#----------------------------------------------------------------------------
# Image save helper.
#----------------------------------------------------------------------------

def save_image(fn, x):
    import imageio
    x = np.rint(x * 255.0)
    x = np.clip(x, 0, 255).astype(np.uint8)
    imageio.imsave(fn, x)

#----------------------------------------------------------------------------

# TensorFlow utilities

#----------------------------------------------------------------------------

def _sanitize_tf_config(config_dict: dict = None) -> dict:
    # Defaults.
    cfg = dict()
    cfg["rnd.np_random_seed"]               = None      # Random seed for NumPy. None = keep as is.
    cfg["rnd.tf_random_seed"]               = "auto"    # Random seed for TensorFlow. 'auto' = derive from NumPy random state. None = keep as is.
    cfg["env.TF_CPP_MIN_LOG_LEVEL"]         = "1"       # 0 = Print all available debug info from TensorFlow. 1 = Print warnings and errors, but disable debug info.
    cfg["env.HDF5_USE_FILE_LOCKING"]        = "FALSE"   # Disable HDF5 file locking to avoid concurrency issues with network shares.
    cfg["graph_options.place_pruned_graph"] = True      # False = Check that all ops are available on the designated device. True = Skip the check for ops that are not used.
    cfg["gpu_options.allow_growth"]         = True      # False = Allocate all GPU memory at the beginning. True = Allocate only as much GPU memory as needed.

    # Remove defaults for environment variables that are already set.
    for key in list(cfg):
        fields = key.split(".")
        if fields[0] == "env":
            assert len(fields) == 2
            if fields[1] in os.environ:
                del cfg[key]

    # User overrides.
    if config_dict is not None:
        cfg.update(config_dict)
    return cfg


def init_tf(config_dict: dict = None) -> None:
    """Initialize TensorFlow session using good default settings."""
    # Skip if already initialized.
    if tf.get_default_session() is not None:
        return

    # Setup config dict and random seeds.
    cfg = _sanitize_tf_config(config_dict)
    np_random_seed = cfg["rnd.np_random_seed"]
    if np_random_seed is not None:
        np.random.seed(np_random_seed)
    tf_random_seed = cfg["rnd.tf_random_seed"]
    if tf_random_seed == "auto":
        tf_random_seed = np.random.randint(1 << 31)
    if tf_random_seed is not None:
        tf.set_random_seed(tf_random_seed)

    # Setup environment variables.
    for key, value in cfg.items():
        fields = key.split(".")
        if fields[0] == "env":
            assert len(fields) == 2
            os.environ[fields[1]] = str(value)

    # Create default TensorFlow session.
    create_session(cfg, force_as_default=True)


def assert_tf_initialized():
    """Check that TensorFlow session has been initialized."""
    if tf.get_default_session() is None:
        raise RuntimeError("No default TensorFlow session found. Please call util.init_tf().")


def create_session(config_dict: dict = None, force_as_default: bool = False) -> tf.Session:
    """Create tf.Session based on config dict."""
    # Setup TensorFlow config proto.
    cfg = _sanitize_tf_config(config_dict)
    config_proto = tf.ConfigProto()
    for key, value in cfg.items():
        fields = key.split(".")
        if fields[0] not in ["rnd", "env"]:
            obj = config_proto
            for field in fields[:-1]:
                obj = getattr(obj, field)
            setattr(obj, fields[-1], value)

    # Create session.
    session = tf.Session(config=config_proto)
    if force_as_default:
        # pylint: disable=protected-access
        session._default_session = session.as_default()
        session._default_session.enforce_nesting = False
        session._default_session.__enter__()
    return session


def is_tf_expression(x: Any) -> bool:
    """Check whether the input is a valid Tensorflow expression, i.e., Tensorflow Tensor, Variable, or Operation."""
    return isinstance(x, (tf.Tensor, tf.Variable, tf.Operation))


def absolute_name_scope(scope: str) -> tf.name_scope:
    """Forcefully enter the specified name scope, ignoring any surrounding scopes."""
    return tf.name_scope(scope + "/")


def init_uninitialized_vars(target_vars: List[tf.Variable] = None) -> None:
    """Initialize all tf.Variables that have not already been initialized.

    Equivalent to the following, but more efficient and does not bloat the tf graph:
    tf.variables_initializer(tf.report_uninitialized_variables()).run()
    """
    assert_tf_initialized()
    if target_vars is None:
        target_vars = tf.global_variables()

    test_vars = []
    test_ops = []

    with tf.control_dependencies(None):  # ignore surrounding control_dependencies
        for var in target_vars:
            assert is_tf_expression(var)

            try:
                tf.get_default_graph().get_tensor_by_name(var.name.replace(":0", "/IsVariableInitialized:0"))
            except KeyError:
                # Op does not exist => variable may be uninitialized.
                test_vars.append(var)

                with absolute_name_scope(var.name.split(":")[0]):
                    test_ops.append(tf.is_variable_initialized(var))

    init_vars = [var for var, inited in zip(test_vars, run(test_ops)) if not inited]
    run([var.initializer for var in init_vars])

def run(*args, **kwargs) -> Any:
    """Run the specified ops in the default session."""
    assert_tf_initialized()
    return tf.get_default_session().run(*args, **kwargs)
