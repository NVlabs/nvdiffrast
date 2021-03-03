// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

//------------------------------------------------------------------------
// Windows-specific headers and types.
//------------------------------------------------------------------------

#ifdef _WIN32
#define NOMINMAX
#include <windows.h> // Required by gl.h in Windows.
#define GLAPIENTRY APIENTRY

struct GLContext
{
    HDC     hdc;
    HGLRC   hglrc;
    int     extInitialized;
};

#endif // _WIN32

//------------------------------------------------------------------------
// Linux-specific headers and types.
//------------------------------------------------------------------------

#ifdef __linux__
#define EGL_NO_X11 // X11/Xlib.h has "#define Status int" which breaks Tensorflow. Avoid it.
#define MESA_EGL_NO_X11_HEADERS
#include <EGL/egl.h>
#include <EGL/eglext.h>
#define GLAPIENTRY

struct GLContext
{
    EGLDisplay  display;
    EGLContext  context;
    int         extInitialized;
};

#endif // __linux__

//------------------------------------------------------------------------
// OpenGL, CUDA interop, GL extensions.
//------------------------------------------------------------------------
#define GL_GLEXT_LEGACY
#include <GL/gl.h>
#include <cuda_gl_interop.h>

// Constants.
#ifndef GL_VERSION_1_2
#define GL_CLAMP_TO_EDGE                 0x812F
#define GL_TEXTURE_3D                    0x806F
#endif
#ifndef GL_VERSION_1_5
#define GL_ARRAY_BUFFER                  0x8892
#define GL_DYNAMIC_DRAW                  0x88E8
#define GL_ELEMENT_ARRAY_BUFFER          0x8893
#endif
#ifndef GL_VERSION_2_0
#define GL_FRAGMENT_SHADER               0x8B30
#define GL_INFO_LOG_LENGTH               0x8B84
#define GL_LINK_STATUS                   0x8B82
#define GL_VERTEX_SHADER                 0x8B31
#endif
#ifndef GL_VERSION_3_0
#define GL_MAJOR_VERSION                 0x821B
#define GL_MINOR_VERSION                 0x821C
#define GL_RGBA32F                       0x8814
#define GL_TEXTURE_2D_ARRAY              0x8C1A
#endif
#ifndef GL_VERSION_3_2
#define GL_GEOMETRY_SHADER               0x8DD9
#endif
#ifndef GL_ARB_framebuffer_object
#define GL_COLOR_ATTACHMENT0             0x8CE0
#define GL_COLOR_ATTACHMENT1             0x8CE1
#define GL_DEPTH_STENCIL                 0x84F9
#define GL_DEPTH_STENCIL_ATTACHMENT      0x821A
#define GL_DEPTH24_STENCIL8              0x88F0
#define GL_FRAMEBUFFER                   0x8D40
#define GL_INVALID_FRAMEBUFFER_OPERATION 0x0506
#define GL_UNSIGNED_INT_24_8             0x84FA
#endif
#ifndef GL_ARB_imaging
#define GL_TABLE_TOO_LARGE               0x8031
#endif
#ifndef GL_KHR_robustness
#define GL_CONTEXT_LOST                  0x0507
#endif

// Declare function pointers to OpenGL extension functions.
#define GLUTIL_EXT(return_type, name, ...) extern return_type (GLAPIENTRY* name)(__VA_ARGS__);
#include "glutil_extlist.h"
#undef GLUTIL_EXT

//------------------------------------------------------------------------
// Common functions.
//------------------------------------------------------------------------

void        setGLContext            (GLContext& glctx);
void        releaseGLContext        (void);
GLContext   createGLContext         (int cudaDeviceIdx);
void        destroyGLContext        (GLContext& glctx);
const char* getGLErrorString        (GLenum err);

//------------------------------------------------------------------------
