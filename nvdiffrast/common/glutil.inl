// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once
#include "framework.h"
#include <iostream>
#include <iomanip>

//------------------------------------------------------------------------
// Windows.
//------------------------------------------------------------------------

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#define GLEW_STATIC
#include "../lib/glew.h"
#include <GL/gl.h>
#include <cuda_gl_interop.h>

//------------------------------------------------------------------------

struct GLContext
{
    HDC     hdc;
    HGLRC   hglrc;
    int     glewInitialized;
};

//------------------------------------------------------------------------

static void setGLContext(GLContext& glctx)
{
    if (!glctx.hglrc)
        LOG(ERROR) << "setGLContext() called with null gltcx";
    if (!wglMakeCurrent(glctx.hdc, glctx.hglrc))
        LOG(ERROR) << "wglMakeCurrent() failed when setting GL context";

    if (glctx.glewInitialized)
        return;
    GLenum result = glewInit();
    if (result != GLEW_OK)
        LOG(ERROR) << "glewInit() failed, return value = " << result;
    glctx.glewInitialized = 1;
}

static void releaseGLContext(void)
{
    if (!wglMakeCurrent(NULL, NULL))
        LOG(ERROR) << "wglMakeCurrent() failed when releasing GL context";
}

static GLContext createGLContext(void)
{
    HINSTANCE hInstance = GetModuleHandle(NULL);
    WNDCLASS wc = {};
    wc.style         = CS_OWNDC;
    wc.lpfnWndProc   = DefWindowProc;
    wc.hInstance     = hInstance;
    wc.lpszClassName = "__DummyGLClassCPP";
    int res = RegisterClass(&wc);

    HWND hwnd = CreateWindow(
        "__DummyGLClassCPP",        // lpClassName
        "__DummyGLWindowCPP",       // lpWindowName
        WS_OVERLAPPEDWINDOW,        // dwStyle
        CW_USEDEFAULT,              // x
        CW_USEDEFAULT,              // y
        0, 0,                       // nWidth, nHeight
        NULL, NULL,                 // hWndParent, hMenu
        hInstance,                  // hInstance
        NULL                        // lpParam
    );

    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.dwFlags      = PFD_SUPPORT_OPENGL;
    pfd.iPixelType   = PFD_TYPE_RGBA;
    pfd.iLayerType   = PFD_MAIN_PLANE;
    pfd.cColorBits   = 32;
    pfd.cDepthBits   = 24;
    pfd.cStencilBits = 8;

    HDC hdc = GetDC(hwnd);
    int pixelformat = ChoosePixelFormat(hdc, &pfd);
    SetPixelFormat(hdc, pixelformat, &pfd);

    HGLRC hglrc = wglCreateContext(hdc);
    LOG(INFO) << std::hex << std::setfill('0')
              << "WGL OpenGL context created (hdc: 0x" << std::setw(8) << (uint32_t)(uintptr_t)hdc
              << ", hglrc: 0x" << std::setw(8) << (uint32_t)(uintptr_t)hglrc << ")";

    GLContext glctx = {hdc, hglrc, 0};
    return glctx;
}

static void destroyGLContext(GLContext& glctx)
{
    if (!glctx.hglrc)
        LOG(ERROR) << "destroyGLContext() called with null gltcx";

    // If this is the current context, release it.
    if (wglGetCurrentContext() == glctx.hglrc)
        releaseGLContext();

    HWND hwnd = WindowFromDC(glctx.hdc);
    if (!hwnd)
        LOG(ERROR) << "WindowFromDC() failed";
    if (!ReleaseDC(hwnd, glctx.hdc))
        LOG(ERROR) << "ReleaseDC() failed";
    if (!wglDeleteContext(glctx.hglrc))
        LOG(ERROR) << "wglDeleteContext() failed";
    if (!DestroyWindow(hwnd))
        LOG(ERROR) << "DestroyWindow() failed";

    LOG(INFO) << std::hex << std::setfill('0')
              << "WGL OpenGL context destroyed (hdc: 0x" << std::setw(8) << (uint32_t)(uintptr_t)glctx.hdc
              << ", hglrc: 0x" << std::setw(8) << (uint32_t)(uintptr_t)glctx.hglrc << ")";

    memset(&glctx, 0, sizeof(GLContext));
}

#endif // _WIN32

//------------------------------------------------------------------------
// Linux.
//------------------------------------------------------------------------

#ifdef __linux__
#define GLEW_NO_GLU
#define EGL_NO_X11 // X11/Xlib.h has "#define Status int" which breaks Tensorflow. Avoid it.
#define MESA_EGL_NO_X11_HEADERS
#if 1
#   include "../lib/glew.h"    // Use local glew.h
#else
#   include <GL/glew.h> // Use system-supplied glew.h
#endif
#include <EGL/egl.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>

//------------------------------------------------------------------------

struct GLContext
{
    EGLDisplay  display;
    EGLSurface  surface;
    EGLContext  context;
    int         glewInitialized;
};

//------------------------------------------------------------------------

static void setGLContext(GLContext& glctx)
{
    if (!glctx.context)
        LOG(ERROR) << "setGLContext() called with null gltcx";

    if (!eglMakeCurrent(glctx.display, glctx.surface, glctx.surface, glctx.context))
        LOG(ERROR) << "eglMakeCurrent() failed when setting GL context";

    if (glctx.glewInitialized)
        return;

    GLenum result = glewInit();
    if (result != GLEW_OK)
        LOG(ERROR) << "glewInit() failed, return value = " << result;
    glctx.glewInitialized = 1;
}

static void releaseGLContext(void)
{
    EGLDisplay display = eglGetCurrentDisplay();
    if (display == EGL_NO_DISPLAY)
        LOG(WARNING) << "releaseGLContext() called with no active display";
    if (!eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT))
        LOG(ERROR) << "eglMakeCurrent() failed when releasing GL context";
}

static GLContext createGLContext(void)
{
    // Initialize.

    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY)
        LOG(ERROR) << "eglGetDisplay() failed";

    EGLint major;
    EGLint minor;
    if (!eglInitialize(display, &major, &minor))
        LOG(ERROR) << "eglInitialize() failed";

    // Choose configuration.

    const EGLint context_attribs[] = {
        EGL_RED_SIZE,           8,
        EGL_GREEN_SIZE,         8,
        EGL_BLUE_SIZE,          8,
        EGL_ALPHA_SIZE,         8,
        EGL_DEPTH_SIZE,         24,
        EGL_STENCIL_SIZE,       8,
        EGL_RENDERABLE_TYPE,    EGL_OPENGL_BIT,
        EGL_SURFACE_TYPE,       EGL_PBUFFER_BIT,
        EGL_NONE
    };

    EGLConfig config;
    EGLint num_config;
    if (!eglChooseConfig(display, context_attribs, &config, 1, &num_config))
        LOG(ERROR) << "eglChooseConfig() failed";

    // Create dummy pbuffer surface.

    const EGLint surface_attribs[] = {
        EGL_WIDTH,      1,
        EGL_HEIGHT,     1,
        EGL_NONE
    };

    EGLSurface surface = eglCreatePbufferSurface(display, config, surface_attribs);
    if (surface == EGL_NO_SURFACE)
        LOG(ERROR) << "eglCreatePbufferSurface() failed";

    // Create GL context.

    if (!eglBindAPI(EGL_OPENGL_API))
        LOG(ERROR) << "eglBindAPI() failed";

    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, NULL);
    if (context == EGL_NO_CONTEXT)
        LOG(ERROR) << "eglCreateContext() failed";

    // Done.

    LOG(INFO) << "EGL " << (int)minor << "." << (int)major << " OpenGL context created (disp: 0x"
              << std::hex << std::setfill('0')
              << std::setw(16) << (uintptr_t)display
              << ", surf: 0x" << std::setw(16) << (uintptr_t)surface
              << ", ctx: 0x" << std::setw(16) << (uintptr_t)context << ")";

    GLContext glctx = {display, surface, context, 0};
    return glctx;
}

static void destroyGLContext(GLContext& glctx)
{
    if (!glctx.context)
        LOG(ERROR) << "destroyGLContext() called with null gltcx";

    // If this is the current context, release it.
    if (eglGetCurrentContext() == glctx.context)
        releaseGLContext();

    if (!eglDestroyContext(glctx.display, glctx.context))
        LOG(ERROR) << "eglDestroyContext() failed";
    if (!eglDestroySurface(glctx.display, glctx.surface))
        LOG(ERROR) << "eglDestroySurface() failed";

    LOG(INFO) << "EGL OpenGL context destroyed (disp: 0x"
              << std::hex << std::setfill('0')
              << std::setw(16) << (uintptr_t)glctx.display
              << ", surf: 0x" << std::setw(16) << (uintptr_t)glctx.surface
              << ", ctx: 0x" << std::setw(16) << (uintptr_t)glctx.context << ")";

    memset(&glctx, 0, sizeof(GLContext));
}

#endif // __linux__

//------------------------------------------------------------------------
// Common.
//------------------------------------------------------------------------

static const char* getGLErrorString(GLenum err)
{
    switch(err)
    {
        case GL_NO_ERROR:                       return "GL_NO_ERROR";
        case GL_INVALID_ENUM:                   return "GL_INVALID_ENUM";
        case GL_INVALID_VALUE:                  return "GL_INVALID_VALUE";
        case GL_INVALID_OPERATION:              return "GL_INVALID_OPERATION";
        case GL_STACK_OVERFLOW:                 return "GL_STACK_OVERFLOW";
        case GL_STACK_UNDERFLOW:                return "GL_STACK_UNDERFLOW";
        case GL_OUT_OF_MEMORY:                  return "GL_OUT_OF_MEMORY";
        case GL_INVALID_FRAMEBUFFER_OPERATION:  return "GL_INVALID_FRAMEBUFFER_OPERATION";
        case GL_TABLE_TOO_LARGE:                return "GL_TABLE_TOO_LARGE";
        case GL_CONTEXT_LOST:                   return "GL_CONTEXT_LOST";
    }
    return "Unknown error";
}

//------------------------------------------------------------------------
