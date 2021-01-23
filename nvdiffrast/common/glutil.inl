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
        LOG(FATAL) << "setGLContext() called with null gltcx";
    if (!wglMakeCurrent(glctx.hdc, glctx.hglrc))
        LOG(FATAL) << "wglMakeCurrent() failed when setting GL context";

    if (glctx.glewInitialized)
        return;
    GLenum result = glewInit();
    if (result != GLEW_OK)
        LOG(FATAL) << "glewInit() failed, return value = " << result;
    glctx.glewInitialized = 1;
}

static void releaseGLContext(void)
{
    if (!wglMakeCurrent(NULL, NULL))
        LOG(FATAL) << "wglMakeCurrent() failed when releasing GL context";
}

extern "C" int set_gpu(const char*);

static GLContext createGLContext(int cudaDeviceIdx)
{
    if (cudaDeviceIdx >= 0)
    {
        char pciBusId[256] = "";
        LOG(INFO) << "Creating GL context for Cuda device " << cudaDeviceIdx;
        if (cudaDeviceGetPCIBusId(pciBusId, 255, cudaDeviceIdx) != CUDA_SUCCESS)
        {
            LOG(INFO) << "PCI bus id query failed";
        }
        else
        {
            int res = set_gpu(pciBusId);
            LOG(INFO) << "Selecting device with PCI bus id " << pciBusId << " - " << (res ? "failed, expect crash or major slowdown" : "success");
        }
    }

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
        LOG(FATAL) << "destroyGLContext() called with null gltcx";

    // If this is the current context, release it.
    if (wglGetCurrentContext() == glctx.hglrc)
        releaseGLContext();

    HWND hwnd = WindowFromDC(glctx.hdc);
    if (!hwnd)
        LOG(FATAL) << "WindowFromDC() failed";
    if (!ReleaseDC(hwnd, glctx.hdc))
        LOG(FATAL) << "ReleaseDC() failed";
    if (!wglDeleteContext(glctx.hglrc))
        LOG(FATAL) << "wglDeleteContext() failed";
    if (!DestroyWindow(hwnd))
        LOG(FATAL) << "DestroyWindow() failed";

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
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>

//------------------------------------------------------------------------

struct GLContext
{
    EGLDisplay  display;
    EGLContext  context;
    int         glewInitialized;
};

//------------------------------------------------------------------------

static void setGLContext(GLContext& glctx)
{
    if (!glctx.context)
        LOG(FATAL) << "setGLContext() called with null gltcx";

    if (!eglMakeCurrent(glctx.display, EGL_NO_SURFACE, EGL_NO_SURFACE, glctx.context))
        LOG(ERROR) << "eglMakeCurrent() failed when setting GL context";

    if (glctx.glewInitialized)
        return;

    GLenum result = glewInit();
    if (result != GLEW_OK)
        LOG(FATAL) << "glewInit() failed, return value = " << result;
    glctx.glewInitialized = 1;
}

static void releaseGLContext(void)
{
    EGLDisplay display = eglGetCurrentDisplay();
    if (display == EGL_NO_DISPLAY)
        LOG(WARNING) << "releaseGLContext() called with no active display";
    if (!eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT))
        LOG(FATAL) << "eglMakeCurrent() failed when releasing GL context";
}

static EGLDisplay getCudaDisplay(int cudaDeviceIdx)
{
    typedef EGLBoolean (*eglQueryDevicesEXT_t)(EGLint, EGLDeviceEXT, EGLint*);
    typedef EGLBoolean (*eglQueryDeviceAttribEXT_t)(EGLDeviceEXT, EGLint, EGLAttrib*);
    typedef EGLDisplay (*eglGetPlatformDisplayEXT_t)(EGLenum, void*, const EGLint*);

    eglQueryDevicesEXT_t eglQueryDevicesEXT = (eglQueryDevicesEXT_t)eglGetProcAddress("eglQueryDevicesEXT");
    if (!eglQueryDevicesEXT)
    {
        LOG(INFO) << "eglGetProcAddress(\"eglQueryDevicesEXT\") failed";
        return 0;
    }

    eglQueryDeviceAttribEXT_t eglQueryDeviceAttribEXT = (eglQueryDeviceAttribEXT_t)eglGetProcAddress("eglQueryDeviceAttribEXT");
    if (!eglQueryDeviceAttribEXT)
    {
        LOG(INFO) << "eglGetProcAddress(\"eglQueryDeviceAttribEXT\") failed";
        return 0;
    }

    eglGetPlatformDisplayEXT_t eglGetPlatformDisplayEXT = (eglGetPlatformDisplayEXT_t)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!eglGetPlatformDisplayEXT)
    {
        LOG(INFO) << "eglGetProcAddress(\"eglGetPlatformDisplayEXT\") failed";
        return 0;
    }

    int num_devices = 0;
    eglQueryDevicesEXT(0, 0, &num_devices);
    if (!num_devices)
        return 0;
    
    EGLDisplay display = 0;
    EGLDeviceEXT* devices = (EGLDeviceEXT*)malloc(num_devices * sizeof(void*));
    eglQueryDevicesEXT(num_devices, devices, &num_devices);
    for (int i=0; i < num_devices; i++)
    {
        EGLDeviceEXT device = devices[i]; 
        intptr_t value = -1;
        if (eglQueryDeviceAttribEXT(device, EGL_CUDA_DEVICE_NV, &value) && value == cudaDeviceIdx)
        {
            display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, device, 0);
            break;
        }
    }

    free(devices);
    return display;
}

static GLContext createGLContext(int cudaDeviceIdx)
{
    EGLDisplay display = 0;

    if (cudaDeviceIdx >= 0)
    {
        char pciBusId[256] = "";
        LOG(INFO) << "Creating GL context for Cuda device " << cudaDeviceIdx;
        display = getCudaDisplay(cudaDeviceIdx);
        if (!display)
            LOG(INFO) << "Failed, falling back to default display";
    }

    if (!display)  
    {
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (display == EGL_NO_DISPLAY)
            LOG(FATAL) << "eglGetDisplay() failed";
    }

    EGLint major;
    EGLint minor;
    if (!eglInitialize(display, &major, &minor))
        LOG(FATAL) << "eglInitialize() failed";

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
        LOG(FATAL) << "eglChooseConfig() failed";

    // Create GL context.

    if (!eglBindAPI(EGL_OPENGL_API))
        LOG(FATAL) << "eglBindAPI() failed";

    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, NULL);
    if (context == EGL_NO_CONTEXT)
        LOG(FATAL) << "eglCreateContext() failed";

    // Done.

    LOG(INFO) << "EGL " << (int)minor << "." << (int)major << " OpenGL context created (disp: 0x"
              << std::hex << std::setfill('0')
              << std::setw(16) << (uintptr_t)display
              << ", ctx: 0x" << std::setw(16) << (uintptr_t)context << ")";

    GLContext glctx = {display, context, 0};
    return glctx;
}

static void destroyGLContext(GLContext& glctx)
{
    if (!glctx.context)
        LOG(FATAL) << "destroyGLContext() called with null gltcx";

    // If this is the current context, release it.
    if (eglGetCurrentContext() == glctx.context)
        releaseGLContext();

    if (!eglDestroyContext(glctx.display, glctx.context))
        LOG(ERROR) << "eglDestroyContext() failed";

    LOG(INFO) << "EGL OpenGL context destroyed (disp: 0x"
              << std::hex << std::setfill('0')
              << std::setw(16) << (uintptr_t)glctx.display
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
