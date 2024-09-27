// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "rasterize_gl.h"
#include "glutil.h"
#include <vector>
#define STRINGIFY_SHADER_SOURCE(x) #x

//------------------------------------------------------------------------
// Helpers.

#define ROUND_UP(x, y) ((((x) + ((y) - 1)) / (y)) * (y))
static int ROUND_UP_BITS(uint32_t x, uint32_t y)
{
    // Round x up so that it has at most y bits of mantissa.
    if (x < (1u << y))
        return x;
    uint32_t m = 0;
    while (x & ~m)
        m = (m << 1) | 1u;
    m >>= y;
    if (!(x & m))
        return x;
    return (x | m) + 1u;
}

//------------------------------------------------------------------------
// Draw command struct used by rasterizer.

struct GLDrawCmd
{
    uint32_t    count;
    uint32_t    instanceCount;
    uint32_t    firstIndex;
    uint32_t    baseVertex;
    uint32_t    baseInstance;
};

//------------------------------------------------------------------------
// GL helpers.

static void compileGLShader(NVDR_CTX_ARGS, const RasterizeGLState& s, GLuint* pShader, GLenum shaderType, const char* src_buf)
{
    std::string src(src_buf);

    // Set preprocessor directives.
    int n = src.find('\n') + 1; // After first line containing #version directive.
    if (s.enableZModify)
        src.insert(n, "#define IF_ZMODIFY(x) x\n");
    else
        src.insert(n, "#define IF_ZMODIFY(x)\n");

    const char *cstr = src.c_str();
    *pShader = 0;
    NVDR_CHECK_GL_ERROR(*pShader = glCreateShader(shaderType));
    NVDR_CHECK_GL_ERROR(glShaderSource(*pShader, 1, &cstr, 0));
    NVDR_CHECK_GL_ERROR(glCompileShader(*pShader));
}

static void constructGLProgram(NVDR_CTX_ARGS, GLuint* pProgram, GLuint glVertexShader, GLuint glGeometryShader, GLuint glFragmentShader)
{
    *pProgram = 0;

    GLuint glProgram = 0;
    NVDR_CHECK_GL_ERROR(glProgram = glCreateProgram());
    NVDR_CHECK_GL_ERROR(glAttachShader(glProgram, glVertexShader));
    NVDR_CHECK_GL_ERROR(glAttachShader(glProgram, glGeometryShader));
    NVDR_CHECK_GL_ERROR(glAttachShader(glProgram, glFragmentShader));
    NVDR_CHECK_GL_ERROR(glLinkProgram(glProgram));

    GLint linkStatus = 0;
    NVDR_CHECK_GL_ERROR(glGetProgramiv(glProgram, GL_LINK_STATUS, &linkStatus));
    if (!linkStatus)
    {
        GLint infoLen = 0;
        NVDR_CHECK_GL_ERROR(glGetProgramiv(glProgram, GL_INFO_LOG_LENGTH, &infoLen));
        if (infoLen)
        {
            const char* hdr = "glLinkProgram() failed:\n";
            std::vector<char> info(strlen(hdr) + infoLen);
            strcpy(&info[0], hdr);
            NVDR_CHECK_GL_ERROR(glGetProgramInfoLog(glProgram, infoLen, &infoLen, &info[strlen(hdr)]));
            NVDR_CHECK(0, &info[0]);
        }
        NVDR_CHECK(0, "glLinkProgram() failed");
    }

    *pProgram = glProgram;
}

//------------------------------------------------------------------------
// Shared C++ functions.

void rasterizeInitGLContext(NVDR_CTX_ARGS, RasterizeGLState& s, int cudaDeviceIdx)
{
    // Create GL context and set it current.
    s.glctx = createGLContext(cudaDeviceIdx);
    setGLContext(s.glctx);

    // Version check.
    GLint vMajor = 0;
    GLint vMinor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &vMajor);
    glGetIntegerv(GL_MINOR_VERSION, &vMinor);
    glGetError(); // Clear possible GL_INVALID_ENUM error in version query.
    LOG(INFO) << "OpenGL version reported as " << vMajor << "." << vMinor;
    NVDR_CHECK((vMajor == 4 && vMinor >= 4) || vMajor > 4, "OpenGL 4.4 or later is required");

    // Enable depth modification workaround on A100 and later.
    int capMajor = 0;
    NVDR_CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&capMajor, cudaDevAttrComputeCapabilityMajor, cudaDeviceIdx));
    s.enableZModify = (capMajor >= 8);

    // Number of output buffers.
    int num_outputs = s.enableDB ? 2 : 1;

    // Set up vertex shader.
    compileGLShader(NVDR_CTX_PARAMS, s, &s.glVertexShader, GL_VERTEX_SHADER,
        "#version 330\n"
        "#extension GL_ARB_shader_draw_parameters : enable\n"
        STRINGIFY_SHADER_SOURCE(
            layout(location = 0) in vec4 in_pos;
            out int v_layer;
            out int v_offset;
            void main()
            {
                int layer = gl_DrawIDARB;
                gl_Position = in_pos;
                v_layer = layer;
                v_offset = gl_BaseInstanceARB; // Sneak in TriID offset here.
            }
        )
    );

    // Geometry and fragment shaders depend on if bary differential output is enabled or not.
    if (s.enableDB)
    {
        // Set up geometry shader. Calculation of per-pixel bary differentials is based on:
        //           u = (u/w) / (1/w)
        //   --> du/dX = d((u/w) / (1/w))/dX
        //   --> du/dX = [d(u/w)/dX - u*d(1/w)/dX] * w
        // and we know both d(u/w)/dX and d(1/w)/dX are constant over triangle.
        compileGLShader(NVDR_CTX_PARAMS, s, &s.glGeometryShader, GL_GEOMETRY_SHADER,
            "#version 430\n"
            STRINGIFY_SHADER_SOURCE(
                layout(triangles) in;
                layout(triangle_strip, max_vertices=3) out;
                layout(location = 0) uniform vec2 vp_scale;
                in int v_layer[];
                in int v_offset[];
                out vec4 var_uvzw;
                out vec4 var_db;
                void main()
                {
                    // Plane equations for bary differentials.
                    float w0 = gl_in[0].gl_Position.w;
                    float w1 = gl_in[1].gl_Position.w;
                    float w2 = gl_in[2].gl_Position.w;
                    vec2 p0 = gl_in[0].gl_Position.xy;
                    vec2 p1 = gl_in[1].gl_Position.xy;
                    vec2 p2 = gl_in[2].gl_Position.xy;
                    vec2 e0 = p0*w2 - p2*w0;
                    vec2 e1 = p1*w2 - p2*w1;
                    float a = e0.x*e1.y - e0.y*e1.x;

                    // Clamp area to an epsilon to avoid arbitrarily high bary differentials.
                    float eps = 1e-6f; // ~1 pixel in 1k x 1k image.
                    float ca = (abs(a) >= eps) ? a : (a < 0.f) ? -eps : eps; // Clamp with sign.
                    float ia = 1.f / ca; // Inverse area.

                    vec2 ascl = ia * vp_scale;
                    float dudx =  e1.y * ascl.x;
                    float dudy = -e1.x * ascl.y;
                    float dvdx = -e0.y * ascl.x;
                    float dvdy =  e0.x * ascl.y;

                    float duwdx = w2 * dudx;
                    float dvwdx = w2 * dvdx;
                    float duvdx = w0 * dudx + w1 * dvdx;
                    float duwdy = w2 * dudy;
                    float dvwdy = w2 * dvdy;
                    float duvdy = w0 * dudy + w1 * dvdy;

                    vec4 db0 = vec4(duvdx - dvwdx, duvdy - dvwdy, dvwdx, dvwdy);
                    vec4 db1 = vec4(duwdx, duwdy, duvdx - duwdx, duvdy - duwdy);
                    vec4 db2 = vec4(duwdx, duwdy, dvwdx, dvwdy);

                    int layer_id = v_layer[0];
                    int prim_id = gl_PrimitiveIDIn + v_offset[0];

                    gl_Layer = layer_id; gl_PrimitiveID = prim_id; gl_Position = vec4(gl_in[0].gl_Position.x, gl_in[0].gl_Position.y, gl_in[0].gl_Position.z, gl_in[0].gl_Position.w); var_uvzw = vec4(1.f, 0.f, gl_in[0].gl_Position.z, gl_in[0].gl_Position.w); var_db = db0; EmitVertex();
                    gl_Layer = layer_id; gl_PrimitiveID = prim_id; gl_Position = vec4(gl_in[1].gl_Position.x, gl_in[1].gl_Position.y, gl_in[1].gl_Position.z, gl_in[1].gl_Position.w); var_uvzw = vec4(0.f, 1.f, gl_in[1].gl_Position.z, gl_in[1].gl_Position.w); var_db = db1; EmitVertex();
                    gl_Layer = layer_id; gl_PrimitiveID = prim_id; gl_Position = vec4(gl_in[2].gl_Position.x, gl_in[2].gl_Position.y, gl_in[2].gl_Position.z, gl_in[2].gl_Position.w); var_uvzw = vec4(0.f, 0.f, gl_in[2].gl_Position.z, gl_in[2].gl_Position.w); var_db = db2; EmitVertex();
                }
            )
        );

        // Set up fragment shader.
        compileGLShader(NVDR_CTX_PARAMS, s, &s.glFragmentShader, GL_FRAGMENT_SHADER,
            "#version 430\n"
            STRINGIFY_SHADER_SOURCE(
                in vec4 var_uvzw;
                in vec4 var_db;
                layout(location = 0) out vec4 out_raster;
                layout(location = 1) out vec4 out_db;
                IF_ZMODIFY(
                    layout(location = 1) uniform float in_dummy;
                )
                void main()
                {
                    int id_int = gl_PrimitiveID + 1;
                    float id_float = (id_int <= 0x01000000) ? float(id_int) : intBitsToFloat(0x4a800000 + id_int);

                    out_raster = vec4(var_uvzw.x, var_uvzw.y, var_uvzw.z / var_uvzw.w, id_float);
                    out_db = var_db * var_uvzw.w;
                    IF_ZMODIFY(gl_FragDepth = gl_FragCoord.z + in_dummy;)
                }
            )
        );

        // Set up fragment shader for depth peeling.
        compileGLShader(NVDR_CTX_PARAMS, s, &s.glFragmentShaderDP, GL_FRAGMENT_SHADER,
            "#version 430\n"
            STRINGIFY_SHADER_SOURCE(
                in vec4 var_uvzw;
                in vec4 var_db;
                layout(binding = 0) uniform sampler2DArray out_prev;
                layout(location = 0) out vec4 out_raster;
                layout(location = 1) out vec4 out_db;
                IF_ZMODIFY(
                    layout(location = 1) uniform float in_dummy;
                )
                void main()
                {
                    int id_int = gl_PrimitiveID + 1;
                    float id_float = (id_int <= 0x01000000) ? float(id_int) : intBitsToFloat(0x4a800000 + id_int);

                    vec4 prev = texelFetch(out_prev, ivec3(gl_FragCoord.x, gl_FragCoord.y, gl_Layer), 0);
                    float depth_new = var_uvzw.z / var_uvzw.w;
                    if (prev.w == 0 || depth_new <= prev.z)
                        discard;
                    out_raster = vec4(var_uvzw.x, var_uvzw.y, depth_new, id_float);
                    out_db = var_db * var_uvzw.w;
                    IF_ZMODIFY(gl_FragDepth = gl_FragCoord.z + in_dummy;)
                }
            )
        );
    }
    else
    {
        // Geometry shader without bary differential output.
        compileGLShader(NVDR_CTX_PARAMS, s, &s.glGeometryShader, GL_GEOMETRY_SHADER,
            "#version 330\n"
            STRINGIFY_SHADER_SOURCE(
                layout(triangles) in;
                layout(triangle_strip, max_vertices=3) out;
                in int v_layer[];
                in int v_offset[];
                out vec4 var_uvzw;
                void main()
                {
                    int layer_id = v_layer[0];
                    int prim_id = gl_PrimitiveIDIn + v_offset[0];

                    gl_Layer = layer_id; gl_PrimitiveID = prim_id; gl_Position = vec4(gl_in[0].gl_Position.x, gl_in[0].gl_Position.y, gl_in[0].gl_Position.z, gl_in[0].gl_Position.w); var_uvzw = vec4(1.f, 0.f, gl_in[0].gl_Position.z, gl_in[0].gl_Position.w); EmitVertex();
                    gl_Layer = layer_id; gl_PrimitiveID = prim_id; gl_Position = vec4(gl_in[1].gl_Position.x, gl_in[1].gl_Position.y, gl_in[1].gl_Position.z, gl_in[1].gl_Position.w); var_uvzw = vec4(0.f, 1.f, gl_in[1].gl_Position.z, gl_in[1].gl_Position.w); EmitVertex();
                    gl_Layer = layer_id; gl_PrimitiveID = prim_id; gl_Position = vec4(gl_in[2].gl_Position.x, gl_in[2].gl_Position.y, gl_in[2].gl_Position.z, gl_in[2].gl_Position.w); var_uvzw = vec4(0.f, 0.f, gl_in[2].gl_Position.z, gl_in[2].gl_Position.w); EmitVertex();
                }
            )
        );

        // Fragment shader without bary differential output.
        compileGLShader(NVDR_CTX_PARAMS, s, &s.glFragmentShader, GL_FRAGMENT_SHADER,
            "#version 430\n"
            STRINGIFY_SHADER_SOURCE(
                in vec4 var_uvzw;
                layout(location = 0) out vec4 out_raster;
                IF_ZMODIFY(
                    layout(location = 1) uniform float in_dummy;
                )
                void main()
                {
                    int id_int = gl_PrimitiveID + 1;
                    float id_float = (id_int <= 0x01000000) ? float(id_int) : intBitsToFloat(0x4a800000 + id_int);

                    out_raster = vec4(var_uvzw.x, var_uvzw.y, var_uvzw.z / var_uvzw.w, id_float);
                    IF_ZMODIFY(gl_FragDepth = gl_FragCoord.z + in_dummy;)
                }
            )
        );

        // Depth peeling variant of fragment shader.
        compileGLShader(NVDR_CTX_PARAMS, s, &s.glFragmentShaderDP, GL_FRAGMENT_SHADER,
            "#version 430\n"
            STRINGIFY_SHADER_SOURCE(
                in vec4 var_uvzw;
                layout(binding = 0) uniform sampler2DArray out_prev;
                layout(location = 0) out vec4 out_raster;
                IF_ZMODIFY(
                    layout(location = 1) uniform float in_dummy;
                )
                void main()
                {
                    int id_int = gl_PrimitiveID + 1;
                    float id_float = (id_int <= 0x01000000) ? float(id_int) : intBitsToFloat(0x4a800000 + id_int);

                    vec4 prev = texelFetch(out_prev, ivec3(gl_FragCoord.x, gl_FragCoord.y, gl_Layer), 0);
                    float depth_new = var_uvzw.z / var_uvzw.w;
                    if (prev.w == 0 || depth_new <= prev.z)
                        discard;
                    out_raster = vec4(var_uvzw.x, var_uvzw.y, var_uvzw.z / var_uvzw.w, id_float);
                    IF_ZMODIFY(gl_FragDepth = gl_FragCoord.z + in_dummy;)
                }
            )
        );
    }

    // Finalize programs.
    constructGLProgram(NVDR_CTX_PARAMS, &s.glProgram, s.glVertexShader, s.glGeometryShader, s.glFragmentShader);
    constructGLProgram(NVDR_CTX_PARAMS, &s.glProgramDP, s.glVertexShader, s.glGeometryShader, s.glFragmentShaderDP);

    // Construct main fbo and bind permanently.
    NVDR_CHECK_GL_ERROR(glGenFramebuffers(1, &s.glFBO));
    NVDR_CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, s.glFBO));

    // Enable two color attachments.
    GLenum draw_buffers[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    NVDR_CHECK_GL_ERROR(glDrawBuffers(num_outputs, draw_buffers));

    // Construct vertex array object.
    NVDR_CHECK_GL_ERROR(glGenVertexArrays(1, &s.glVAO));
    NVDR_CHECK_GL_ERROR(glBindVertexArray(s.glVAO));

    // Construct position buffer, bind permanently, enable, set ptr.
    NVDR_CHECK_GL_ERROR(glGenBuffers(1, &s.glPosBuffer));
    NVDR_CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, s.glPosBuffer));
    NVDR_CHECK_GL_ERROR(glEnableVertexAttribArray(0));
    NVDR_CHECK_GL_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0));

    // Construct index buffer and bind permanently.
    NVDR_CHECK_GL_ERROR(glGenBuffers(1, &s.glTriBuffer));
    NVDR_CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s.glTriBuffer));

    // Set up depth test.
    NVDR_CHECK_GL_ERROR(glEnable(GL_DEPTH_TEST));
    NVDR_CHECK_GL_ERROR(glDepthFunc(GL_LESS));
    NVDR_CHECK_GL_ERROR(glClearDepth(1.0));

    // Create and bind output buffers. Storage is allocated later.
    NVDR_CHECK_GL_ERROR(glGenTextures(num_outputs, s.glColorBuffer));
    for (int i=0; i < num_outputs; i++)
    {
        NVDR_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_ARRAY, s.glColorBuffer[i]));
        NVDR_CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, s.glColorBuffer[i], 0));
    }

    // Create and bind depth/stencil buffer. Storage is allocated later.
    NVDR_CHECK_GL_ERROR(glGenTextures(1, &s.glDepthStencilBuffer));
    NVDR_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_ARRAY, s.glDepthStencilBuffer));
    NVDR_CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, s.glDepthStencilBuffer, 0));

    // Create texture name for previous output buffer (depth peeling).
    NVDR_CHECK_GL_ERROR(glGenTextures(1, &s.glPrevOutBuffer));
}

void rasterizeResizeBuffers(NVDR_CTX_ARGS, RasterizeGLState& s, bool& changes, int posCount, int triCount, int width, int height, int depth)
{
    changes = false;

    // Resize vertex buffer?
    if (posCount > s.posCount)
    {
        if (s.cudaPosBuffer)
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaPosBuffer));
        s.posCount = (posCount > 64) ? ROUND_UP_BITS(posCount, 2) : 64;
        LOG(INFO) << "Increasing position buffer size to " << s.posCount << " float32";
        NVDR_CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER, s.posCount * sizeof(float), NULL, GL_DYNAMIC_DRAW));
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&s.cudaPosBuffer, s.glPosBuffer, cudaGraphicsRegisterFlagsWriteDiscard));
        changes = true;
    }

    // Resize triangle buffer?
    if (triCount > s.triCount)
    {
        if (s.cudaTriBuffer)
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaTriBuffer));
        s.triCount = (triCount > 64) ? ROUND_UP_BITS(triCount, 2) : 64;
        LOG(INFO) << "Increasing triangle buffer size to " << s.triCount << " int32";
        NVDR_CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER, s.triCount * sizeof(int32_t), NULL, GL_DYNAMIC_DRAW));
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&s.cudaTriBuffer, s.glTriBuffer, cudaGraphicsRegisterFlagsWriteDiscard));
        changes = true;
    }

    // Resize framebuffer?
    if (width > s.width || height > s.height || depth > s.depth)
    {
        int num_outputs = s.enableDB ? 2 : 1;
        if (s.cudaColorBuffer[0])
            for (int i=0; i < num_outputs; i++)
                NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaColorBuffer[i]));

        if (s.cudaPrevOutBuffer)
        {
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaPrevOutBuffer));
            s.cudaPrevOutBuffer = 0;
        }

        // New framebuffer size.
        s.width  = (width > s.width) ? width : s.width;
        s.height = (height > s.height) ? height : s.height;
        s.depth  = (depth > s.depth) ? depth : s.depth;
        s.width  = ROUND_UP(s.width, 32);
        s.height = ROUND_UP(s.height, 32);
        LOG(INFO) << "Increasing frame buffer size to (width, height, depth) = (" << s.width << ", " << s.height << ", " << s.depth << ")";

        // Allocate color buffers.
        for (int i=0; i < num_outputs; i++)
        {
            NVDR_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_ARRAY, s.glColorBuffer[i]));
            NVDR_CHECK_GL_ERROR(glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA32F, s.width, s.height, s.depth, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0));
            NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
            NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
            NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
            NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
        }

        // Allocate depth/stencil buffer.
        NVDR_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_ARRAY, s.glDepthStencilBuffer));
        NVDR_CHECK_GL_ERROR(glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH24_STENCIL8, s.width, s.height, s.depth, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, 0));

        // (Re-)register all GL buffers into Cuda.
        for (int i=0; i < num_outputs; i++)
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&s.cudaColorBuffer[i], s.glColorBuffer[i], GL_TEXTURE_3D, cudaGraphicsRegisterFlagsReadOnly));

        changes = true;
    }
}

void rasterizeRender(NVDR_CTX_ARGS, RasterizeGLState& s, cudaStream_t stream, const float* posPtr, int posCount, int vtxPerInstance, const int32_t* triPtr, int triCount, const int32_t* rangesPtr, int width, int height, int depth, int peeling_idx)
{
    // Only copy inputs if we are on first iteration of depth peeling or not doing it at all.
    if (peeling_idx < 1)
    {
        if (triPtr)
        {
            // Copy both position and triangle buffers.
            void* glPosPtr = NULL;
            void* glTriPtr = NULL;
            size_t posBytes = 0;
            size_t triBytes = 0;
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources(2, &s.cudaPosBuffer, stream));
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&glPosPtr, &posBytes, s.cudaPosBuffer));
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&glTriPtr, &triBytes, s.cudaTriBuffer));
            NVDR_CHECK(posBytes >= posCount * sizeof(float), "mapped GL position buffer size mismatch");
            NVDR_CHECK(triBytes >= triCount * sizeof(int32_t), "mapped GL triangle buffer size mismatch");
            NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(glPosPtr, posPtr, posCount * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(glTriPtr, triPtr, triCount * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(2, &s.cudaPosBuffer, stream));
        }
        else
        {
            // Copy position buffer only. Triangles are already copied and known to be constant.
            void* glPosPtr = NULL;
            size_t posBytes = 0;
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &s.cudaPosBuffer, stream));
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&glPosPtr, &posBytes, s.cudaPosBuffer));
            NVDR_CHECK(posBytes >= posCount * sizeof(float), "mapped GL position buffer size mismatch");
            NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(glPosPtr, posPtr, posCount * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &s.cudaPosBuffer, stream));
        }
    }

    // Select program based on whether we have a depth peeling input or not.
    if (peeling_idx < 1)
    {
        // Normal case: No peeling, or peeling disabled.
        NVDR_CHECK_GL_ERROR(glUseProgram(s.glProgram));
    }
    else
    {
        // If we don't have a third buffer yet, create one.
        if (!s.cudaPrevOutBuffer)
        {
            NVDR_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_ARRAY, s.glPrevOutBuffer));
            NVDR_CHECK_GL_ERROR(glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA32F, s.width, s.height, s.depth, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0));
            NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
            NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
            NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
            NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&s.cudaPrevOutBuffer, s.glPrevOutBuffer, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsReadOnly));
        }

        // Swap the GL buffers.
        GLuint glTempBuffer = s.glPrevOutBuffer;
        s.glPrevOutBuffer = s.glColorBuffer[0];
        s.glColorBuffer[0] = glTempBuffer;

        // Swap the Cuda buffers.
        cudaGraphicsResource_t cudaTempBuffer = s.cudaPrevOutBuffer;
        s.cudaPrevOutBuffer = s.cudaColorBuffer[0];
        s.cudaColorBuffer[0] = cudaTempBuffer;

        // Bind the new output buffer.
        NVDR_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_ARRAY, s.glColorBuffer[0]));
        NVDR_CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, s.glColorBuffer[0], 0));

        // Bind old buffer as the input texture.
        NVDR_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_ARRAY, s.glPrevOutBuffer));

        // Activate the correct program.
        NVDR_CHECK_GL_ERROR(glUseProgram(s.glProgramDP));
    }

    // Set viewport, clear color buffer(s) and depth/stencil buffer.
    NVDR_CHECK_GL_ERROR(glViewport(0, 0, width, height));
    NVDR_CHECK_GL_ERROR(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

    // If outputting bary differentials, set resolution uniform
    if (s.enableDB)
        NVDR_CHECK_GL_ERROR(glUniform2f(0, 2.f / (float)width, 2.f / (float)height));

    // Set the dummy uniform if depth modification workaround is active.
    if (s.enableZModify)
        NVDR_CHECK_GL_ERROR(glUniform1f(1, 0.f));

    // Render the meshes.
    if (depth == 1 && !rangesPtr)
    {
        // Trivial case.
        NVDR_CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, triCount, GL_UNSIGNED_INT, 0));
    }
    else
    {
        // Populate a buffer for draw commands and execute it.
        std::vector<GLDrawCmd> drawCmdBuffer(depth);

        if (!rangesPtr)
        {
            // Fill in range array to instantiate the same triangles for each output layer.
            // Triangle IDs starts at zero (i.e., one) for each layer, so they correspond to
            // the first dimension in addressing the triangle array.
            for (int i=0; i < depth; i++)
            {
                GLDrawCmd& cmd = drawCmdBuffer[i];
                cmd.firstIndex    = 0;
                cmd.count         = triCount;
                cmd.baseVertex    = vtxPerInstance * i;
                cmd.baseInstance  = 0;
                cmd.instanceCount = 1;
            }
        }
        else
        {
            // Fill in the range array according to user-given ranges. Triangle IDs point
            // to the input triangle array, NOT index within range, so they correspond to
            // the first dimension in addressing the triangle array.
            for (int i=0, j=0; i < depth; i++)
            {
                GLDrawCmd& cmd = drawCmdBuffer[i];
                int first = rangesPtr[j++];
                int count = rangesPtr[j++];
                NVDR_CHECK(first >= 0 && count >= 0, "range contains negative values");
                NVDR_CHECK((first + count) * 3 <= triCount, "range extends beyond end of triangle buffer");
                cmd.firstIndex    = first * 3;
                cmd.count         = count * 3;
                cmd.baseVertex    = 0;
                cmd.baseInstance  = first;
                cmd.instanceCount = 1;
            }
        }

        // Draw!
        NVDR_CHECK_GL_ERROR(glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, &drawCmdBuffer[0], depth, sizeof(GLDrawCmd)));
    }
}

void rasterizeCopyResults(NVDR_CTX_ARGS, RasterizeGLState& s, cudaStream_t stream, float** outputPtr, int width, int height, int depth)
{
    // Copy color buffers to output tensors.
    cudaArray_t array = 0;
    cudaChannelFormatDesc arrayDesc = {};   // For error checking.
    cudaExtent arrayExt = {};               // For error checking.
    int num_outputs = s.enableDB ? 2 : 1;
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources(num_outputs, s.cudaColorBuffer, stream));
    for (int i=0; i < num_outputs; i++)
    {
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, s.cudaColorBuffer[i], 0, 0));
        NVDR_CHECK_CUDA_ERROR(cudaArrayGetInfo(&arrayDesc, &arrayExt, NULL, array));
        NVDR_CHECK(arrayDesc.f == cudaChannelFormatKindFloat, "CUDA mapped array data kind mismatch");
        NVDR_CHECK(arrayDesc.x == 32 && arrayDesc.y == 32 && arrayDesc.z == 32 && arrayDesc.w == 32, "CUDA mapped array data width mismatch");
        NVDR_CHECK(arrayExt.width >= width && arrayExt.height >= height && arrayExt.depth >= depth, "CUDA mapped array extent mismatch");
        cudaMemcpy3DParms p = {0};
        p.srcArray = array;
        p.dstPtr.ptr = outputPtr[i];
        p.dstPtr.pitch = width * 4 * sizeof(float);
        p.dstPtr.xsize = width;
        p.dstPtr.ysize = height;
        p.extent.width = width;
        p.extent.height = height;
        p.extent.depth = depth;
        p.kind = cudaMemcpyDeviceToDevice;
        NVDR_CHECK_CUDA_ERROR(cudaMemcpy3DAsync(&p, stream));
    }
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(num_outputs, s.cudaColorBuffer, stream));
}

void rasterizeReleaseBuffers(NVDR_CTX_ARGS, RasterizeGLState& s)
{
    int num_outputs = s.enableDB ? 2 : 1;

    if (s.cudaPosBuffer)
    {
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaPosBuffer));
        s.cudaPosBuffer = 0;
    }

    if (s.cudaTriBuffer)
    {
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaTriBuffer));
        s.cudaTriBuffer = 0;
    }

    for (int i=0; i < num_outputs; i++)
    {
        if (s.cudaColorBuffer[i])
        {
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaColorBuffer[i]));
            s.cudaColorBuffer[i] = 0;
        }
    }

    if (s.cudaPrevOutBuffer)
    {
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaPrevOutBuffer));
        s.cudaPrevOutBuffer = 0;
    }
}

//------------------------------------------------------------------------
