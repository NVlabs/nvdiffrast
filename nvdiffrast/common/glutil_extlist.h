// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef GL_VERSION_1_2
GLUTIL_EXT(void,   glTexImage3D,                GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels);
#endif
#ifndef GL_VERSION_1_5
GLUTIL_EXT(void,   glBindBuffer,                GLenum target, GLuint buffer);
GLUTIL_EXT(void,   glBufferData,                GLenum target, ptrdiff_t size, const void* data, GLenum usage);
GLUTIL_EXT(void,   glGenBuffers,                GLsizei n, GLuint* buffers);
#endif
#ifndef GL_VERSION_2_0
GLUTIL_EXT(void,   glAttachShader,              GLuint program, GLuint shader);
GLUTIL_EXT(void,   glCompileShader,             GLuint shader);
GLUTIL_EXT(GLuint, glCreateProgram,             void);
GLUTIL_EXT(GLuint, glCreateShader,              GLenum type);
GLUTIL_EXT(void,   glDrawBuffers,               GLsizei n, const GLenum* bufs);
GLUTIL_EXT(void,   glEnableVertexAttribArray,   GLuint index);
GLUTIL_EXT(void,   glGetProgramInfoLog,         GLuint program, GLsizei bufSize, GLsizei* length, char* infoLog);
GLUTIL_EXT(void,   glGetProgramiv,              GLuint program, GLenum pname, GLint* param);
GLUTIL_EXT(void,   glLinkProgram,               GLuint program);
GLUTIL_EXT(void,   glShaderSource,              GLuint shader, GLsizei count, const char *const* string, const GLint* length);
GLUTIL_EXT(void,   glUniform1f,                 GLint location, GLfloat v0);
GLUTIL_EXT(void,   glUniform2f,                 GLint location, GLfloat v0, GLfloat v1);
GLUTIL_EXT(void,   glUseProgram,                GLuint program);
GLUTIL_EXT(void,   glVertexAttribPointer,       GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void* pointer);
#endif
#ifndef GL_VERSION_3_2
GLUTIL_EXT(void,   glFramebufferTexture,        GLenum target, GLenum attachment, GLuint texture, GLint level);
#endif
#ifndef GL_ARB_framebuffer_object
GLUTIL_EXT(void,   glBindFramebuffer,           GLenum target, GLuint framebuffer);
GLUTIL_EXT(void,   glGenFramebuffers,           GLsizei n, GLuint* framebuffers);
#endif
#ifndef GL_ARB_vertex_array_object
GLUTIL_EXT(void,   glBindVertexArray,           GLuint array);
GLUTIL_EXT(void,   glGenVertexArrays,           GLsizei n, GLuint* arrays);
#endif
#ifndef GL_ARB_multi_draw_indirect
GLUTIL_EXT(void,   glMultiDrawElementsIndirect, GLenum mode, GLenum type, const void *indirect, GLsizei primcount, GLsizei stride);
#endif

//------------------------------------------------------------------------
