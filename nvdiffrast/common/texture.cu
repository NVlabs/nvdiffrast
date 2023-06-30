// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "common.h"
#include "texture.h"

//------------------------------------------------------------------------
// Memory access and math helpers.

static __device__ __forceinline__ void accum_from_mem(float* a, int s, float  b, float c) { a[0] += b * c; }
static __device__ __forceinline__ void accum_from_mem(float* a, int s, float2 b, float c) { a[0] += b.x * c; a[s] += b.y * c; }
static __device__ __forceinline__ void accum_from_mem(float* a, int s, float4 b, float c) { a[0] += b.x * c; a[s] += b.y * c; a[2*s] += b.z * c; a[3*s] += b.w * c; }
static __device__ __forceinline__ void accum_to_mem(float&  a, float* b, int s) { a += b[0]; }
static __device__ __forceinline__ void accum_to_mem(float2& a, float* b, int s) { float2 v = a; v.x += b[0]; v.y += b[s]; a = v; }
static __device__ __forceinline__ void accum_to_mem(float4& a, float* b, int s) { float4 v = a; v.x += b[0]; v.y += b[s]; v.z += b[2*s]; v.w += b[3*s]; a = v; }
static __device__ __forceinline__ bool isfinite_vec3(const float3& a) { return isfinite(a.x) && isfinite(a.y) && isfinite(a.z); }
static __device__ __forceinline__ bool isfinite_vec4(const float4& a) { return isfinite(a.x) && isfinite(a.y) && isfinite(a.z) && isfinite(a.w); }
template<class T> static __device__ __forceinline__ T lerp  (const T& a, const T& b, float c) { return a + c * (b - a); }
template<class T> static __device__ __forceinline__ T bilerp(const T& a, const T& b, const T& c, const T& d, const float2& e) { return lerp(lerp(a, b, e.x), lerp(c, d, e.x), e.y); }

//------------------------------------------------------------------------
// Cube map wrapping for smooth filtering across edges and corners. At corners,
// one of the texture coordinates will be negative. For correct interpolation,
// the missing texel must take the average color of the other three.

static __constant__ uint32_t c_cubeWrapMask1[48] =
{
    0x1530a440, 0x1133a550, 0x6103a110, 0x1515aa44, 0x6161aa11, 0x40154a04, 0x44115a05, 0x04611a01,
    0x2630a440, 0x2233a550, 0x5203a110, 0x2626aa44, 0x5252aa11, 0x40264a04, 0x44225a05, 0x04521a01,
    0x32608064, 0x3366a055, 0x13062091, 0x32328866, 0x13132299, 0x50320846, 0x55330a55, 0x05130219,
    0x42508064, 0x4455a055, 0x14052091, 0x42428866, 0x14142299, 0x60420846, 0x66440a55, 0x06140219,
    0x5230a044, 0x5533a055, 0x1503a011, 0x5252aa44, 0x1515aa11, 0x40520a44, 0x44550a55, 0x04150a11,
    0x6130a044, 0x6633a055, 0x2603a011, 0x6161aa44, 0x2626aa11, 0x40610a44, 0x44660a55, 0x04260a11,
};

static __constant__ uint8_t c_cubeWrapMask2[48] =
{
    0x26, 0x33, 0x11, 0x05, 0x00, 0x09, 0x0c, 0x04, 0x04, 0x00, 0x00, 0x05, 0x00, 0x81, 0xc0, 0x40,
    0x02, 0x03, 0x09, 0x00, 0x0a, 0x00, 0x00, 0x02, 0x64, 0x30, 0x90, 0x55, 0xa0, 0x99, 0xcc, 0x64,
    0x24, 0x30, 0x10, 0x05, 0x00, 0x01, 0x00, 0x00, 0x06, 0x03, 0x01, 0x05, 0x00, 0x89, 0xcc, 0x44,
};

static __device__ __forceinline__ int4 wrapCubeMap(int face, int ix0, int ix1, int iy0, int iy1, int w)
{
    // Calculate case number.
    int cx = (ix0 < 0) ? 0 : (ix1 >= w) ? 2 : 1;
    int cy = (iy0 < 0) ? 0 : (iy1 >= w) ? 6 : 3;
    int c = cx + cy;
    if (c >= 5)
        c--;
    c = (face << 3) + c;

    // Compute coordinates and faces.
    unsigned int m = c_cubeWrapMask1[c];
    int x0 = (m >>  0) & 3; x0 = (x0 == 0) ? 0 : (x0 == 1) ? ix0 : iy0;
    int x1 = (m >>  2) & 3; x1 = (x1 == 0) ? 0 : (x1 == 1) ? ix1 : iy0;
    int x2 = (m >>  4) & 3; x2 = (x2 == 0) ? 0 : (x2 == 1) ? ix0 : iy1;
    int x3 = (m >>  6) & 3; x3 = (x3 == 0) ? 0 : (x3 == 1) ? ix1 : iy1;
    int y0 = (m >>  8) & 3; y0 = (y0 == 0) ? 0 : (y0 == 1) ? ix0 : iy0;
    int y1 = (m >> 10) & 3; y1 = (y1 == 0) ? 0 : (y1 == 1) ? ix1 : iy0;
    int y2 = (m >> 12) & 3; y2 = (y2 == 0) ? 0 : (y2 == 1) ? ix0 : iy1;
    int y3 = (m >> 14) & 3; y3 = (y3 == 0) ? 0 : (y3 == 1) ? ix1 : iy1;
    int f0 = ((m >> 16) & 15) - 1;
    int f1 = ((m >> 20) & 15) - 1;
    int f2 = ((m >> 24) & 15) - 1;
    int f3 = ((m >> 28)     ) - 1;

    // Flips.
    unsigned int f = c_cubeWrapMask2[c];
    int w1 = w - 1;
    if (f & 0x01) x0 = w1 - x0;
    if (f & 0x02) x1 = w1 - x1;
    if (f & 0x04) x2 = w1 - x2;
    if (f & 0x08) x3 = w1 - x3;
    if (f & 0x10) y0 = w1 - y0;
    if (f & 0x20) y1 = w1 - y1;
    if (f & 0x40) y2 = w1 - y2;
    if (f & 0x80) y3 = w1 - y3;

    // Done.
    int4 tcOut;
    tcOut.x = x0 + (y0 + f0 * w) * w;
    tcOut.y = x1 + (y1 + f1 * w) * w;
    tcOut.z = x2 + (y2 + f2 * w) * w;
    tcOut.w = x3 + (y3 + f3 * w) * w;
    return tcOut;
}

//------------------------------------------------------------------------
// Cube map indexing and gradient functions.

// Map a 3D lookup vector into an (s,t) face coordinates (returned in first .
// two parameters) and face index.
static __device__ __forceinline__ int indexCubeMap(float& x, float& y, float z)
{
    float ax = fabsf(x);
    float ay = fabsf(y);
    float az = fabsf(z);
    int idx;
    float c;
    if (az > fmaxf(ax, ay)) { idx = 4; c = z; }
    else if (ay > ax)       { idx = 2; c = y; y = z; }
    else                    { idx = 0; c = x; x = z; }
    if (c < 0.f) idx += 1;
    float m = __frcp_rz(fabsf(c)) * .5;
    float m0 = __uint_as_float(__float_as_uint(m) ^ ((0x21u >> idx) << 31));
    float m1 = (idx != 2) ? -m : m;
    x = x * m0 + .5;
    y = y * m1 + .5;
    if (!isfinite(x) || !isfinite(y))
        return -1; // Invalid uv.
    x = fminf(fmaxf(x, 0.f), 1.f);
    y = fminf(fmaxf(y, 0.f), 1.f);
    return idx;
}

// Based on dA/d{s,t}, compute dA/d{x,y,z} at a given 3D lookup vector.
static __device__ __forceinline__ float3 indexCubeMapGrad(float3 uv, float gu, float gv)
{
    float ax = fabsf(uv.x);
    float ay = fabsf(uv.y);
    float az = fabsf(uv.z);
    int idx;
    float c;
    float c0 = gu;
    float c1 = gv;
    if (az > fmaxf(ax, ay)) { idx = 0x10; c = uv.z; c0 *= uv.x; c1 *= uv.y; }
    else if (ay > ax)       { idx = 0x04; c = uv.y; c0 *= uv.x; c1 *= uv.z; }
    else                    { idx = 0x01; c = uv.x; c0 *= uv.z; c1 *= uv.y; }
    if (c < 0.f) idx += idx;
    float m = __frcp_rz(fabsf(c));
    c0 = (idx & 0x34) ? -c0 : c0;
    c1 = (idx & 0x2e) ? -c1 : c1;
    float gl = (c0 + c1) * m;
    float gx = (idx & 0x03) ? gl : (idx & 0x20) ? -gu : gu;
    float gy = (idx & 0x0c) ? gl : -gv;
    float gz = (idx & 0x30) ? gl : (idx & 0x03) ? gu : gv;
    gz = (idx & 0x09) ? -gz : gz;
    float3 res = make_float3(gx, gy, gz) * (m * .5f);
    if (!isfinite_vec3(res))
        return make_float3(0.f, 0.f, 0.f); // Invalid uv.
    return res;
}

// Based on dL/d(d{s,t}/s{X,Y}), compute dL/d(d{x,y,z}/d{X,Y}). This is just two
// indexCubeMapGrad() functions rolled together.
static __device__ __forceinline__ void indexCubeMapGrad4(float3 uv, float4 dw, float3& g0, float3& g1)
{
    float ax = fabsf(uv.x);
    float ay = fabsf(uv.y);
    float az = fabsf(uv.z);
    int idx;
    float c, c0, c1;
    if (az > fmaxf(ax, ay)) { idx = 0x10; c = uv.z; c0 = uv.x; c1 = uv.y; }
    else if (ay > ax)       { idx = 0x04; c = uv.y; c0 = uv.x; c1 = uv.z; }
    else                    { idx = 0x01; c = uv.x; c0 = uv.z; c1 = uv.y; }
    if (c < 0.f) idx += idx;
    float m = __frcp_rz(fabsf(c));
    c0 = (idx & 0x34) ? -c0 : c0;
    c1 = (idx & 0x2e) ? -c1 : c1;
    float gl0 = (dw.x * c0 + dw.z * c1) * m;
    float gl1 = (dw.y * c0 + dw.w * c1) * m;
    float gx0 = (idx & 0x03) ? gl0 : (idx & 0x20) ? -dw.x : dw.x;
    float gx1 = (idx & 0x03) ? gl1 : (idx & 0x20) ? -dw.y : dw.y;
    float gy0 = (idx & 0x0c) ? gl0 : -dw.z;
    float gy1 = (idx & 0x0c) ? gl1 : -dw.w;
    float gz0 = (idx & 0x30) ? gl0 : (idx & 0x03) ? dw.x : dw.z;
    float gz1 = (idx & 0x30) ? gl1 : (idx & 0x03) ? dw.y : dw.w;
    if (idx & 0x09)
    {
        gz0 = -gz0;
        gz1 = -gz1;
    }
    g0 = make_float3(gx0, gy0, gz0) * (m * .5f);
    g1 = make_float3(gx1, gy1, gz1) * (m * .5f);
    if (!isfinite_vec3(g0) || !isfinite_vec3(g1))
    {
        g0 = make_float3(0.f, 0.f, 0.f); // Invalid uv.
        g1 = make_float3(0.f, 0.f, 0.f);
    }
}

// Compute d{s,t}/d{X,Y} based on d{x,y,z}/d{X,Y} at a given 3D lookup vector.
// Result is (ds/dX, ds/dY, dt/dX, dt/dY).
static __device__ __forceinline__ float4 indexCubeMapGradST(float3 uv, float3 dvdX, float3 dvdY)
{
    float ax = fabsf(uv.x);
    float ay = fabsf(uv.y);
    float az = fabsf(uv.z);
    int idx;
    float c, gu, gv;
    if (az > fmaxf(ax, ay)) { idx = 0x10; c = uv.z; gu = uv.x; gv = uv.y; }
    else if (ay > ax)       { idx = 0x04; c = uv.y; gu = uv.x; gv = uv.z; }
    else                    { idx = 0x01; c = uv.x; gu = uv.z; gv = uv.y; }
    if (c < 0.f) idx += idx;
    if (idx & 0x09)
    {
        dvdX.z = -dvdX.z;
        dvdY.z = -dvdY.z;
    }
    float m = __frcp_rz(fabsf(c));
    float dm = m * .5f;
    float mm = m * dm;
    gu *= (idx & 0x34) ? -mm : mm;
    gv *= (idx & 0x2e) ? -mm : mm;

    float4 res;
    if (idx & 0x03)
    {
        res = make_float4(gu * dvdX.x + dm * dvdX.z,
                          gu * dvdY.x + dm * dvdY.z,
                          gv * dvdX.x - dm * dvdX.y,
                          gv * dvdY.x - dm * dvdY.y);
    }
    else if (idx & 0x0c)
    {
        res = make_float4(gu * dvdX.y + dm * dvdX.x,
                          gu * dvdY.y + dm * dvdY.x,
                          gv * dvdX.y + dm * dvdX.z,
                          gv * dvdY.y + dm * dvdY.z);
    }
    else // (idx & 0x30)
    {
        res = make_float4(gu * dvdX.z + copysignf(dm, c) * dvdX.x,
                          gu * dvdY.z + copysignf(dm, c) * dvdY.x,
                          gv * dvdX.z - dm * dvdX.y,
                          gv * dvdY.z - dm * dvdY.y);
    }

    if (!isfinite_vec4(res))
        return make_float4(0.f, 0.f, 0.f, 0.f);

    return res;
}

// Compute d(d{s,t}/d{X,Y})/d{x,y,z}, i.e., how the pixel derivatives of 2D face
// coordinates change w.r.t. 3D texture coordinate vector, returned as follows:
//   |  d(ds/dX)/dx  d(ds/dY)/dx  d(dt/dX)/dx  d(dt/dY)/dx  |
//   |  d(ds/dX)/dy  d(ds/dY)/dy  d(dt/dX)/dy  d(dt/dY)/dy  |
//   |  d(ds/dX)/dz  d(ds/dY)/dz  d(dt/dX)/dz  d(dt/dY)/dz  |
static __device__ __forceinline__ void indexCubeMapGrad2(float3 uv, float3 dvdX, float3 dvdY, float4& dx, float4& dy, float4& dz)
{
    float ax = fabsf(uv.x);
    float ay = fabsf(uv.y);
    float az = fabsf(uv.z);
    int idx;
    float c, gu, gv;
    if (az > fmaxf(ax, ay)) { idx = 0x10; c = uv.z; gu = uv.x; gv = uv.y; }
    else if (ay > ax)       { idx = 0x04; c = uv.y; gu = uv.x; gv = uv.z; }
    else                    { idx = 0x01; c = uv.x; gu = uv.z; gv = uv.y; }
    if (c < 0.f) idx += idx;

    if (idx & 0x09)
    {
        dvdX.z = -dvdX.z;
        dvdY.z = -dvdY.z;
    }

    float m = __frcp_rz(c);
    float dm = -m * fabsf(m) * .5;
    float mm = m * m * .5;
    float mu = (idx & 0x34) ? -mm : mm;
    float mv = (idx & 0x2e) ? -mm : mm;
    gu *= -2.0 * m * mu;
    gv *= -2.0 * m * mv;

    if (idx & 0x03)
    {
        dx.x = gu * dvdX.x + dm * dvdX.z;
        dx.y = gu * dvdY.x + dm * dvdY.z;
        dx.z = gv * dvdX.x - dm * dvdX.y;
        dx.w = gv * dvdY.x - dm * dvdY.y;
        dy.x = 0.f;
        dy.y = 0.f;
        dy.z = mv * dvdX.x;
        dy.w = mv * dvdY.x;
        dz.x = mu * dvdX.x;
        dz.y = mu * dvdY.x;
        dz.z = 0.f;
        dz.w = 0.f;
    }
    else if (idx & 0x0c)
    {
        dx.x = mu * dvdX.y;
        dx.y = mu * dvdY.y;
        dx.z = 0.f;
        dx.w = 0.f;
        dy.x = gu * dvdX.y + dm * dvdX.x;
        dy.y = gu * dvdY.y + dm * dvdY.x;
        dy.z = gv * dvdX.y + dm * dvdX.z;
        dy.w = gv * dvdY.y + dm * dvdY.z;
        dz.x = 0.f;
        dz.y = 0.f;
        dz.z = mv * dvdX.y;
        dz.w = mv * dvdY.y;
    }
    else // (idx & 0x30)
    {
        dx.x = mu * dvdX.z;
        dx.y = mu * dvdY.z;
        dx.z = 0.f;
        dx.w = 0.f;
        dy.x = 0.f;
        dy.y = 0.f;
        dy.z = mv * dvdX.z;
        dy.w = mv * dvdY.z;
        dz.x = gu * dvdX.z - fabsf(dm) * dvdX.x;
        dz.y = gu * dvdY.z - fabsf(dm) * dvdY.x;
        dz.z = gv * dvdX.z - dm * dvdX.y;
        dz.w = gv * dvdY.z - dm * dvdY.y;
    }
}

//------------------------------------------------------------------------
// General texture indexing.

template <bool CUBE_MODE>
static __device__ __forceinline__ int indexTextureNearest(const TextureKernelParams& p, float3 uv, int tz)
{
    int w = p.texWidth;
    int h = p.texHeight;
    float u = uv.x;
    float v = uv.y;

    // Cube map indexing.
    if (CUBE_MODE)
    {
        // No wrap. Fold face index into tz right away.
        int idx = indexCubeMap(u, v, uv.z); // Rewrites u, v.
        if (idx < 0)
            return -1; // Invalid uv.
        tz = 6 * tz + idx;
    }
    else
    {
        // Handle boundary.
        if (p.boundaryMode == TEX_BOUNDARY_MODE_WRAP)
        {
            u = u - (float)__float2int_rd(u);
            v = v - (float)__float2int_rd(v);
        }
    }

    u = u * (float)w;
    v = v * (float)h;

    int iu = __float2int_rd(u);
    int iv = __float2int_rd(v);

    // In zero boundary mode, return texture address -1.
    if (!CUBE_MODE && p.boundaryMode == TEX_BOUNDARY_MODE_ZERO)
    {
        if (iu < 0 || iu >= w || iv < 0 || iv >= h)
            return -1;
    }

    // Otherwise clamp and calculate the coordinate properly.
    iu = min(max(iu, 0), w-1);
    iv = min(max(iv, 0), h-1);
    return iu + w * (iv + tz * h);
}

template <bool CUBE_MODE>
static __device__ __forceinline__ float2 indexTextureLinear(const TextureKernelParams& p, float3 uv, int tz, int4& tcOut, int level)
{
    // Mip level size.
    int2 sz = mipLevelSize(p, level);
    int w = sz.x;
    int h = sz.y;

    // Compute texture-space u, v.
    float u = uv.x;
    float v = uv.y;
    bool clampU = false;
    bool clampV = false;

    // Cube map indexing.
    int face = 0;
    if (CUBE_MODE)
    {
        // Neither clamp or wrap.
        face = indexCubeMap(u, v, uv.z); // Rewrites u, v.
        if (face < 0)
        {
            tcOut.x = tcOut.y = tcOut.z = tcOut.w = -1; // Invalid uv.
            return make_float2(0.f, 0.f);
        }
        u = u * (float)w - 0.5f;
        v = v * (float)h - 0.5f;
    }
    else
    {
        if (p.boundaryMode == TEX_BOUNDARY_MODE_WRAP)
        {
            // Wrap.
            u = u - (float)__float2int_rd(u);
            v = v - (float)__float2int_rd(v);
        }

        // Move to texel space.
        u = u * (float)w - 0.5f;
        v = v * (float)h - 0.5f;

        if (p.boundaryMode == TEX_BOUNDARY_MODE_CLAMP)
        {
            // Clamp to center of edge texels.
            u = fminf(fmaxf(u, 0.f), w - 1.f);
            v = fminf(fmaxf(v, 0.f), h - 1.f);
            clampU = (u == 0.f || u == w - 1.f);
            clampV = (v == 0.f || v == h - 1.f);
        }
    }

    // Compute texel coordinates and weights.
    int iu0 = __float2int_rd(u);
    int iv0 = __float2int_rd(v);
    int iu1 = iu0 + (clampU ? 0 : 1); // Ensure zero u/v gradients with clamped.
    int iv1 = iv0 + (clampV ? 0 : 1);
    u -= (float)iu0;
    v -= (float)iv0;

    // Cube map wrapping.
    bool cubeWrap = CUBE_MODE && (iu0 < 0 || iv0 < 0 || iu1 >= w || iv1 >= h);
    if (cubeWrap)
    {
        tcOut = wrapCubeMap(face, iu0, iu1, iv0, iv1, w);
        tcOut += 6 * tz * w * h;  // Bring in tz.
        return make_float2(u, v); // Done.
    }

    // Fold cube map face into tz.
    if (CUBE_MODE)
        tz = 6 * tz + face;

    // Wrap overflowing texel indices.
    if (!CUBE_MODE && p.boundaryMode == TEX_BOUNDARY_MODE_WRAP)
    {
        if (iu0 < 0) iu0 += w;
        if (iv0 < 0) iv0 += h;
        if (iu1 >= w) iu1 -= w;
        if (iv1 >= h) iv1 -= h;
    }

    // Coordinates with tz folded in.
    int iu0z = iu0 + tz * w * h;
    int iu1z = iu1 + tz * w * h;
    tcOut.x = iu0z + w * iv0;
    tcOut.y = iu1z + w * iv0;
    tcOut.z = iu0z + w * iv1;
    tcOut.w = iu1z + w * iv1;

    // Invalidate texture addresses outside unit square if we are in zero mode.
    if (!CUBE_MODE && p.boundaryMode == TEX_BOUNDARY_MODE_ZERO)
    {
        bool iu0_out = (iu0 < 0 || iu0 >= w);
        bool iu1_out = (iu1 < 0 || iu1 >= w);
        bool iv0_out = (iv0 < 0 || iv0 >= h);
        bool iv1_out = (iv1 < 0 || iv1 >= h);
        if (iu0_out || iv0_out) tcOut.x = -1;
        if (iu1_out || iv0_out) tcOut.y = -1;
        if (iu0_out || iv1_out) tcOut.z = -1;
        if (iu1_out || iv1_out) tcOut.w = -1;
    }

    // All done.
    return make_float2(u, v);
}

//------------------------------------------------------------------------
// Mip level calculation.

template <bool CUBE_MODE, bool BIAS_ONLY, int FILTER_MODE>
static __device__ __forceinline__ void calculateMipLevel(int& level0, int& level1, float& flevel, const TextureKernelParams& p, int pidx, float3 uv, float4* pdw, float3* pdfdv)
{
    // Do nothing if mips not in use.
    if (FILTER_MODE == TEX_MODE_NEAREST || FILTER_MODE == TEX_MODE_LINEAR)
        return;

    // Determine mip level based on UV pixel derivatives. If no derivatives are given (mip level bias only), leave as zero.
    if (!BIAS_ONLY)
    {
        // Get pixel derivatives of texture coordinates.
        float4 uvDA;
        float3 dvdX, dvdY; // Gradients use these later.
        if (CUBE_MODE)
        {
            // Fetch.
            float2 d0 = ((const float2*)p.uvDA)[3 * pidx + 0];
            float2 d1 = ((const float2*)p.uvDA)[3 * pidx + 1];
            float2 d2 = ((const float2*)p.uvDA)[3 * pidx + 2];

            // Map d{x,y,z}/d{X,Y} into d{s,t}/d{X,Y}.
            dvdX = make_float3(d0.x, d1.x, d2.x); // d{x,y,z}/dX
            dvdY = make_float3(d0.y, d1.y, d2.y); // d{x,y,z}/dY
            uvDA = indexCubeMapGradST(uv, dvdX, dvdY); // d{s,t}/d{X,Y}
        }
        else
        {
            // Fetch.
            uvDA = ((const float4*)p.uvDA)[pidx];
        }

        // Scaling factors.
        float uscl = p.texWidth;
        float vscl = p.texHeight;

        // d[s,t]/d[X,Y].
        float dsdx = uvDA.x * uscl;
        float dsdy = uvDA.y * uscl;
        float dtdx = uvDA.z * vscl;
        float dtdy = uvDA.w * vscl;

        // Calculate footprint axis lengths.
        float A = dsdx*dsdx + dtdx*dtdx;
        float B = dsdy*dsdy + dtdy*dtdy;
        float C = dsdx*dsdy + dtdx*dtdy;
        float l2b = 0.5 * (A + B);
        float l2n = 0.25 * (A-B)*(A-B) + C*C;
        float l2a = sqrt(l2n);
        float lenMinorSqr = fmaxf(0.0, l2b - l2a);
        float lenMajorSqr = l2b + l2a;

        // Footprint vs. mip level gradient.
        if (pdw && FILTER_MODE == TEX_MODE_LINEAR_MIPMAP_LINEAR)
        {
            float dw   = 0.72134752f / (l2n + l2a * l2b); // Constant is 0.5/ln(2).
            float AB   = dw * .5f * (A - B);
            float Cw   = dw * C;
            float l2aw = dw * l2a;
            float d_f_ddsdX = uscl * (dsdx * (l2aw + AB) + dsdy * Cw);
            float d_f_ddsdY = uscl * (dsdy * (l2aw - AB) + dsdx * Cw);
            float d_f_ddtdX = vscl * (dtdx * (l2aw + AB) + dtdy * Cw);
            float d_f_ddtdY = vscl * (dtdy * (l2aw - AB) + dtdx * Cw);

            float4 d_f_dw = make_float4(d_f_ddsdX, d_f_ddsdY, d_f_ddtdX, d_f_ddtdY);
            if (!CUBE_MODE)
                *pdw = isfinite_vec4(d_f_dw) ? d_f_dw : make_float4(0.f, 0.f, 0.f, 0.f);

            // In cube maps, there is also a texture coordinate vs. mip level gradient.
            // Only output nonzero vectors if both are free of inf/Nan garbage.
            if (CUBE_MODE)
            {
                float4 dx, dy, dz;
                indexCubeMapGrad2(uv, dvdX, dvdY, dx, dy, dz);
                float3 d_dsdX_dv = make_float3(dx.x, dy.x, dz.x);
                float3 d_dsdY_dv = make_float3(dx.y, dy.y, dz.y);
                float3 d_dtdX_dv = make_float3(dx.z, dy.z, dz.z);
                float3 d_dtdY_dv = make_float3(dx.w, dy.w, dz.w);

                float3 d_f_dv = make_float3(0.f, 0.f, 0.f);
                d_f_dv += d_dsdX_dv * d_f_ddsdX;
                d_f_dv += d_dsdY_dv * d_f_ddsdY;
                d_f_dv += d_dtdX_dv * d_f_ddtdX;
                d_f_dv += d_dtdY_dv * d_f_ddtdY;

                bool finite = isfinite_vec4(d_f_dw) && isfinite_vec3(d_f_dv);
                *pdw   = finite ? d_f_dw : make_float4(0.f, 0.f, 0.f, 0.f);
                *pdfdv = finite ? d_f_dv : make_float3(0.f, 0.f, 0.f);
            }
        }

        // Finally, calculate mip level.
        flevel = .5f * __log2f(lenMajorSqr); // May be inf/NaN, but clamp fixes it.
    }

    // Bias the mip level and clamp.
    if (p.mipLevelBias)
        flevel += p.mipLevelBias[pidx];
    flevel = fminf(fmaxf(flevel, 0.f), (float)p.mipLevelMax);

    // Calculate levels depending on filter mode.
    level0 = __float2int_rd(flevel);

    // Leave everything else at zero if flevel == 0 (magnification) or when in linear-mipmap-nearest mode.
    if (FILTER_MODE == TEX_MODE_LINEAR_MIPMAP_LINEAR && flevel > 0.f)
    {
        level1 = min(level0 + 1, p.mipLevelMax);
        flevel -= level0; // Fractional part. Zero if clamped on last level.
    }
}

//------------------------------------------------------------------------
// Texel fetch and accumulator helpers that understand cube map corners.

template<class T>
static __device__ __forceinline__ void fetchQuad(T& a00, T& a10, T& a01, T& a11, const float* pIn, int4 tc, bool corner)
{
    // For invalid cube map uv, tc will be all negative, and all texel values will be zero.
    if (corner)
    {
        T avg = zero_value<T>();
        if (tc.x >= 0) avg += (a00 = *((const T*)&pIn[tc.x]));
        if (tc.y >= 0) avg += (a10 = *((const T*)&pIn[tc.y]));
        if (tc.z >= 0) avg += (a01 = *((const T*)&pIn[tc.z]));
        if (tc.w >= 0) avg += (a11 = *((const T*)&pIn[tc.w]));
        avg *= 0.33333333f;
        if (tc.x < 0) a00 = avg;
        if (tc.y < 0) a10 = avg;
        if (tc.z < 0) a01 = avg;
        if (tc.w < 0) a11 = avg;
    }
    else
    {
        a00 = (tc.x >= 0) ? *((const T*)&pIn[tc.x]) : zero_value<T>();
        a10 = (tc.y >= 0) ? *((const T*)&pIn[tc.y]) : zero_value<T>();
        a01 = (tc.z >= 0) ? *((const T*)&pIn[tc.z]) : zero_value<T>();
        a11 = (tc.w >= 0) ? *((const T*)&pIn[tc.w]) : zero_value<T>();
    }
}

static __device__ __forceinline__ void accumQuad(float4 c, float* pOut, int level, int4 tc, bool corner, CA_TEMP_PARAM)
{
    // For invalid cube map uv, tc will be all negative, and no accumulation will take place.
    if (corner)
    {
        float cb;
        if (tc.x < 0) cb = c.x;
        if (tc.y < 0) cb = c.y;
        if (tc.z < 0) cb = c.z;
        if (tc.w < 0) cb = c.w;
        cb *= 0.33333333f;
        if (tc.x >= 0) caAtomicAddTexture(pOut, level, tc.x, c.x + cb);
        if (tc.y >= 0) caAtomicAddTexture(pOut, level, tc.y, c.y + cb);
        if (tc.z >= 0) caAtomicAddTexture(pOut, level, tc.z, c.z + cb);
        if (tc.w >= 0) caAtomicAddTexture(pOut, level, tc.w, c.w + cb);
    }
    else
    {
        if (tc.x >= 0) caAtomicAddTexture(pOut, level, tc.x, c.x);
        if (tc.y >= 0) caAtomicAddTexture(pOut, level, tc.y, c.y);
        if (tc.z >= 0) caAtomicAddTexture(pOut, level, tc.z, c.z);
        if (tc.w >= 0) caAtomicAddTexture(pOut, level, tc.w, c.w);
    }
}

//------------------------------------------------------------------------
// Mip builder kernel.

template<class T, int C>
static __forceinline__ __device__ void MipBuildKernelTemplate(const TextureKernelParams p)
{
    // Sizes.
    int2 sz_in = mipLevelSize(p, p.mipLevelOut - 1);
    int2 sz_out = mipLevelSize(p, p.mipLevelOut);

    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= sz_out.x || py >= sz_out.y)
        return;

    // Pixel indices.
    int pidx_in0 = p.channels * (((px + sz_in.x * py) << 1) + (pz * sz_in.x * sz_in.y));
    int pidx_in1 = pidx_in0 + p.channels * sz_in.x; // Next pixel down.
    int pidx_out = p.channels * (px + sz_out.x * (py + sz_out.y * pz));

    // Input and output pointers.
    const float* pin = p.tex[p.mipLevelOut - 1];
    float* pout = (float*)p.tex[p.mipLevelOut];

    // Special case: Input texture height or width is 1.
    if (sz_in.x == 1 || sz_in.y == 1)
    {
        if (sz_in.y == 1)
            pidx_in1 = pidx_in0 + p.channels; // Next pixel on the right.

        for (int i=0; i < p.channels; i += C)
        {
            T v0 = *((const T*)&pin[pidx_in0 + i]);
            T v1 = *((const T*)&pin[pidx_in1 + i]);
            T avg = .5f * (v0 + v1);
#if TEX_DEBUG_MIP_RETAIN_VARIANCE
            avg = (avg - .5f) * 1.41421356f + .5f;
#endif
            *((T*)&pout[pidx_out + i]) = avg;
        }

        return;
    }

    for (int i=0; i < p.channels; i += C)
    {
        T v0 = *((const T*)&pin[pidx_in0 + i]);
        T v1 = *((const T*)&pin[pidx_in0 + i + p.channels]);
        T v2 = *((const T*)&pin[pidx_in1 + i]);
        T v3 = *((const T*)&pin[pidx_in1 + i + p.channels]);
        T avg = .25f * (v0 + v1 + v2 + v3);
#if TEX_DEBUG_MIP_RETAIN_VARIANCE
        avg = (avg - .5f) * 2.f + .5f;
#endif
        *((T*)&pout[pidx_out + i]) = avg;
    }
}

// Template specializations.
__global__ void MipBuildKernel1(const TextureKernelParams p) { MipBuildKernelTemplate<float,  1>(p); }
__global__ void MipBuildKernel2(const TextureKernelParams p) { MipBuildKernelTemplate<float2, 2>(p); }
__global__ void MipBuildKernel4(const TextureKernelParams p) { MipBuildKernelTemplate<float4, 4>(p); }

//------------------------------------------------------------------------
// Forward kernel.

template <class T, int C, bool CUBE_MODE, bool BIAS_ONLY, int FILTER_MODE>
static __forceinline__ __device__ void TextureFwdKernelTemplate(const TextureKernelParams p)
{
    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    int tz = (p.texDepth == 1) ? 0 : pz;
    if (px >= p.imgWidth || py >= p.imgHeight || pz >= p.n)
        return;

    // Pixel index.
    int pidx = px + p.imgWidth * (py + p.imgHeight * pz);

    // Output ptr.
    float* pOut = p.out + pidx * p.channels;

    // Get UV.
    float3 uv;
    if (CUBE_MODE)
        uv = ((const float3*)p.uv)[pidx];
    else
        uv = make_float3(((const float2*)p.uv)[pidx], 0.f);

    // Nearest mode.
    if (FILTER_MODE == TEX_MODE_NEAREST)
    {
        int tc = indexTextureNearest<CUBE_MODE>(p, uv, tz);
        tc *= p.channels;
        const float* pIn = p.tex[0];

        // Copy if valid tc, otherwise output zero.
        for (int i=0; i < p.channels; i += C)
            *((T*)&pOut[i]) = (tc >= 0) ? *((const T*)&pIn[tc + i]) : zero_value<T>();

        return; // Exit.
    }

    // Calculate mip level. In 'linear' mode these will all stay zero.
    float  flevel = 0.f; // Fractional level.
    int    level0 = 0;   // Discrete level 0.
    int    level1 = 0;   // Discrete level 1.
    calculateMipLevel<CUBE_MODE, BIAS_ONLY, FILTER_MODE>(level0, level1, flevel, p, pidx, uv, 0, 0);

    // Get texel indices and pointer for level 0.
    int4 tc0 = make_int4(0, 0, 0, 0);
    float2 uv0 = indexTextureLinear<CUBE_MODE>(p, uv, tz, tc0, level0);
    const float* pIn0 = p.tex[level0];
    bool corner0 = CUBE_MODE && ((tc0.x | tc0.y | tc0.z | tc0.w) < 0);
    tc0 *= p.channels;

    // Bilinear fetch.
    if (FILTER_MODE == TEX_MODE_LINEAR || FILTER_MODE == TEX_MODE_LINEAR_MIPMAP_NEAREST)
    {
        // Interpolate.
        for (int i=0; i < p.channels; i += C, tc0 += C)
        {
            T a00, a10, a01, a11;
            fetchQuad<T>(a00, a10, a01, a11, pIn0, tc0, corner0);
            *((T*)&pOut[i]) = bilerp(a00, a10, a01, a11, uv0);
        }
        return; // Exit.
    }

    // Get texel indices and pointer for level 1.
    int4 tc1 = make_int4(0, 0, 0, 0);
    float2 uv1 = indexTextureLinear<CUBE_MODE>(p, uv, tz, tc1, level1);
    const float* pIn1 = p.tex[level1];
    bool corner1 = CUBE_MODE && ((tc1.x | tc1.y | tc1.z | tc1.w) < 0);
    tc1 *= p.channels;

    // Trilinear fetch.
    for (int i=0; i < p.channels; i += C, tc0 += C, tc1 += C)
    {
        // First level.
        T a00, a10, a01, a11;
        fetchQuad<T>(a00, a10, a01, a11, pIn0, tc0, corner0);
        T a = bilerp(a00, a10, a01, a11, uv0);

        // Second level unless in magnification mode.
        if (flevel > 0.f)
        {
            T b00, b10, b01, b11;
            fetchQuad<T>(b00, b10, b01, b11, pIn1, tc1, corner1);
            T b = bilerp(b00, b10, b01, b11, uv1);
            a = lerp(a, b, flevel); // Interpolate between levels.
        }

        // Write.
        *((T*)&pOut[i]) = a;
    }
}

// Template specializations.
__global__ void TextureFwdKernelNearest1                    (const TextureKernelParams p) { TextureFwdKernelTemplate<float,  1, false, false, TEX_MODE_NEAREST>(p); }
__global__ void TextureFwdKernelNearest2                    (const TextureKernelParams p) { TextureFwdKernelTemplate<float2, 2, false, false, TEX_MODE_NEAREST>(p); }
__global__ void TextureFwdKernelNearest4                    (const TextureKernelParams p) { TextureFwdKernelTemplate<float4, 4, false, false, TEX_MODE_NEAREST>(p); }
__global__ void TextureFwdKernelLinear1                     (const TextureKernelParams p) { TextureFwdKernelTemplate<float,  1, false, false, TEX_MODE_LINEAR>(p); }
__global__ void TextureFwdKernelLinear2                     (const TextureKernelParams p) { TextureFwdKernelTemplate<float2, 2, false, false, TEX_MODE_LINEAR>(p); }
__global__ void TextureFwdKernelLinear4                     (const TextureKernelParams p) { TextureFwdKernelTemplate<float4, 4, false, false, TEX_MODE_LINEAR>(p); }
__global__ void TextureFwdKernelLinearMipmapNearest1        (const TextureKernelParams p) { TextureFwdKernelTemplate<float,  1, false, false, TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void TextureFwdKernelLinearMipmapNearest2        (const TextureKernelParams p) { TextureFwdKernelTemplate<float2, 2, false, false, TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void TextureFwdKernelLinearMipmapNearest4        (const TextureKernelParams p) { TextureFwdKernelTemplate<float4, 4, false, false, TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void TextureFwdKernelLinearMipmapLinear1         (const TextureKernelParams p) { TextureFwdKernelTemplate<float,  1, false, false, TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void TextureFwdKernelLinearMipmapLinear2         (const TextureKernelParams p) { TextureFwdKernelTemplate<float2, 2, false, false, TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void TextureFwdKernelLinearMipmapLinear4         (const TextureKernelParams p) { TextureFwdKernelTemplate<float4, 4, false, false, TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void TextureFwdKernelCubeNearest1                (const TextureKernelParams p) { TextureFwdKernelTemplate<float,  1, true,  false, TEX_MODE_NEAREST>(p); }
__global__ void TextureFwdKernelCubeNearest2                (const TextureKernelParams p) { TextureFwdKernelTemplate<float2, 2, true,  false, TEX_MODE_NEAREST>(p); }
__global__ void TextureFwdKernelCubeNearest4                (const TextureKernelParams p) { TextureFwdKernelTemplate<float4, 4, true,  false, TEX_MODE_NEAREST>(p); }
__global__ void TextureFwdKernelCubeLinear1                 (const TextureKernelParams p) { TextureFwdKernelTemplate<float,  1, true,  false, TEX_MODE_LINEAR>(p); }
__global__ void TextureFwdKernelCubeLinear2                 (const TextureKernelParams p) { TextureFwdKernelTemplate<float2, 2, true,  false, TEX_MODE_LINEAR>(p); }
__global__ void TextureFwdKernelCubeLinear4                 (const TextureKernelParams p) { TextureFwdKernelTemplate<float4, 4, true,  false, TEX_MODE_LINEAR>(p); }
__global__ void TextureFwdKernelCubeLinearMipmapNearest1    (const TextureKernelParams p) { TextureFwdKernelTemplate<float,  1, true,  false, TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void TextureFwdKernelCubeLinearMipmapNearest2    (const TextureKernelParams p) { TextureFwdKernelTemplate<float2, 2, true,  false, TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void TextureFwdKernelCubeLinearMipmapNearest4    (const TextureKernelParams p) { TextureFwdKernelTemplate<float4, 4, true,  false, TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void TextureFwdKernelCubeLinearMipmapLinear1     (const TextureKernelParams p) { TextureFwdKernelTemplate<float,  1, true,  false, TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void TextureFwdKernelCubeLinearMipmapLinear2     (const TextureKernelParams p) { TextureFwdKernelTemplate<float2, 2, true,  false, TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void TextureFwdKernelCubeLinearMipmapLinear4     (const TextureKernelParams p) { TextureFwdKernelTemplate<float4, 4, true,  false, TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void TextureFwdKernelLinearMipmapNearestBO1      (const TextureKernelParams p) { TextureFwdKernelTemplate<float,  1, false, true,  TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void TextureFwdKernelLinearMipmapNearestBO2      (const TextureKernelParams p) { TextureFwdKernelTemplate<float2, 2, false, true,  TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void TextureFwdKernelLinearMipmapNearestBO4      (const TextureKernelParams p) { TextureFwdKernelTemplate<float4, 4, false, true,  TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void TextureFwdKernelLinearMipmapLinearBO1       (const TextureKernelParams p) { TextureFwdKernelTemplate<float,  1, false, true,  TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void TextureFwdKernelLinearMipmapLinearBO2       (const TextureKernelParams p) { TextureFwdKernelTemplate<float2, 2, false, true,  TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void TextureFwdKernelLinearMipmapLinearBO4       (const TextureKernelParams p) { TextureFwdKernelTemplate<float4, 4, false, true,  TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void TextureFwdKernelCubeLinearMipmapNearestBO1  (const TextureKernelParams p) { TextureFwdKernelTemplate<float,  1, true,  true,  TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void TextureFwdKernelCubeLinearMipmapNearestBO2  (const TextureKernelParams p) { TextureFwdKernelTemplate<float2, 2, true,  true,  TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void TextureFwdKernelCubeLinearMipmapNearestBO4  (const TextureKernelParams p) { TextureFwdKernelTemplate<float4, 4, true,  true,  TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void TextureFwdKernelCubeLinearMipmapLinearBO1   (const TextureKernelParams p) { TextureFwdKernelTemplate<float,  1, true,  true,  TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void TextureFwdKernelCubeLinearMipmapLinearBO2   (const TextureKernelParams p) { TextureFwdKernelTemplate<float2, 2, true,  true,  TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void TextureFwdKernelCubeLinearMipmapLinearBO4   (const TextureKernelParams p) { TextureFwdKernelTemplate<float4, 4, true,  true,  TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }

//------------------------------------------------------------------------
// Gradient mip puller kernel.

template<class T, int C>
static __forceinline__ __device__ void MipGradKernelTemplate(const TextureKernelParams p)
{
    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.texWidth || py >= p.texHeight)
        return;

    // Number of wide elements.
    int c = p.channels;
    if (C == 2) c >>= 1;
    if (C == 4) c >>= 2;

    // Dynamically allocated shared memory for holding a texel.
    extern __shared__ float s_texelAccum[];
    int sharedOfs = threadIdx.x + threadIdx.y * blockDim.x;
    int sharedStride = blockDim.x * blockDim.y;
#   define TEXEL_ACCUM(_i) (s_texelAccum + (sharedOfs + (_i) * sharedStride))

    // Clear the texel.
    for (int i=0; i < p.channels; i++)
        *TEXEL_ACCUM(i) = 0.f;

    // Track texel position and accumulation weight over the mip stack.
    int x = px;
    int y = py;
    float w = 1.f;

    // Pull gradients from all levels.
    int2 sz = mipLevelSize(p, 0); // Previous level size.
    for (int level=1; level <= p.mipLevelMax; level++)
    {
        // Weight decay depends on previous level size.
        if (sz.x > 1) w *= .5f;
        if (sz.y > 1) w *= .5f;

        // Current level size and coordinates.
        sz = mipLevelSize(p, level);
        x >>= 1;
        y >>= 1;

        T* pIn = (T*)(p.gradTex[level] + (x + sz.x * (y + sz.y * pz)) * p.channels);
        for (int i=0; i < c; i++)
            accum_from_mem(TEXEL_ACCUM(i * C), sharedStride, pIn[i], w);
    }

    // Add to main texture gradients.
    T* pOut = (T*)(p.gradTex[0] + (px + p.texWidth * (py + p.texHeight * pz)) * p.channels);
    for (int i=0; i < c; i++)
        accum_to_mem(pOut[i], TEXEL_ACCUM(i * C), sharedStride);
}

// Template specializations.
__global__ void MipGradKernel1(const TextureKernelParams p) { MipGradKernelTemplate<float,  1>(p); }
__global__ void MipGradKernel2(const TextureKernelParams p) { MipGradKernelTemplate<float2, 2>(p); }
__global__ void MipGradKernel4(const TextureKernelParams p) { MipGradKernelTemplate<float4, 4>(p); }

//------------------------------------------------------------------------
// Gradient kernel.

template <bool CUBE_MODE, bool BIAS_ONLY, int FILTER_MODE>
static __forceinline__ __device__ void TextureGradKernelTemplate(const TextureKernelParams p)
{
    // Temporary space for coalesced atomics.
    CA_DECLARE_TEMP(TEX_GRAD_MAX_KERNEL_BLOCK_WIDTH * TEX_GRAD_MAX_KERNEL_BLOCK_HEIGHT);

    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    int tz = (p.texDepth == 1) ? 0 : pz;
    if (px >= p.imgWidth || py >= p.imgHeight || pz >= p.n)
        return;

    // Pixel index.
    int pidx = px + p.imgWidth * (py + p.imgHeight * pz);

    // Early exit if output gradients are zero.
    const float* pDy = p.dy + pidx * p.channels;
    unsigned int dmax = 0u;
    if ((p.channels & 3) == 0)
    {
        for (int i=0; i < p.channels; i += 4)
        {
            uint4 dy = *((const uint4*)&pDy[i]);
            dmax |= (dy.x | dy.y | dy.z | dy.w);
        }
    }
    else
    {
        for (int i=0; i < p.channels; i++)
            dmax |= __float_as_uint(pDy[i]);
    }

    // Store zeros and exit.
    if (__uint_as_float(dmax) == 0.f)
    {
        if (CUBE_MODE)
        {
            if (FILTER_MODE != TEX_MODE_NEAREST)
                ((float3*)p.gradUV)[pidx] = make_float3(0.f, 0.f, 0.f);
            if (FILTER_MODE == TEX_MODE_LINEAR_MIPMAP_LINEAR)
            {
                if (p.gradUVDA)
                {
                    ((float2*)p.gradUVDA)[3 * pidx + 0] = make_float2(0.f, 0.f);
                    ((float2*)p.gradUVDA)[3 * pidx + 1] = make_float2(0.f, 0.f);
                    ((float2*)p.gradUVDA)[3 * pidx + 2] = make_float2(0.f, 0.f);
                }
                if (p.gradMipLevelBias)
                    p.gradMipLevelBias[pidx] = 0.f;
            }
        }
        else
        {
            if (FILTER_MODE != TEX_MODE_NEAREST)
                ((float2*)p.gradUV)[pidx] = make_float2(0.f, 0.f);
            if (FILTER_MODE == TEX_MODE_LINEAR_MIPMAP_LINEAR)
            {
                if (p.gradUVDA)
                    ((float4*)p.gradUVDA)[pidx] = make_float4(0.f, 0.f, 0.f, 0.f);
                if (p.gradMipLevelBias)
                    p.gradMipLevelBias[pidx] = 0.f;
            }
        }
        return;
    }

    // Get UV.
    float3 uv;
    if (CUBE_MODE)
        uv = ((const float3*)p.uv)[pidx];
    else
        uv = make_float3(((const float2*)p.uv)[pidx], 0.f);

    // Nearest mode - texture gradients only.
    if (FILTER_MODE == TEX_MODE_NEAREST)
    {
        int tc = indexTextureNearest<CUBE_MODE>(p, uv, tz);
        if (tc < 0)
            return; // Outside texture.

        tc *= p.channels;
        float* pOut = p.gradTex[0];

        // Accumulate texture gradients.
        for (int i=0; i < p.channels; i++)
            caAtomicAddTexture(pOut, 0, tc + i, pDy[i]);

        return; // Exit.
    }

    // Calculate mip level. In 'linear' mode these will all stay zero.
    float4 dw = make_float4(0.f, 0.f, 0.f, 0.f);
    float3 dfdv = make_float3(0.f, 0.f, 0.f);
    float  flevel = 0.f; // Fractional level.
    int    level0 = 0;   // Discrete level 0.
    int    level1 = 0;   // Discrete level 1.
    calculateMipLevel<CUBE_MODE, BIAS_ONLY, FILTER_MODE>(level0, level1, flevel, p, pidx, uv, &dw, &dfdv);

    // UV gradient accumulators.
    float gu = 0.f;
    float gv = 0.f;

    // Get texel indices and pointers for level 0.
    int4 tc0 = make_int4(0, 0, 0, 0);
    float2 uv0 = indexTextureLinear<CUBE_MODE>(p, uv, tz, tc0, level0);
    const float* pIn0 = p.tex[level0];
    float* pOut0 = p.gradTex[level0];
    bool corner0 = CUBE_MODE && ((tc0.x | tc0.y | tc0.z | tc0.w) < 0);
    tc0 *= p.channels;

    // Texel weights.
    float uv011 = uv0.x * uv0.y;
    float uv010 = uv0.x - uv011;
    float uv001 = uv0.y - uv011;
    float uv000 = 1.f - uv0.x - uv001;
    float4 tw0 = make_float4(uv000, uv010, uv001, uv011);

    // Attribute weights.
    int2 sz0 = mipLevelSize(p, level0);
    float sclu0 = (float)sz0.x;
    float sclv0 = (float)sz0.y;

    // Bilinear mode - texture and uv gradients.
    if (FILTER_MODE == TEX_MODE_LINEAR || FILTER_MODE == TEX_MODE_LINEAR_MIPMAP_NEAREST)
    {
        for (int i=0; i < p.channels; i++, tc0 += 1)
        {
            float dy = pDy[i];
            accumQuad(tw0 * dy, pOut0, level0, tc0, corner0, CA_TEMP);

            float a00, a10, a01, a11;
            fetchQuad<float>(a00, a10, a01, a11, pIn0, tc0, corner0);
            float ad = (a11 + a00 - a10 - a01);
            gu += dy * ((a10 - a00) + uv0.y * ad) * sclu0;
            gv += dy * ((a01 - a00) + uv0.x * ad) * sclv0;
        }

        // Store UV gradients and exit.
        if (CUBE_MODE)
            ((float3*)p.gradUV)[pidx] = indexCubeMapGrad(uv, gu, gv);
        else
            ((float2*)p.gradUV)[pidx] = make_float2(gu, gv);

        return;
    }

    // Accumulate fractional mip level gradient.
    float df = 0; // dL/df.

    // Get texel indices and pointers for level 1.
    int4 tc1 = make_int4(0, 0, 0, 0);
    float2 uv1 = indexTextureLinear<CUBE_MODE>(p, uv, tz, tc1, level1);
    const float* pIn1 = p.tex[level1];
    float* pOut1 = p.gradTex[level1];
    bool corner1 = CUBE_MODE && ((tc1.x | tc1.y | tc1.z | tc1.w) < 0);
    tc1 *= p.channels;

    // Texel weights.
    float uv111 = uv1.x * uv1.y;
    float uv110 = uv1.x - uv111;
    float uv101 = uv1.y - uv111;
    float uv100 = 1.f - uv1.x - uv101;
    float4 tw1 = make_float4(uv100, uv110, uv101, uv111);

    // Attribute weights.
    int2 sz1 = mipLevelSize(p, level1);
    float sclu1 = (float)sz1.x;
    float sclv1 = (float)sz1.y;

    // Trilinear mode.
    for (int i=0; i < p.channels; i++, tc0 += 1, tc1 += 1)
    {
        float dy = pDy[i];
        float dy0 = (1.f - flevel) * dy;
        accumQuad(tw0 * dy0, pOut0, level0, tc0, corner0, CA_TEMP);

        // UV gradients for first level.
        float a00, a10, a01, a11;
        fetchQuad<float>(a00, a10, a01, a11, pIn0, tc0, corner0);
        float ad = (a11 + a00 - a10 - a01);
        gu += dy0 * ((a10 - a00) + uv0.y * ad) * sclu0;
        gv += dy0 * ((a01 - a00) + uv0.x * ad) * sclv0;

        // Second level unless in magnification mode.
        if (flevel > 0.f)
        {
            // Texture gradients for second level.
            float dy1 = flevel * dy;
            accumQuad(tw1 * dy1, pOut1, level1, tc1, corner1, CA_TEMP);

            // UV gradients for second level.
            float b00, b10, b01, b11;
            fetchQuad<float>(b00, b10, b01, b11, pIn1, tc1, corner1);
            float bd = (b11 + b00 - b10 - b01);
            gu += dy1 * ((b10 - b00) + uv1.y * bd) * sclu1;
            gv += dy1 * ((b01 - b00) + uv1.x * bd) * sclv1;

            // Mip level gradient.
            float a = bilerp(a00, a10, a01, a11, uv0);
            float b = bilerp(b00, b10, b01, b11, uv1);
            df += (b-a) * dy;
        }
    }

    // Store UV gradients.
    if (CUBE_MODE)
        ((float3*)p.gradUV)[pidx] = indexCubeMapGrad(uv, gu, gv) + (dfdv * df);
    else
        ((float2*)p.gradUV)[pidx] = make_float2(gu, gv);

    // Store mip level bias gradient.
    if (p.gradMipLevelBias)
        p.gradMipLevelBias[pidx] = df;

    // Store UV pixel differential gradients.
    if (!BIAS_ONLY)
    {
        // Final gradients.
        dw *= df; // dL/(d{s,y}/d{X,Y}) = df/(d{s,y}/d{X,Y}) * dL/df.

        // Store them.
        if (CUBE_MODE)
        {
            // Remap from dL/(d{s,t}/s{X,Y}) to dL/(d{x,y,z}/d{X,Y}).
            float3 g0, g1;
            indexCubeMapGrad4(uv, dw, g0, g1);
            ((float2*)p.gradUVDA)[3 * pidx + 0] = make_float2(g0.x, g1.x);
            ((float2*)p.gradUVDA)[3 * pidx + 1] = make_float2(g0.y, g1.y);
            ((float2*)p.gradUVDA)[3 * pidx + 2] = make_float2(g0.z, g1.z);
        }
        else
            ((float4*)p.gradUVDA)[pidx] = dw;
    }
}

// Template specializations.
__global__ void TextureGradKernelNearest                    (const TextureKernelParams p) { TextureGradKernelTemplate<false, false, TEX_MODE_NEAREST>(p); }
__global__ void TextureGradKernelLinear                     (const TextureKernelParams p) { TextureGradKernelTemplate<false, false, TEX_MODE_LINEAR>(p); }
__global__ void TextureGradKernelLinearMipmapNearest        (const TextureKernelParams p) { TextureGradKernelTemplate<false, false, TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void TextureGradKernelLinearMipmapLinear         (const TextureKernelParams p) { TextureGradKernelTemplate<false, false, TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void TextureGradKernelCubeNearest                (const TextureKernelParams p) { TextureGradKernelTemplate<true,  false, TEX_MODE_NEAREST>(p); }
__global__ void TextureGradKernelCubeLinear                 (const TextureKernelParams p) { TextureGradKernelTemplate<true,  false, TEX_MODE_LINEAR>(p); }
__global__ void TextureGradKernelCubeLinearMipmapNearest    (const TextureKernelParams p) { TextureGradKernelTemplate<true,  false, TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void TextureGradKernelCubeLinearMipmapLinear     (const TextureKernelParams p) { TextureGradKernelTemplate<true,  false, TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void TextureGradKernelLinearMipmapNearestBO      (const TextureKernelParams p) { TextureGradKernelTemplate<false, true,  TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void TextureGradKernelLinearMipmapLinearBO       (const TextureKernelParams p) { TextureGradKernelTemplate<false, true,  TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void TextureGradKernelCubeLinearMipmapNearestBO  (const TextureKernelParams p) { TextureGradKernelTemplate<true,  true,  TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void TextureGradKernelCubeLinearMipmapLinearBO   (const TextureKernelParams p) { TextureGradKernelTemplate<true,  true,  TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }

//------------------------------------------------------------------------
