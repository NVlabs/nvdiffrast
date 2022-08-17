// Copyright (c) 2009-2022, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "PrivateDefs.hpp"

namespace CR
{
//------------------------------------------------------------------------

template<class T> __device__ __inline__ void swap(T& a, T& b)               { T t = a; a = b; b = t; }

__device__ __inline__ U32   getLo                   (U64 a)                 { return __double2loint(__longlong_as_double(a)); }
__device__ __inline__ S32   getLo                   (S64 a)                 { return __double2loint(__longlong_as_double(a)); }
__device__ __inline__ U32   getHi                   (U64 a)                 { return __double2hiint(__longlong_as_double(a)); }
__device__ __inline__ S32   getHi                   (S64 a)                 { return __double2hiint(__longlong_as_double(a)); }
__device__ __inline__ U64   combineLoHi             (U32 lo, U32 hi)        { return __double_as_longlong(__hiloint2double(hi, lo)); }
__device__ __inline__ S64   combineLoHi             (S32 lo, S32 hi)        { return __double_as_longlong(__hiloint2double(hi, lo)); }
__device__ __inline__ U32   getLaneMaskLt           (void)                  { U32 r; asm("mov.u32 %0, %lanemask_lt;" : "=r"(r)); return r; }
__device__ __inline__ U32   getLaneMaskLe           (void)                  { U32 r; asm("mov.u32 %0, %lanemask_le;" : "=r"(r)); return r; }
__device__ __inline__ U32   getLaneMaskGt           (void)                  { U32 r; asm("mov.u32 %0, %lanemask_gt;" : "=r"(r)); return r; }
__device__ __inline__ U32   getLaneMaskGe           (void)                  { U32 r; asm("mov.u32 %0, %lanemask_ge;" : "=r"(r)); return r; }
__device__ __inline__ int   findLeadingOne          (U32 v)                 { U32 r; asm("bfind.u32 %0, %1;" : "=r"(r) : "r"(v)); return r; }
__device__ __inline__ bool  singleLane              (void)                  { return ((::__ballot_sync(~0u, true) & getLaneMaskLt()) == 0); }

__device__ __inline__ void  add_add_carry           (U32& rlo, U32 alo, U32 blo, U32& rhi, U32 ahi, U32 bhi) { U64 r = combineLoHi(alo, ahi) + combineLoHi(blo, bhi); rlo = getLo(r); rhi = getHi(r); }
__device__ __inline__ S32   f32_to_s32_sat          (F32 a)                 { S32 v; asm("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(v) : "f"(a)); return v; }
__device__ __inline__ U32   f32_to_u32_sat          (F32 a)                 { U32 v; asm("cvt.rni.sat.u32.f32 %0, %1;" : "=r"(v) : "f"(a)); return v; }
__device__ __inline__ U32   f32_to_u32_sat_rmi      (F32 a)                 { U32 v; asm("cvt.rmi.sat.u32.f32 %0, %1;" : "=r"(v) : "f"(a)); return v; }
__device__ __inline__ U32   f32_to_u8_sat           (F32 a)                 { U32 v; asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(v) : "f"(a)); return v; }
__device__ __inline__ S64   f32_to_s64              (F32 a)                 { S64 v; asm("cvt.rni.s64.f32 %0, %1;" : "=l"(v) : "f"(a)); return v; }
__device__ __inline__ S32   add_s16lo_s16lo			(S32 a, S32 b)			{ S32 v; asm("vadd.s32.s32.s32 %0, %1.h0, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   add_s16hi_s16lo			(S32 a, S32 b)			{ S32 v; asm("vadd.s32.s32.s32 %0, %1.h1, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   add_s16lo_s16hi			(S32 a, S32 b)			{ S32 v; asm("vadd.s32.s32.s32 %0, %1.h0, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   add_s16hi_s16hi			(S32 a, S32 b)			{ S32 v; asm("vadd.s32.s32.s32 %0, %1.h1, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_s16lo_s16lo			(S32 a, S32 b)			{ S32 v; asm("vsub.s32.s32.s32 %0, %1.h0, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_s16hi_s16lo			(S32 a, S32 b)			{ S32 v; asm("vsub.s32.s32.s32 %0, %1.h1, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_s16lo_s16hi			(S32 a, S32 b)			{ S32 v; asm("vsub.s32.s32.s32 %0, %1.h0, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_s16hi_s16hi			(S32 a, S32 b)			{ S32 v; asm("vsub.s32.s32.s32 %0, %1.h1, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_u16lo_u16lo			(U32 a, U32 b)			{ S32 v; asm("vsub.s32.u32.u32 %0, %1.h0, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_u16hi_u16lo			(U32 a, U32 b)			{ S32 v; asm("vsub.s32.u32.u32 %0, %1.h1, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_u16lo_u16hi			(U32 a, U32 b)			{ S32 v; asm("vsub.s32.u32.u32 %0, %1.h0, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_u16hi_u16hi			(U32 a, U32 b)			{ S32 v; asm("vsub.s32.u32.u32 %0, %1.h1, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ U32   add_b0					(U32 a, U32 b)			{ U32 v; asm("vadd.u32.u32.u32 %0, %1.b0, %2;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ U32   add_b1					(U32 a, U32 b)			{ U32 v; asm("vadd.u32.u32.u32 %0, %1.b1, %2;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ U32   add_b2					(U32 a, U32 b)			{ U32 v; asm("vadd.u32.u32.u32 %0, %1.b2, %2;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ U32   add_b3					(U32 a, U32 b)			{ U32 v; asm("vadd.u32.u32.u32 %0, %1.b3, %2;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ U32   vmad_b0					(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b0, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b1					(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b2					(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b2, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b3					(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b3, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b0_b3				(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b0, %2.b3, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b1_b3				(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b1, %2.b3, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b2_b3				(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b2, %2.b3, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b3_b3				(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b3, %2.b3, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   add_mask8				(U32 a, U32 b)			{ U32 v; U32 z=0; asm("vadd.u32.u32.u32 %0.b0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(z)); return v; }
__device__ __inline__ U32   sub_mask8				(U32 a, U32 b)			{ U32 v; U32 z=0; asm("vsub.u32.u32.u32 %0.b0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(z)); return v; }
__device__ __inline__ S32   max_max					(S32 a, S32 b, S32 c)	{ S32 v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   min_min					(S32 a, S32 b, S32 c)	{ S32 v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   max_add					(S32 a, S32 b, S32 c)	{ S32 v; asm("vmax.s32.s32.s32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   min_add					(S32 a, S32 b, S32 c)	{ S32 v; asm("vmin.s32.s32.s32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   add_add					(U32 a, U32 b, U32 c)	{ U32 v; asm("vadd.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   sub_add					(U32 a, U32 b, U32 c)	{ U32 v; asm("vsub.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   add_sub					(U32 a, U32 b, U32 c)	{ U32 v; asm("vsub.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(c), "r"(b)); return v; }
__device__ __inline__ S32   add_clamp_0_x			(S32 a, S32 b, S32 c)	{ S32 v; asm("vadd.u32.s32.s32.sat.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   add_clamp_b0			(S32 a, S32 b, S32 c)	{ S32 v; asm("vadd.u32.s32.s32.sat %0.b0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   add_clamp_b2			(S32 a, S32 b, S32 c)	{ S32 v; asm("vadd.u32.s32.s32.sat %0.b2, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   prmt					(U32 a, U32 b, U32 c)   { U32 v; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   u32lo_sext              (U32 a)                 { U32 v; asm("cvt.s16.u32 %0, %1;" : "=r"(v) : "r"(a)); return v; }
__device__ __inline__ U32   slct                    (U32 a, U32 b, S32 c)   { U32 v; asm("slct.u32.s32 %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   slct                    (S32 a, S32 b, S32 c)   { S32 v; asm("slct.s32.s32 %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ F32   slct                    (F32 a, F32 b, S32 c)   { F32 v; asm("slct.f32.s32 %0, %1, %2, %3;" : "=f"(v) : "f"(a), "f"(b), "r"(c)); return v; }
__device__ __inline__ U32   isetge                  (S32 a, S32 b)          { U32 v; asm("set.ge.u32.s32 %0, %1, %2;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ F64   rcp_approx              (F64 a)                 { F64 v; asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(v) : "d"(a)); return v; }
__device__ __inline__ F32   fma_rm                  (F32 a, F32 b, F32 c)   { F32 v; asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(v) : "f"(a), "f"(b), "f"(c)); return v; }
__device__ __inline__ U32   idiv_fast               (U32 a, U32 b);

__device__ __inline__ uint3 setupPleq               (float3 values, int2 v0, int2 d1, int2 d2, F32 areaRcp);

__device__ __inline__ void  cover8x8_setupLUT           (volatile U64* lut);
__device__ __inline__ U64   cover8x8_exact_fast         (S32 ox, S32 oy, S32 dx, S32 dy, U32 flips, volatile const U64* lut); // Assumes viewport <= 2^11, subpixels <= 2^4, no guardband.
__device__ __inline__ U64   cover8x8_lookupMask         (S64 yinit, U32 yinc, U32 flips, volatile const U64* lut);

__device__ __inline__ U64   cover8x8_exact_noLUT        (S32 ox, S32 oy, S32 dx, S32 dy); // optimized reference implementation, does not require look-up table
__device__ __inline__ U64   cover8x8_conservative_noLUT (S32 ox, S32 oy, S32 dx, S32 dy);
__device__ __inline__ U64   cover8x8_generateMask_noLUT (S32 curr, S32 dx, S32 dy);

template <class T> __device__ __inline__ void sortShared(T* ptr, int numItems); // Assumes that numItems <= threadsInBlock. Must sync before & after the call.

__device__ __inline__ const CRImageParams& getImageParams(const CRParams& p, int idx)
{
    return (idx < CR_EMBED_IMAGE_PARAMS) ? p.imageParamsFirst[idx] : p.imageParamsExtra[idx - CR_EMBED_IMAGE_PARAMS];
}

//------------------------------------------------------------------------

__device__ __inline__ int clipPolygonWithPlane(F32* baryOut, const F32* baryIn, int numIn, F32 v0, F32 v1, F32 v2)
{
    int numOut = 0;
    if (numIn >= 3)
    {
        int ai = (numIn - 1) * 2;
        F32 av = v0 + v1 * baryIn[ai + 0] + v2 * baryIn[ai + 1];
        for (int bi = 0; bi < numIn * 2; bi += 2)
        {
            F32 bv = v0 + v1 * baryIn[bi + 0] + v2 * baryIn[bi + 1];
            if (av * bv < 0.0f)
            {
                F32 bc = av / (av - bv);
                F32 ac = 1.0f - bc;
                baryOut[numOut + 0] = baryIn[ai + 0] * ac + baryIn[bi + 0] * bc;
                baryOut[numOut + 1] = baryIn[ai + 1] * ac + baryIn[bi + 1] * bc;
                numOut += 2;
            }
            if (bv >= 0.0f)
            {
                baryOut[numOut + 0] = baryIn[bi + 0];
                baryOut[numOut + 1] = baryIn[bi + 1];
                numOut += 2;
            }
            ai = bi;
            av = bv;
        }
    }
    return (numOut >> 1);
}

//------------------------------------------------------------------------

__device__ __inline__ int clipTriangleWithFrustum(F32* bary, const F32* v0, const F32* v1, const F32* v2, const F32* d1, const F32* d2)
{
    int num = 3;
    bary[0] = 0.0f, bary[1] = 0.0f;
    bary[2] = 1.0f, bary[3] = 0.0f;
    bary[4] = 0.0f, bary[5] = 1.0f;

    if ((v0[3] < fabsf(v0[0])) | (v1[3] < fabsf(v1[0])) | (v2[3] < fabsf(v2[0])))
    {
        F32 temp[18];
        num = clipPolygonWithPlane(temp, bary, num, v0[3] + v0[0], d1[3] + d1[0], d2[3] + d2[0]);
        num = clipPolygonWithPlane(bary, temp, num, v0[3] - v0[0], d1[3] - d1[0], d2[3] - d2[0]);
    }
    if ((v0[3] < fabsf(v0[1])) | (v1[3] < fabsf(v1[1])) | (v2[3] < fabsf(v2[1])))
    {
        F32 temp[18];
        num = clipPolygonWithPlane(temp, bary, num, v0[3] + v0[1], d1[3] + d1[1], d2[3] + d2[1]);
        num = clipPolygonWithPlane(bary, temp, num, v0[3] - v0[1], d1[3] - d1[1], d2[3] - d2[1]);
    }
    if ((v0[3] < fabsf(v0[2])) | (v1[3] < fabsf(v1[2])) | (v2[3] < fabsf(v2[2])))
    {
        F32 temp[18];
        num = clipPolygonWithPlane(temp, bary, num, v0[3] + v0[2], d1[3] + d1[2], d2[3] + d2[2]);
        num = clipPolygonWithPlane(bary, temp, num, v0[3] - v0[2], d1[3] - d1[2], d2[3] - d2[2]);
    }
    return num;
}

//------------------------------------------------------------------------

__device__ __inline__ U32 idiv_fast(U32 a, U32 b)
{
    return f32_to_u32_sat_rmi(((F32)a + 0.5f) / (F32)b);
}

//------------------------------------------------------------------------

__device__ __inline__ U32 toABGR(float4 color)
{
	// 11 instructions: 4*FFMA, 4*F2I, 3*PRMT
	U32 x = f32_to_u32_sat_rmi(fma_rm(color.x, (1 << 24) * 255.0f, (1 << 24) * 0.5f));
	U32 y = f32_to_u32_sat_rmi(fma_rm(color.y, (1 << 24) * 255.0f, (1 << 24) * 0.5f));
	U32 z = f32_to_u32_sat_rmi(fma_rm(color.z, (1 << 24) * 255.0f, (1 << 24) * 0.5f));
	U32 w = f32_to_u32_sat_rmi(fma_rm(color.w, (1 << 24) * 255.0f, (1 << 24) * 0.5f));
	return prmt(prmt(x, y, 0x0073), prmt(z, w, 0x0073), 0x5410);
}

//------------------------------------------------------------------------
// v0 = subpixels relative to the bottom-left sampling point

__device__ __inline__ uint3 setupPleq(float3 values, int2 v0, int2 d1, int2 d2, F32 areaRcp)
{
    F32 mx = fmaxf(fmaxf(values.x, values.y), values.z);
    int sh = ::min(::max((__float_as_int(mx) >> 23) - (127 + 22), 0), 8);
    S32 t0 = (U32)values.x >> sh;
    S32 t1 = ((U32)values.y >> sh) - t0;
    S32 t2 = ((U32)values.z >> sh) - t0;

    U32 rcpMant = (__float_as_int(areaRcp) & 0x007FFFFF) | 0x00800000;
    int rcpShift = (23 + 127) - (__float_as_int(areaRcp) >> 23);

    uint3 pleq;
    S64 xc = ((S64)t1 * d2.y - (S64)t2 * d1.y) * rcpMant;
    S64 yc = ((S64)t2 * d1.x - (S64)t1 * d2.x) * rcpMant;
    pleq.x = (U32)(xc >> (rcpShift - (sh + CR_SUBPIXEL_LOG2)));
    pleq.y = (U32)(yc >> (rcpShift - (sh + CR_SUBPIXEL_LOG2)));

    S32 centerX = (v0.x * 2 + min_min(d1.x, d2.x, 0) + max_max(d1.x, d2.x, 0)) >> (CR_SUBPIXEL_LOG2 + 1);
    S32 centerY = (v0.y * 2 + min_min(d1.y, d2.y, 0) + max_max(d1.y, d2.y, 0)) >> (CR_SUBPIXEL_LOG2 + 1);
    S32 vcx = v0.x - (centerX << CR_SUBPIXEL_LOG2);
    S32 vcy = v0.y - (centerY << CR_SUBPIXEL_LOG2);

    pleq.z = t0 << sh;
    pleq.z -= (U32)(((xc >> 13) * vcx + (yc >> 13) * vcy) >> (rcpShift - (sh + 13)));
    pleq.z -= pleq.x * centerX + pleq.y * centerY;
    return pleq;
}

//------------------------------------------------------------------------

__device__ __inline__ void cover8x8_setupLUT(volatile U64* lut)
{
    for (S32 lutIdx = threadIdx.x + blockDim.x * threadIdx.y; lutIdx < CR_COVER8X8_LUT_SIZE; lutIdx += blockDim.x * blockDim.y)
    {
        int half       = (lutIdx < (12 << 5)) ? 0 : 1;
        int yint       = (lutIdx >> 5) - half * 12 - 3;
        U32 shape      = ((lutIdx >> 2) & 7) << (31 - 2);
        S32 slctSwapXY = lutIdx << (31 - 1);
        S32 slctNegX   = lutIdx << (31 - 0);
        S32 slctCompl  = slctSwapXY ^ slctNegX;

        U64 mask = 0;
        int xlo = half * 4;
        int xhi = xlo + 4;
        for (int x = xlo; x < xhi; x++)
        {
            int ylo = slct(0, ::max(yint, 0), slctCompl);
            int yhi = slct(::min(yint, 8), 8, slctCompl);
            for (int y = ylo; y < yhi; y++)
            {
                int xx = slct(x, y, slctSwapXY);
                int yy = slct(y, x, slctSwapXY);
                xx = slct(xx, 7 - xx, slctNegX);
                mask |= (U64)1 << (xx + yy * 8);
            }
            yint += shape >> 31;
            shape <<= 1;
        }
        lut[lutIdx] = mask;
    }
}

//------------------------------------------------------------------------

__device__ __inline__ U64 cover8x8_exact_fast(S32 ox, S32 oy, S32 dx, S32 dy, U32 flips, volatile const U64* lut) // 52 instr
{
    F32  yinitBias  = (F32)(1 << (31 - CR_MAXVIEWPORT_LOG2 - CR_SUBPIXEL_LOG2 * 2));
    F32  yinitScale = (F32)(1 << (32 - CR_SUBPIXEL_LOG2));
    F32  yincScale  = 65536.0f * 65536.0f;

    S32  slctFlipY  = flips << (31 - CR_FLIPBIT_FLIP_Y);
    S32  slctFlipX  = flips << (31 - CR_FLIPBIT_FLIP_X);
    S32  slctSwapXY = flips << (31 - CR_FLIPBIT_SWAP_XY);

    // Evaluate cross product.

    S32 t = ox * dy - oy * dx;
    F32 det = (F32)slct(t, t - dy * (7 << CR_SUBPIXEL_LOG2), slctFlipX);
    if (flips >= (1 << CR_FLIPBIT_COMPL))
        det = -det;

    // Represent Y as a function of X.

    F32 xrcp  = 1.0f / (F32)::abs(slct(dx, dy, slctSwapXY));
    F32 yzero = det * yinitScale * xrcp + yinitBias;
    S64 yinit = f32_to_s64(slct(yzero, -yzero, slctFlipY));
    U32 yinc  = f32_to_u32_sat((F32)::abs(slct(dy, dx, slctSwapXY)) * xrcp * yincScale);

    // Lookup.

    return cover8x8_lookupMask(yinit, yinc, flips, lut);
}

//------------------------------------------------------------------------

__device__ __inline__ U64 cover8x8_lookupMask(S64 yinit, U32 yinc, U32 flips, volatile const U64* lut)
{
    // First half.

    U32 yfrac = getLo(yinit);
    U32 shape = add_clamp_0_x(getHi(yinit) + 4, 0, 11);
    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    int oct = flips & ((1 << CR_FLIPBIT_FLIP_X) | (1 << CR_FLIPBIT_SWAP_XY));
    U64 mask = *(U64*)((U8*)lut + oct + (shape << 5));

    // Second half.

    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    shape = add_clamp_0_x(getHi(yinit) + 4, __popc(shape & 15), 11);
    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    mask |= *(U64*)((U8*)lut + oct + (shape << 5) + (12 << 8));
    return (flips >= (1 << CR_FLIPBIT_COMPL)) ? ~mask : mask;
}

//------------------------------------------------------------------------

__device__ __inline__ U64 cover8x8_exact_noLUT(S32 ox, S32 oy, S32 dx, S32 dy)
{
    S32 curr = ox * dy - oy * dx;
    if (dy > 0 || (dy == 0 && dx <= 0)) curr--; // exclusive
    return cover8x8_generateMask_noLUT(curr, dx, dy);
}

//------------------------------------------------------------------------

__device__ __inline__ U64 cover8x8_conservative_noLUT(S32 ox, S32 oy, S32 dx, S32 dy)
{
    S32 curr = ox * dy - oy * dx;
    if (dy > 0 || (dy == 0 && dx <= 0)) curr--; // exclusive
    curr += (::abs(dx) + ::abs(dy)) << (CR_SUBPIXEL_LOG2 - 1);
    return cover8x8_generateMask_noLUT(curr, dx, dy);
}

//------------------------------------------------------------------------

__device__ __inline__ U64 cover8x8_generateMask_noLUT(S32 curr, S32 dx, S32 dy)
{
    curr += (dx - dy) * (7 << CR_SUBPIXEL_LOG2);
    S32 stepX = dy << (CR_SUBPIXEL_LOG2 + 1);
    S32 stepYorig = -dx - dy * 7;
    S32 stepY = stepYorig << (CR_SUBPIXEL_LOG2 + 1);

    U32 hi = isetge(curr, 0);
    U32 frac = curr + curr;
    for (int i = 62; i >= 32; i--)
        add_add_carry(frac, frac, ((i & 7) == 7) ? stepY : stepX, hi, hi, hi);

	U32 lo = 0;
    for (int i = 31; i >= 0; i--)
        add_add_carry(frac, frac, ((i & 7) == 7) ? stepY : stepX, lo, lo, lo);

	lo ^= lo >> 1,  hi ^= hi >> 1;
	lo ^= lo >> 2,  hi ^= hi >> 2;
	lo ^= lo >> 4,  hi ^= hi >> 4;
	lo ^= lo >> 8,  hi ^= hi >> 8;
	lo ^= lo >> 16, hi ^= hi >> 16;

	if (dy < 0)
    {
        lo ^= 0x55AA55AA;
        hi ^= 0x55AA55AA;
    }
	if (stepYorig < 0)
    {
        lo ^= 0xFF00FF00;
        hi ^= 0x00FF00FF;
    }
	if ((hi & 1) != 0)
		lo = ~lo;

    return combineLoHi(lo, hi);
}

//------------------------------------------------------------------------

template <class T> __device__ __inline__ void sortShared(T* ptr, int numItems)
{
    int thrInBlock = threadIdx.x + threadIdx.y * blockDim.x;
    int range = 16;

    // Use transposition sort within each 16-wide subrange.

    int base = thrInBlock * 2;
    bool act = (base < numItems - 1);
    U32 actMask = __ballot_sync(~0u, act);
    if (act)
    {
        bool tryOdd = (base < numItems - 2 && (~base & (range - 2)) != 0);
        T mid = ptr[base + 1];

        for (int iter = 0; iter < range; iter += 2)
        {
            // Evens.

            T tmp = ptr[base + 0];
            if (tmp > mid)
            {
                ptr[base + 0] = mid;
                mid = tmp;
            }
            __syncwarp(actMask);

            // Odds.

            if (tryOdd)
            {
                tmp = ptr[base + 2];
                if (mid > tmp)
                {
                    ptr[base + 2] = mid;
                    mid = tmp;
                }
            }
            __syncwarp(actMask);
        }
        ptr[base + 1] = mid;
    }

    // Multiple subranges => Merge hierarchically.

    for (; range < numItems; range <<= 1)
    {
        // Assuming that we would insert the current item into the other
        // subrange, use binary search to find the appropriate slot.

        __syncthreads();

        T item;
        int slot;
        if (thrInBlock < numItems)
        {
            item = ptr[thrInBlock];
            slot = (thrInBlock & -range) ^ range;
            if (slot < numItems)
            {
                T tmp = ptr[slot];
                bool inclusive = ((thrInBlock & range) != 0);
                if (tmp < item || (inclusive && tmp == item))
                {
                    for (int step = (range >> 1); step != 0; step >>= 1)
                    {
                        int probe = slot + step;
                        if (probe < numItems)
                        {
                            tmp = ptr[probe];
                            if (tmp < item || (inclusive && tmp == item))
                                slot = probe;
                        }
                    }
                    slot++;
                }
            }
        }

        // Store the item at an appropriate place.

        __syncthreads();

        if (thrInBlock < numItems)
            ptr[slot + (thrInBlock & (range * 2 - 1)) - range] = item;
    }
}

//------------------------------------------------------------------------
}
