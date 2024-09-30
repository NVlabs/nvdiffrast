// Copyright (c) 2009-2022, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

//------------------------------------------------------------------------

__device__ __inline__ void snapTriangle(
    const CRParams& p,
    float4 v0, float4 v1, float4 v2,
    int2& p0, int2& p1, int2& p2, float3& rcpW, int2& lo, int2& hi)
{
    F32 viewScaleX = (F32)(p.widthPixelsVp  << (CR_SUBPIXEL_LOG2 - 1));
    F32 viewScaleY = (F32)(p.heightPixelsVp << (CR_SUBPIXEL_LOG2 - 1));
    rcpW = make_float3(1.0f / v0.w, 1.0f / v1.w, 1.0f / v2.w);
    p0 = make_int2(f32_to_s32_sat(v0.x * rcpW.x * viewScaleX), f32_to_s32_sat(v0.y * rcpW.x * viewScaleY));
    p1 = make_int2(f32_to_s32_sat(v1.x * rcpW.y * viewScaleX), f32_to_s32_sat(v1.y * rcpW.y * viewScaleY));
    p2 = make_int2(f32_to_s32_sat(v2.x * rcpW.z * viewScaleX), f32_to_s32_sat(v2.y * rcpW.z * viewScaleY));
    lo = make_int2(min_min(p0.x, p1.x, p2.x), min_min(p0.y, p1.y, p2.y));
    hi = make_int2(max_max(p0.x, p1.x, p2.x), max_max(p0.y, p1.y, p2.y));
}

//------------------------------------------------------------------------

__device__ __inline__ U32 cover8x8_selectFlips(S32 dx, S32 dy) // 10 instr
{
    U32 flips = 0;
    if (dy > 0 || (dy == 0 && dx <= 0))
        flips ^= (1 << CR_FLIPBIT_FLIP_X) ^ (1 << CR_FLIPBIT_FLIP_Y) ^ (1 << CR_FLIPBIT_COMPL);
    if (dx > 0)
        flips ^= (1 << CR_FLIPBIT_FLIP_X) ^ (1 << CR_FLIPBIT_FLIP_Y);
    if (::abs(dx) < ::abs(dy))
        flips ^= (1 << CR_FLIPBIT_SWAP_XY) ^ (1 << CR_FLIPBIT_FLIP_Y);
    return flips;
}

//------------------------------------------------------------------------

__device__ __inline__ bool prepareTriangle(
    const CRParams& p,
    int2 p0, int2 p1, int2 p2, int2 lo, int2 hi,
    int2& d1, int2& d2, S32& area)
{
    // Backfacing or degenerate => cull.

    d1 = make_int2(p1.x - p0.x, p1.y - p0.y);
    d2 = make_int2(p2.x - p0.x, p2.y - p0.y);
    area = d1.x * d2.y - d1.y * d2.x;

    if (area == 0)
        return false; // Degenerate.

    if (area < 0 && (p.renderModeFlags & CudaRaster::RenderModeFlag_EnableBackfaceCulling) != 0)
        return false; // Backfacing.

    // AABB falls between samples => cull.

    int sampleSize = 1 << CR_SUBPIXEL_LOG2;
    int biasX = (p.widthPixelsVp  << (CR_SUBPIXEL_LOG2 - 1)) - (sampleSize >> 1);
    int biasY = (p.heightPixelsVp << (CR_SUBPIXEL_LOG2 - 1)) - (sampleSize >> 1);
    int lox = (int)add_add(lo.x, sampleSize - 1, biasX) & -sampleSize;
    int loy = (int)add_add(lo.y, sampleSize - 1, biasY) & -sampleSize;
    int hix = (hi.x + biasX) & -sampleSize;
    int hiy = (hi.y + biasY) & -sampleSize;

    if (lox > hix || loy > hiy)
        return false; // Between pixels.

    // AABB covers 1 or 2 samples => cull if they are not covered.

    int diff = add_sub(hix, hiy, lox) - loy;
    if (diff <= sampleSize)
    {
        int2 t0 = make_int2(add_sub(p0.x, biasX, lox), add_sub(p0.y, biasY, loy));
        int2 t1 = make_int2(add_sub(p1.x, biasX, lox), add_sub(p1.y, biasY, loy));
        int2 t2 = make_int2(add_sub(p2.x, biasX, lox), add_sub(p2.y, biasY, loy));
        S32 e0 = t0.x * t1.y - t0.y * t1.x;
        S32 e1 = t1.x * t2.y - t1.y * t2.x;
        S32 e2 = t2.x * t0.y - t2.y * t0.x;
        if (area < 0)
        {
            e0 = -e0;
            e1 = -e1;
            e2 = -e2;
        }

        if (e0 < 0 || e1 < 0 || e2 < 0)
        {
            if (diff == 0)
                return false; // Between pixels.

            t0 = make_int2(add_sub(p0.x, biasX, hix), add_sub(p0.y, biasY, hiy));
            t1 = make_int2(add_sub(p1.x, biasX, hix), add_sub(p1.y, biasY, hiy));
            t2 = make_int2(add_sub(p2.x, biasX, hix), add_sub(p2.y, biasY, hiy));
            e0 = t0.x * t1.y - t0.y * t1.x;
            e1 = t1.x * t2.y - t1.y * t2.x;
            e2 = t2.x * t0.y - t2.y * t0.x;
            if (area < 0)
            {
                e0 = -e0;
                e1 = -e1;
                e2 = -e2;
            }

            if (e0 < 0 || e1 < 0 || e2 < 0)
                return false; // Between pixels.
        }
    }

    // Otherwise => proceed to output the triangle.

    return true; // Visible.
}

//------------------------------------------------------------------------

__device__ __inline__ void setupTriangle(
    const CRParams& p,
    CRTriangleHeader* th, CRTriangleData* td, int triId,
    float v0z, float v1z, float v2z,
    int2 p0, int2 p1, int2 p2, float3 rcpW,
    int2 d1, int2 d2, S32 area)
{
    // Swap vertices 1 and 2 if area is negative. Only executed if backface culling is
    // disabled (if it is enabled, we never come here with area < 0).

    if (area < 0)
    {
        swap(d1, d2);
        swap(p1, p2);
        swap(v1z, v2z);
        swap(rcpW.y, rcpW.z);
        area = -area;
    }

    int2 wv0;
    wv0.x = p0.x + (p.widthPixelsVp  << (CR_SUBPIXEL_LOG2 - 1));
    wv0.y = p0.y + (p.heightPixelsVp << (CR_SUBPIXEL_LOG2 - 1));

    // Setup depth plane equation.

    F32 zcoef = (F32)(CR_DEPTH_MAX - CR_DEPTH_MIN) * 0.5f;
    F32 zbias = (F32)(CR_DEPTH_MAX + CR_DEPTH_MIN) * 0.5f;
    float3 zvert = make_float3(
        (v0z * zcoef) * rcpW.x + zbias,
        (v1z * zcoef) * rcpW.y + zbias,
        (v2z * zcoef) * rcpW.z + zbias
    );
    int2 zv0 = make_int2(
        wv0.x - (1 << (CR_SUBPIXEL_LOG2 - 1)),
        wv0.y - (1 << (CR_SUBPIXEL_LOG2 - 1))
    );
    uint3 zpleq = setupPleq(zvert, zv0, d1, d2, 1.0f / (F32)area);

    U32 zmin = f32_to_u32_sat(fminf(fminf(zvert.x, zvert.y), zvert.z) - (F32)CR_LERP_ERROR(0));

    // Write CRTriangleData.

    *(uint4*)td = make_uint4(zpleq.x, zpleq.y, zpleq.z, triId);

    // Determine flipbits.

    U32 f01 = cover8x8_selectFlips(d1.x, d1.y);
    U32 f12 = cover8x8_selectFlips(d2.x - d1.x, d2.y - d1.y);
    U32 f20 = cover8x8_selectFlips(-d2.x, -d2.y);

    // Write CRTriangleHeader.

    *(uint4*)th = make_uint4(
        prmt(p0.x, p0.y, 0x5410),
        prmt(p1.x, p1.y, 0x5410),
        prmt(p2.x, p2.y, 0x5410),
        (zmin & 0xfffff000u) | (f01 << 6) | (f12 << 2) | (f20 >> 2));
}

//------------------------------------------------------------------------

__device__ __inline__ void triangleSetupImpl(const CRParams p)
{
    __shared__ F32 s_bary[CR_SETUP_WARPS * 32][18];
    F32* bary = s_bary[threadIdx.x + threadIdx.y * 32];

    // Compute task and image indices.

    int taskIdx = threadIdx.x + 32 * (threadIdx.y + CR_SETUP_WARPS * blockIdx.x);
    int imageIdx = 0;
    if (p.instanceMode)
    {
        imageIdx = blockIdx.z;
        if (taskIdx >= p.numTriangles)
            return;
    }
    else
    {
        while (imageIdx < p.numImages)
        {
            int count = getImageParams(p, imageIdx).triCount;
            if (taskIdx < count)
                break;
            taskIdx -= count;
            imageIdx += 1;
        }
        if (imageIdx == p.numImages)
            return;
    }

    // Per-image data structures.

    const CRImageParams& ip = getImageParams(p, imageIdx);
    CRAtomics& atomics = p.atomics[imageIdx];

    const int*          indexBuffer = (const int*)p.indexBuffer;
    U8*                 triSubtris  = (U8*)p.triSubtris               + imageIdx * p.maxSubtris;
    CRTriangleHeader*   triHeader   = (CRTriangleHeader*)p.triHeader  + imageIdx * p.maxSubtris;
    CRTriangleData*     triData     = (CRTriangleData*)p.triData      + imageIdx * p.maxSubtris;

    // Determine triangle index.

    int triIdx = taskIdx;
    if (!p.instanceMode)
        triIdx += ip.triOffset;

    // Read vertex indices.

    if ((U32)triIdx >= (U32)p.numTriangles)
    {
        // Bad triangle index.
        triSubtris[taskIdx] = 0;
        return;
    }

    uint4 vidx;
    vidx.x = indexBuffer[triIdx * 3 + 0];
    vidx.y = indexBuffer[triIdx * 3 + 1];
    vidx.z = indexBuffer[triIdx * 3 + 2];
    vidx.w = triIdx + 1; // Triangle index.

    if (vidx.x >= (U32)p.numVertices ||
        vidx.y >= (U32)p.numVertices ||
        vidx.z >= (U32)p.numVertices)
    {
        // Bad vertex index.
        triSubtris[taskIdx] = 0;
        return;
    }

    // Read vertex positions.

    const float4* vertexBuffer = (const float4*)p.vertexBuffer;
    if (p.instanceMode)
        vertexBuffer += p.numVertices * imageIdx; // Instance offset.

    float4 v0 = vertexBuffer[vidx.x];
    float4 v1 = vertexBuffer[vidx.y];
    float4 v2 = vertexBuffer[vidx.z];

    // Adjust vertex positions according to current viewport size and offset.

    v0.x = v0.x * p.xs + v0.w * p.xo;
    v0.y = v0.y * p.ys + v0.w * p.yo;
    v1.x = v1.x * p.xs + v1.w * p.xo;
    v1.y = v1.y * p.ys + v1.w * p.yo;
    v2.x = v2.x * p.xs + v2.w * p.xo;
    v2.y = v2.y * p.ys + v2.w * p.yo;

    // Outside view frustum => cull.

    if (v0.w < fabsf(v0.x) | v0.w < fabsf(v0.y) | v0.w < fabsf(v0.z))
    {
        if ((v0.w < +v0.x & v1.w < +v1.x & v2.w < +v2.x) |
            (v0.w < -v0.x & v1.w < -v1.x & v2.w < -v2.x) |
            (v0.w < +v0.y & v1.w < +v1.y & v2.w < +v2.y) |
            (v0.w < -v0.y & v1.w < -v1.y & v2.w < -v2.y) |
            (v0.w < +v0.z & v1.w < +v1.z & v2.w < +v2.z) |
            (v0.w < -v0.z & v1.w < -v1.z & v2.w < -v2.z))
        {
            triSubtris[taskIdx] = 0;
            return;
        }
    }

    // Inside depth range => try to snap vertices.

    if (v0.w >= fabsf(v0.z) & v1.w >= fabsf(v1.z) & v2.w >= fabsf(v2.z))
    {
        // Inside S16 range and small enough => fast path.
        // Note: aabbLimit comes from the fact that cover8x8
        // does not support guardband with maximal viewport.

        int2 p0, p1, p2, lo, hi;
        float3 rcpW;

        snapTriangle(p, v0, v1, v2, p0, p1, p2, rcpW, lo, hi);
        S32 loxy = ::min(lo.x, lo.y);
        S32 hixy = ::max(hi.x, hi.y);
        S32 aabbLimit = (1 << (CR_MAXVIEWPORT_LOG2 + CR_SUBPIXEL_LOG2)) - 1;

        if (loxy >= -32768 && hixy <= 32767 && hixy - loxy <= aabbLimit)
        {
            int2 d1, d2;
            S32 area;
            bool res = prepareTriangle(p, p0, p1, p2, lo, hi, d1, d2, area);
            triSubtris[taskIdx] = res ? 1 : 0;

            if (res)
                setupTriangle(
                    p,
                    &triHeader[taskIdx], &triData[taskIdx], vidx.w,
                    v0.z, v1.z, v2.z,
                    p0, p1, p2, rcpW,
                    d1, d2, area);

            return;
        }
    }

    // Clip to view frustum.

    float4 ov0 = v0;
    float4 od1 = make_float4(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z, v1.w - v0.w);
    float4 od2 = make_float4(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z, v2.w - v0.w);
    int numVerts = clipTriangleWithFrustum(bary, &ov0.x, &v1.x, &v2.x, &od1.x, &od2.x);

    // Count non-culled subtriangles.

    v0.x = ov0.x + od1.x * bary[0] + od2.x * bary[1];
    v0.y = ov0.y + od1.y * bary[0] + od2.y * bary[1];
    v0.z = ov0.z + od1.z * bary[0] + od2.z * bary[1];
    v0.w = ov0.w + od1.w * bary[0] + od2.w * bary[1];
    v1.x = ov0.x + od1.x * bary[2] + od2.x * bary[3];
    v1.y = ov0.y + od1.y * bary[2] + od2.y * bary[3];
    v1.z = ov0.z + od1.z * bary[2] + od2.z * bary[3];
    v1.w = ov0.w + od1.w * bary[2] + od2.w * bary[3];
    float4 tv1 = v1;

    int numSubtris = 0;
    for (int i = 2; i < numVerts; i++)
    {
        v2.x = ov0.x + od1.x * bary[i * 2 + 0] + od2.x * bary[i * 2 + 1];
        v2.y = ov0.y + od1.y * bary[i * 2 + 0] + od2.y * bary[i * 2 + 1];
        v2.z = ov0.z + od1.z * bary[i * 2 + 0] + od2.z * bary[i * 2 + 1];
        v2.w = ov0.w + od1.w * bary[i * 2 + 0] + od2.w * bary[i * 2 + 1];

        int2 p0, p1, p2, lo, hi, d1, d2;
        float3 rcpW;
        S32 area;

        snapTriangle(p, v0, v1, v2, p0, p1, p2, rcpW, lo, hi);
        if (prepareTriangle(p, p0, p1, p2, lo, hi, d1, d2, area))
            numSubtris++;

        v1 = v2;
    }

    triSubtris[taskIdx] = numSubtris;

    // Multiple subtriangles => allocate.

    int subtriBase = taskIdx;
    if (numSubtris > 1)
    {
        subtriBase = atomicAdd(&atomics.numSubtris, numSubtris);
        triHeader[taskIdx].misc = subtriBase;
        if (subtriBase + numSubtris > p.maxSubtris)
            numVerts = 0;
    }

    // Setup subtriangles.

    v1 = tv1;
    for (int i = 2; i < numVerts; i++)
    {
        v2.x = ov0.x + od1.x * bary[i * 2 + 0] + od2.x * bary[i * 2 + 1];
        v2.y = ov0.y + od1.y * bary[i * 2 + 0] + od2.y * bary[i * 2 + 1];
        v2.z = ov0.z + od1.z * bary[i * 2 + 0] + od2.z * bary[i * 2 + 1];
        v2.w = ov0.w + od1.w * bary[i * 2 + 0] + od2.w * bary[i * 2 + 1];

        int2 p0, p1, p2, lo, hi, d1, d2;
        float3 rcpW;
        S32 area;

        snapTriangle(p, v0, v1, v2, p0, p1, p2, rcpW, lo, hi);
        if (prepareTriangle(p, p0, p1, p2, lo, hi, d1, d2, area))
        {
            setupTriangle(
                p,
                &triHeader[subtriBase], &triData[subtriBase], vidx.w,
                v0.z, v1.z, v2.z,
                p0, p1, p2, rcpW,
                d1, d2, area);

            subtriBase++;
        }

        v1 = v2;
    }
}

//------------------------------------------------------------------------
