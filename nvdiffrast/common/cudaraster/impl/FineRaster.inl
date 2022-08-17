// Copyright (c) 2009-2022, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

//------------------------------------------------------------------------
// Utility funcs.
//------------------------------------------------------------------------

__device__ __inline__ void initTileZMax(U32& tileZMax, bool& tileZUpd, volatile U32* tileDepth)
{
    tileZMax = CR_DEPTH_MAX;
    tileZUpd = (::min(tileDepth[threadIdx.x], tileDepth[threadIdx.x + 32]) < tileZMax);
}

__device__ __inline__ void updateTileZMax(U32& tileZMax, bool& tileZUpd, volatile U32* tileDepth, volatile U32* temp)
{
    // Entry is warp-coherent.
    if (__any_sync(~0u, tileZUpd))
    {
        U32 z = ::max(tileDepth[threadIdx.x], tileDepth[threadIdx.x + 32]); __syncwarp();
        temp[threadIdx.x + 16] = z; __syncwarp();
        z = ::max(z, temp[threadIdx.x + 16 -  1]); __syncwarp(); temp[threadIdx.x + 16] = z; __syncwarp();
        z = ::max(z, temp[threadIdx.x + 16 -  2]); __syncwarp(); temp[threadIdx.x + 16] = z; __syncwarp();
        z = ::max(z, temp[threadIdx.x + 16 -  4]); __syncwarp(); temp[threadIdx.x + 16] = z; __syncwarp();
        z = ::max(z, temp[threadIdx.x + 16 -  8]); __syncwarp(); temp[threadIdx.x + 16] = z; __syncwarp();
        z = ::max(z, temp[threadIdx.x + 16 - 16]); __syncwarp(); temp[threadIdx.x + 16] = z; __syncwarp();
        tileZMax = temp[47];
        tileZUpd = false;
    }
}

//------------------------------------------------------------------------

__device__ __inline__ void getTriangle(const CRParams& p, S32& triIdx, S32& dataIdx, uint4& triHeader, S32& segment)
{
    const CRTriangleHeader* triHeaderPtr    = (const CRTriangleHeader*)p.triHeader + blockIdx.z * p.maxSubtris;;
    const S32*              tileSegData     = (const S32*)p.tileSegData  + p.maxTileSegs * CR_TILE_SEG_SIZE * blockIdx.z;
    const S32*              tileSegNext     = (const S32*)p.tileSegNext  + p.maxTileSegs * blockIdx.z;
    const S32*              tileSegCount    = (const S32*)p.tileSegCount + p.maxTileSegs * blockIdx.z;

    if (threadIdx.x >= tileSegCount[segment])
    {
        triIdx = -1;
        dataIdx = -1;
    }
    else
    {
        int subtriIdx = tileSegData[segment * CR_TILE_SEG_SIZE + threadIdx.x];
        triIdx = subtriIdx >> 3;
        dataIdx = triIdx;
        subtriIdx &= 7;
        if (subtriIdx != 7)
            dataIdx = triHeaderPtr[triIdx].misc + subtriIdx;
        triHeader = *((uint4*)triHeaderPtr + dataIdx);
    }

    // advance to next segment
    segment = tileSegNext[segment];
}

//------------------------------------------------------------------------

__device__ __inline__ bool earlyZCull(uint4 triHeader, U32 tileZMax)
{
    U32 zmin = triHeader.w & 0xFFFFF000;
    return (zmin > tileZMax);
}

//------------------------------------------------------------------------

__device__ __inline__ U64 trianglePixelCoverage(const CRParams& p, const uint4& triHeader, int tileX, int tileY, volatile U64* s_cover8x8_lut)
{
    int baseX = (tileX << (CR_TILE_LOG2 + CR_SUBPIXEL_LOG2)) - ((p.widthPixels  - 1) << (CR_SUBPIXEL_LOG2 - 1));
    int baseY = (tileY << (CR_TILE_LOG2 + CR_SUBPIXEL_LOG2)) - ((p.heightPixels - 1) << (CR_SUBPIXEL_LOG2 - 1));

    // extract S16 vertex positions while subtracting tile coordinates
    S32 v0x  = sub_s16lo_s16lo(triHeader.x, baseX);
    S32 v0y  = sub_s16hi_s16lo(triHeader.x, baseY);
    S32 v01x = sub_s16lo_s16lo(triHeader.y, triHeader.x);
    S32 v01y = sub_s16hi_s16hi(triHeader.y, triHeader.x);
    S32 v20x = sub_s16lo_s16lo(triHeader.x, triHeader.z);
    S32 v20y = sub_s16hi_s16hi(triHeader.x, triHeader.z);

    // extract flipbits
    U32 f01 = (triHeader.w >> 6) & 0x3C;
    U32 f12 = (triHeader.w >> 2) & 0x3C;
    U32 f20 = (triHeader.w << 2) & 0x3C;

    // compute per-edge coverage masks
    U64 c01, c12, c20;
    c01 = cover8x8_exact_fast(v0x, v0y, v01x, v01y, f01, s_cover8x8_lut);
    c12 = cover8x8_exact_fast(v0x + v01x, v0y + v01y, -v01x - v20x, -v01y - v20y, f12, s_cover8x8_lut);
    c20 = cover8x8_exact_fast(v0x, v0y, v20x, v20y, f20, s_cover8x8_lut);

    // combine masks
    return c01 & c12 & c20;
}

//------------------------------------------------------------------------

__device__ __inline__ U32 scan32_value(U32 value, volatile U32* temp)
{
    __syncwarp();
    temp[threadIdx.x + 16] = value; __syncwarp();
    value += temp[threadIdx.x + 16 -  1]; __syncwarp(); temp[threadIdx.x + 16] = value; __syncwarp();
    value += temp[threadIdx.x + 16 -  2]; __syncwarp(); temp[threadIdx.x + 16] = value; __syncwarp();
    value += temp[threadIdx.x + 16 -  4]; __syncwarp(); temp[threadIdx.x + 16] = value; __syncwarp();
    value += temp[threadIdx.x + 16 -  8]; __syncwarp(); temp[threadIdx.x + 16] = value; __syncwarp();
    value += temp[threadIdx.x + 16 - 16]; __syncwarp(); temp[threadIdx.x + 16] = value; __syncwarp();
    return value;
}

__device__ __inline__ volatile const U32& scan32_total(volatile U32* temp)
{
    return temp[47];
}

//------------------------------------------------------------------------

__device__ __inline__ S32 findBit(U64 mask, int idx)
{
    U32 x = getLo(mask);
    int  pop = __popc(x);
    bool p   = (pop <= idx);
    if (p) x = getHi(mask);
    if (p) idx -= pop;
    int bit = p ? 32 : 0;

    pop = __popc(x & 0x0000ffffu);
    p   = (pop <= idx);
    if (p) x >>= 16;
    if (p) bit += 16;
    if (p) idx -= pop;

    U32 tmp = x & 0x000000ffu;
    pop = __popc(tmp);
    p   = (pop <= idx);
    if (p) tmp = x & 0x0000ff00u;
    if (p) idx -= pop;

    return findLeadingOne(tmp) + bit - idx;
}

//------------------------------------------------------------------------
// Single-sample implementation.
//------------------------------------------------------------------------

__device__ __inline__ void executeROP(U32 color, U32 depth, volatile U32* pColor, volatile U32* pDepth, U32 ropMask)
{
    atomicMin((U32*)pDepth, depth);
    __syncwarp(ropMask);
    bool act = (depth == *pDepth);
    __syncwarp(ropMask);
    U32 actMask = __ballot_sync(ropMask, act);
    if (act)
    {
        *pDepth = 0;
        __syncwarp(actMask);
        atomicMax((U32*)pDepth, threadIdx.x);
        __syncwarp(actMask);
        if (*pDepth == threadIdx.x)
        {
            *pDepth = depth;
            *pColor = color;
        }
        __syncwarp(actMask);
    }
}

//------------------------------------------------------------------------

__device__ __inline__ void fineRasterImpl(const CRParams p)
{
                                                                            // for 20 warps:
    __shared__ volatile U64 s_cover8x8_lut[CR_COVER8X8_LUT_SIZE];           // 6KB
    __shared__ volatile U32 s_tileColor   [CR_FINE_MAX_WARPS][CR_TILE_SQR]; // 5KB
    __shared__ volatile U32 s_tileDepth   [CR_FINE_MAX_WARPS][CR_TILE_SQR]; // 5KB
    __shared__ volatile U32 s_tilePeel    [CR_FINE_MAX_WARPS][CR_TILE_SQR]; // 5KB
    __shared__ volatile U32 s_triDataIdx  [CR_FINE_MAX_WARPS][64];          // 5KB  CRTriangleData index
    __shared__ volatile U64 s_triangleCov [CR_FINE_MAX_WARPS][64];          // 10KB coverage mask
    __shared__ volatile U32 s_triangleFrag[CR_FINE_MAX_WARPS][64];          // 5KB  fragment index
    __shared__ volatile U32 s_temp        [CR_FINE_MAX_WARPS][80];          // 6.25KB
                                                                            // = 47.25KB total

    CRAtomics&            atomics   = p.atomics[blockIdx.z];
    const CRTriangleData* triData   = (const CRTriangleData*)p.triData + blockIdx.z * p.maxSubtris;

    const S32*      activeTiles     = (const S32*)p.activeTiles  + CR_MAXTILES_SQR * blockIdx.z;
    const S32*      tileFirstSeg    = (const S32*)p.tileFirstSeg + CR_MAXTILES_SQR * blockIdx.z;

    volatile U32*   tileColor       = s_tileColor[threadIdx.y];
    volatile U32*   tileDepth       = s_tileDepth[threadIdx.y];
    volatile U32*   tilePeel        = s_tilePeel[threadIdx.y];
    volatile U32*   triDataIdx      = s_triDataIdx[threadIdx.y];
    volatile U64*   triangleCov     = s_triangleCov[threadIdx.y];
    volatile U32*   triangleFrag    = s_triangleFrag[threadIdx.y];
    volatile U32*   temp            = s_temp[threadIdx.y];

    if (atomics.numSubtris > p.maxSubtris || atomics.numBinSegs > p.maxBinSegs || atomics.numTileSegs > p.maxTileSegs)
        return;

    temp[threadIdx.x] = 0; // first 16 elements of temp are always zero
    cover8x8_setupLUT(s_cover8x8_lut);
    __syncthreads();

    // loop over tiles
    for (;;)
    {
        // pick a tile
        if (threadIdx.x == 0)
            temp[16] = atomicAdd(&atomics.fineCounter, 1);
        __syncwarp();
        int activeIdx = temp[16];
        if (activeIdx >= atomics.numActiveTiles)
            break;

        int tileIdx = activeTiles[activeIdx];
        S32 segment = tileFirstSeg[tileIdx];
        int tileY = tileIdx / p.widthTiles;
        int tileX = tileIdx - tileY * p.widthTiles;
        int px = (tileX << CR_TILE_LOG2) + (threadIdx.x & (CR_TILE_SIZE - 1));
        int py = (tileY << CR_TILE_LOG2) + (threadIdx.x >> CR_TILE_LOG2);

        // initialize per-tile state
        int triRead = 0, triWrite = 0;
        int fragRead = 0, fragWrite = 0;
        if (threadIdx.x == 0)
            triangleFrag[63] = 0; // "previous triangle"

        // deferred clear => clear tile
        if (p.deferredClear)
        {
			tileColor[threadIdx.x] = p.clearColor;
            tileDepth[threadIdx.x] = p.clearDepth;
            tileColor[threadIdx.x + 32] = p.clearColor;
            tileDepth[threadIdx.x + 32] = p.clearDepth;
        }
        else // otherwise => read tile from framebuffer
        {
            U32* pColor = (U32*)p.colorBuffer + p.widthPixels * p.heightPixels * blockIdx.z;
            U32* pDepth = (U32*)p.depthBuffer + p.widthPixels * p.heightPixels * blockIdx.z;
			tileColor[threadIdx.x] = pColor[px + p.widthPixels * py];
            tileDepth[threadIdx.x] = pDepth[px + p.widthPixels * py];
            tileColor[threadIdx.x + 32] = pColor[px + p.widthPixels * (py + 4)];
            tileDepth[threadIdx.x + 32] = pDepth[px + p.widthPixels * (py + 4)];
        }

        // read peeling inputs if enabled
        if (p.renderModeFlags & CudaRaster::RenderModeFlag_EnableDepthPeeling)
        {
            U32* pPeel = (U32*)p.peelBuffer + p.widthPixels * p.heightPixels * blockIdx.z;
            tilePeel[threadIdx.x] = pPeel[px + p.widthPixels * py];
            tilePeel[threadIdx.x + 32] = pPeel[px + p.widthPixels * (py + 4)];
        }

        U32 tileZMax;
        bool tileZUpd;
        initTileZMax(tileZMax, tileZUpd, tileDepth);

        // process fragments
        for(;;)
        {
            // need to queue more fragments?
            if (fragWrite - fragRead < 32 && segment >= 0)
            {
                // update tile z - coherent over warp
                updateTileZMax(tileZMax, tileZUpd, tileDepth, temp);

                // read triangles
                do
                {
                    // read triangle index and data, advance to next segment
                    S32 triIdx, dataIdx;
                    uint4 triHeader;
                    getTriangle(p, triIdx, dataIdx, triHeader, segment);

                    // early z cull
                    if (triIdx >= 0 && earlyZCull(triHeader, tileZMax))
                        triIdx = -1;

                    // determine coverage
                    U64 coverage = trianglePixelCoverage(p, triHeader, tileX, tileY, s_cover8x8_lut);
                    S32 pop = (triIdx == -1) ? 0 : __popcll(coverage);

                    // fragment count scan
                    U32 frag = scan32_value(pop, temp);
                    frag += fragWrite; // frag now holds cumulative fragment count
                    fragWrite += scan32_total(temp);

                    // queue non-empty triangles
                    U32 goodMask = __ballot_sync(~0u, pop != 0);
                    if (pop != 0)
                    {
                        int idx = (triWrite + __popc(goodMask & getLaneMaskLt())) & 63;
                        triDataIdx  [idx] = dataIdx;
                        triangleFrag[idx] = frag;
                        triangleCov [idx] = coverage;
                    }
                    triWrite += __popc(goodMask);
                }
                while (fragWrite - fragRead < 32 && segment >= 0);
            }
            __syncwarp();

            // end of segment?
            if (fragRead == fragWrite)
                break;

            // clear triangle boundaries
            temp[threadIdx.x + 16] = 0;
            __syncwarp();

            // tag triangle boundaries
            if (triRead + threadIdx.x < triWrite)
            {
                int idx = triangleFrag[(triRead + threadIdx.x) & 63] - fragRead;
                if (idx <= 32)
                    temp[idx + 16 - 1] = 1;
            }
            __syncwarp();

            int ropLaneIdx = threadIdx.x;
            U32 boundaryMask = __ballot_sync(~0u, temp[ropLaneIdx + 16]);

            // distribute fragments
            bool hasFragment = (ropLaneIdx < fragWrite - fragRead);
            U32 fragmentMask = __ballot_sync(~0u, hasFragment);
            if (hasFragment)
            {
                int triBufIdx = (triRead + __popc(boundaryMask & getLaneMaskLt())) & 63;
                int fragIdx = add_sub(fragRead, ropLaneIdx, triangleFrag[(triBufIdx - 1) & 63]);
                U64 coverage = triangleCov[triBufIdx];
                int pixelInTile = findBit(coverage, fragIdx);
                int dataIdx = triDataIdx[triBufIdx];

                // determine pixel position
                U32 pixelX = (tileX << CR_TILE_LOG2) + (pixelInTile & 7);
                U32 pixelY = (tileY << CR_TILE_LOG2) + (pixelInTile >> 3);

                // depth test
                U32 depth = 0;
                uint4 td = *((uint4*)triData + dataIdx * (sizeof(CRTriangleData) >> 4));

                depth = td.x * pixelX + td.y * pixelY + td.z;
                bool zkill = (p.renderModeFlags & CudaRaster::RenderModeFlag_EnableDepthPeeling) && (depth <= tilePeel[pixelInTile]);
                if (!zkill)
                {
                    U32 oldDepth = tileDepth[pixelInTile];
                    if (depth > oldDepth)
                        zkill = true;
                    else if (oldDepth == tileZMax)
                        tileZUpd = true; // we are replacing previous zmax => need to update
                }

                U32 ropMask = __ballot_sync(fragmentMask, !zkill);
                if (!zkill)
					executeROP(td.w, depth, &tileColor[pixelInTile], &tileDepth[pixelInTile], ropMask);
            }
            // no need to sync, as next up is updateTileZMax that does internal warp sync

            // update counters
            fragRead = ::min(fragRead + 32, fragWrite);
            triRead += __popc(boundaryMask);
        }

        // Write tile back to the framebuffer.
        if (true)
        {
            int px = (tileX << CR_TILE_LOG2) + (threadIdx.x & (CR_TILE_SIZE - 1));
            int py = (tileY << CR_TILE_LOG2) + (threadIdx.x >> CR_TILE_LOG2);
            U32* pColor = (U32*)p.colorBuffer + p.widthPixels * p.heightPixels * blockIdx.z;
            U32* pDepth = (U32*)p.depthBuffer + p.widthPixels * p.heightPixels * blockIdx.z;
            pColor[px + p.widthPixels * py] = tileColor[threadIdx.x];
            pDepth[px + p.widthPixels * py] = tileDepth[threadIdx.x];
            pColor[px + p.widthPixels * (py + 4)] = tileColor[threadIdx.x + 32];
            pDepth[px + p.widthPixels * (py + 4)] = tileDepth[threadIdx.x + 32];
        }
    }
}

//------------------------------------------------------------------------
