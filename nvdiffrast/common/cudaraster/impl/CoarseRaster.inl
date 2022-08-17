// Copyright (c) 2009-2022, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

//------------------------------------------------------------------------

__device__ __inline__ int globalTileIdx(int tileInBin, int widthTiles)
{
    int tileX = tileInBin & (CR_BIN_SIZE - 1);
    int tileY = tileInBin >> CR_BIN_LOG2;
    return tileX + tileY * widthTiles;
}

//------------------------------------------------------------------------

__device__ __inline__ void coarseRasterImpl(const CRParams p)
{
    // Common.

    __shared__ volatile U32 s_workCounter;
    __shared__ volatile U32 s_scanTemp          [CR_COARSE_WARPS][48];              // 3KB

    // Input.

    __shared__ volatile U32 s_binOrder          [CR_MAXBINS_SQR];                   // 1KB
    __shared__ volatile S32 s_binStreamCurrSeg  [CR_BIN_STREAMS_SIZE];              // 0KB
    __shared__ volatile S32 s_binStreamFirstTri [CR_BIN_STREAMS_SIZE];              // 0KB
    __shared__ volatile S32 s_triQueue          [CR_COARSE_QUEUE_SIZE];             // 4KB
    __shared__ volatile S32 s_triQueueWritePos;
    __shared__ volatile U32 s_binStreamSelectedOfs;
    __shared__ volatile U32 s_binStreamSelectedSize;

    // Output.

    __shared__ volatile U32 s_warpEmitMask      [CR_COARSE_WARPS][CR_BIN_SQR + 1];  // 16KB, +1 to avoid bank collisions
    __shared__ volatile U32 s_warpEmitPrefixSum [CR_COARSE_WARPS][CR_BIN_SQR + 1];  // 16KB, +1 to avoid bank collisions
    __shared__ volatile U32 s_tileEmitPrefixSum [CR_BIN_SQR + 1];                   // 1KB, zero at the beginning
    __shared__ volatile U32 s_tileAllocPrefixSum[CR_BIN_SQR + 1];                   // 1KB, zero at the beginning
    __shared__ volatile S32 s_tileStreamCurrOfs [CR_BIN_SQR];                       // 1KB
    __shared__ volatile U32 s_firstAllocSeg;
    __shared__ volatile U32 s_firstActiveIdx;

    // Pointers and constants.

    CRAtomics&              atomics         = p.atomics[blockIdx.z];
    const CRTriangleHeader* triHeader       = (const CRTriangleHeader*)p.triHeader + p.maxSubtris * blockIdx.z;
    const S32*              binFirstSeg     = (const S32*)p.binFirstSeg + CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * blockIdx.z;
    const S32*              binTotal        = (const S32*)p.binTotal    + CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * blockIdx.z;
    const S32*              binSegData      = (const S32*)p.binSegData  + p.maxBinSegs * CR_BIN_SEG_SIZE * blockIdx.z;
    const S32*              binSegNext      = (const S32*)p.binSegNext  + p.maxBinSegs * blockIdx.z;
    const S32*              binSegCount     = (const S32*)p.binSegCount + p.maxBinSegs * blockIdx.z;
    S32*                    activeTiles     = (S32*)p.activeTiles  + CR_MAXTILES_SQR * blockIdx.z;
    S32*                    tileFirstSeg    = (S32*)p.tileFirstSeg + CR_MAXTILES_SQR * blockIdx.z;
    S32*                    tileSegData     = (S32*)p.tileSegData  + p.maxTileSegs * CR_TILE_SEG_SIZE * blockIdx.z;
    S32*                    tileSegNext     = (S32*)p.tileSegNext  + p.maxTileSegs * blockIdx.z;
    S32*                    tileSegCount    = (S32*)p.tileSegCount + p.maxTileSegs * blockIdx.z;

    int tileLog     = CR_TILE_LOG2 + CR_SUBPIXEL_LOG2;
    int thrInBlock  = threadIdx.x + threadIdx.y * 32;
    int emitShift   = CR_BIN_LOG2 * 2 + 5; // We scan ((numEmits << emitShift) | numAllocs) over tiles.

    if (atomics.numSubtris > p.maxSubtris || atomics.numBinSegs > p.maxBinSegs)
        return;

    // Initialize sharedmem arrays.

    if (thrInBlock == 0)
    {
        s_tileEmitPrefixSum[0] = 0;
        s_tileAllocPrefixSum[0] = 0;
    }
    s_scanTemp[threadIdx.y][threadIdx.x] = 0;

    // Sort bins in descending order of triangle count.

    for (int binIdx = thrInBlock; binIdx < p.numBins; binIdx += CR_COARSE_WARPS * 32)
    {
        int count = 0;
        for (int i = 0; i < CR_BIN_STREAMS_SIZE; i++)
            count += binTotal[(binIdx << CR_BIN_STREAMS_LOG2) + i];
        s_binOrder[binIdx] = (~count << (CR_MAXBINS_LOG2 * 2)) | binIdx;
    }

    __syncthreads();
    sortShared(s_binOrder, p.numBins);

    // Process each bin by one block.

    for (;;)
    {
        // Pick a bin for the block.

        if (thrInBlock == 0)
            s_workCounter = atomicAdd(&atomics.coarseCounter, 1);
        __syncthreads();

        int workCounter = s_workCounter;
        if (workCounter >= p.numBins)
            break;

        U32 binOrder = s_binOrder[workCounter];
        bool binEmpty = ((~binOrder >> (CR_MAXBINS_LOG2 * 2)) == 0);
        if (binEmpty && !p.deferredClear)
            break;

        int binIdx = binOrder & (CR_MAXBINS_SQR - 1);

        // Initialize input/output streams.

        int triQueueWritePos = 0;
        int triQueueReadPos = 0;

        if (thrInBlock < CR_BIN_STREAMS_SIZE)
        {
            int segIdx = binFirstSeg[(binIdx << CR_BIN_STREAMS_LOG2) + thrInBlock];
            s_binStreamCurrSeg[thrInBlock] = segIdx;
            s_binStreamFirstTri[thrInBlock] = (segIdx == -1) ? ~0u : binSegData[segIdx << CR_BIN_SEG_LOG2];
        }

        for (int tileInBin = CR_COARSE_WARPS * 32 - 1 - thrInBlock; tileInBin < CR_BIN_SQR; tileInBin += CR_COARSE_WARPS * 32)
            s_tileStreamCurrOfs[tileInBin] = -CR_TILE_SEG_SIZE;

        // Initialize per-bin state.

        int binY = idiv_fast(binIdx, p.widthBins);
        int binX = binIdx - binY * p.widthBins;
        int originX = (binX << (CR_BIN_LOG2 + tileLog)) - (p.widthPixels << (CR_SUBPIXEL_LOG2 - 1));
        int originY = (binY << (CR_BIN_LOG2 + tileLog)) - (p.heightPixels << (CR_SUBPIXEL_LOG2 - 1));
        int maxTileXInBin = ::min(p.widthTiles - (binX << CR_BIN_LOG2), CR_BIN_SIZE) - 1;
        int maxTileYInBin = ::min(p.heightTiles - (binY << CR_BIN_LOG2), CR_BIN_SIZE) - 1;
        int binTileIdx = (binX + binY * p.widthTiles) << CR_BIN_LOG2;

        // Entire block: Merge input streams and process triangles.

        if (!binEmpty)
        do
        {
            //------------------------------------------------------------------------
            // Merge.
            //------------------------------------------------------------------------

            // Entire block: Not enough triangles => merge and queue segments.
            // NOTE: The bin exit criterion assumes that we queue more triangles than we actually need.

            while (triQueueWritePos - triQueueReadPos <= CR_COARSE_WARPS * 32)
            {
                // First warp: Choose the segment with the lowest initial triangle index.

                bool hasStream = (thrInBlock < CR_BIN_STREAMS_SIZE);
                U32 hasStreamMask = __ballot_sync(~0u, hasStream);
                if (hasStream)
                {
                    // Find the stream with the lowest triangle index.

                    U32 firstTri = s_binStreamFirstTri[thrInBlock];
                    U32 t = firstTri;
                    volatile U32* v = &s_scanTemp[0][thrInBlock + 16];

                    #if (CR_BIN_STREAMS_SIZE > 1)
                        v[0] = t; __syncwarp(hasStreamMask); t = ::min(t, v[-1]); __syncwarp(hasStreamMask);
                    #endif
                    #if (CR_BIN_STREAMS_SIZE > 2)
                        v[0] = t; __syncwarp(hasStreamMask); t = ::min(t, v[-2]); __syncwarp(hasStreamMask);
                    #endif
                    #if (CR_BIN_STREAMS_SIZE > 4)
                        v[0] = t; __syncwarp(hasStreamMask); t = ::min(t, v[-4]); __syncwarp(hasStreamMask);
                    #endif
                    #if (CR_BIN_STREAMS_SIZE > 8)
                        v[0] = t; __syncwarp(hasStreamMask); t = ::min(t, v[-8]); __syncwarp(hasStreamMask);
                    #endif
                    #if (CR_BIN_STREAMS_SIZE > 16)
                        v[0] = t; __syncwarp(hasStreamMask); t = ::min(t, v[-16]); __syncwarp(hasStreamMask);
                    #endif
                    v[0] = t; __syncwarp(hasStreamMask);

                    // Consume and broadcast.

                    bool first = (s_scanTemp[0][CR_BIN_STREAMS_SIZE - 1 + 16] == firstTri);
                    U32 firstMask = __ballot_sync(hasStreamMask, first);
                    if (first && (firstMask >> threadIdx.x) == 1u)
                    {
                        int segIdx = s_binStreamCurrSeg[thrInBlock];
                        s_binStreamSelectedOfs = segIdx << CR_BIN_SEG_LOG2;
                        if (segIdx != -1)
                        {
                            int segSize = binSegCount[segIdx];
                            int segNext = binSegNext[segIdx];
                            s_binStreamSelectedSize = segSize;
                            s_triQueueWritePos = triQueueWritePos + segSize;
                            s_binStreamCurrSeg[thrInBlock] = segNext;
                            s_binStreamFirstTri[thrInBlock] = (segNext == -1) ? ~0u : binSegData[segNext << CR_BIN_SEG_LOG2];
                        }
                    }
                }

                // No more segments => break.

                __syncthreads();
                triQueueWritePos = s_triQueueWritePos;
                int segOfs = s_binStreamSelectedOfs;
                if (segOfs < 0)
                    break;

                int segSize = s_binStreamSelectedSize;
                __syncthreads();

                // Fetch triangles into the queue.

                for (int idxInSeg = CR_COARSE_WARPS * 32 - 1 - thrInBlock; idxInSeg < segSize; idxInSeg += CR_COARSE_WARPS * 32)
                {
                    S32 triIdx = binSegData[segOfs + idxInSeg];
                    s_triQueue[(triQueueWritePos - segSize + idxInSeg) & (CR_COARSE_QUEUE_SIZE - 1)] = triIdx;
                }
            }

            // All threads: Clear emit masks.

            for (int maskIdx = thrInBlock; maskIdx < CR_COARSE_WARPS * CR_BIN_SQR; maskIdx += CR_COARSE_WARPS * 32)
                s_warpEmitMask[maskIdx >> (CR_BIN_LOG2 * 2)][maskIdx & (CR_BIN_SQR - 1)] = 0;

            __syncthreads();

            //------------------------------------------------------------------------
            // Raster.
            //------------------------------------------------------------------------

            // Triangle per thread: Read from the queue.

            int triIdx = -1;
            if (triQueueReadPos + thrInBlock < triQueueWritePos)
                triIdx = s_triQueue[(triQueueReadPos + thrInBlock) & (CR_COARSE_QUEUE_SIZE - 1)];

            uint4 triData = make_uint4(0, 0, 0, 0);
            if (triIdx != -1)
            {
                int dataIdx = triIdx >> 3;
                int subtriIdx = triIdx & 7;
                if (subtriIdx != 7)
                    dataIdx = triHeader[dataIdx].misc + subtriIdx;
                triData = *((uint4*)triHeader + dataIdx);
            }

            // 32 triangles per warp: Record emits (= tile intersections).

            if (__any_sync(~0u, triIdx != -1))
            {
                S32 v0x = sub_s16lo_s16lo(triData.x, originX);
                S32 v0y = sub_s16hi_s16lo(triData.x, originY);
                S32 d01x = sub_s16lo_s16lo(triData.y, triData.x);
                S32 d01y = sub_s16hi_s16hi(triData.y, triData.x);
                S32 d02x = sub_s16lo_s16lo(triData.z, triData.x);
                S32 d02y = sub_s16hi_s16hi(triData.z, triData.x);

                // Compute tile-based AABB.

                int lox = add_clamp_0_x((v0x + min_min(d01x, 0, d02x)) >> tileLog, 0, maxTileXInBin);
                int loy = add_clamp_0_x((v0y + min_min(d01y, 0, d02y)) >> tileLog, 0, maxTileYInBin);
                int hix = add_clamp_0_x((v0x + max_max(d01x, 0, d02x)) >> tileLog, 0, maxTileXInBin);
                int hiy = add_clamp_0_x((v0y + max_max(d01y, 0, d02y)) >> tileLog, 0, maxTileYInBin);
                int sizex = add_sub(hix, 1, lox);
                int sizey = add_sub(hiy, 1, loy);
                int area = sizex * sizey;

                // Miscellaneous init.

                U8* currPtr = (U8*)&s_warpEmitMask[threadIdx.y][lox + (loy << CR_BIN_LOG2)];
                int ptrYInc = CR_BIN_SIZE * 4 - (sizex << 2);
                U32 maskBit = 1 << threadIdx.x;

                // Case A: All AABBs are small => record the full AABB using atomics.

                if (__all_sync(~0u, sizex <= 2 && sizey <= 2))
                {
                    if (triIdx != -1)
                    {
                        atomicOr((U32*)currPtr, maskBit);
                        if (sizex == 2) atomicOr((U32*)(currPtr + 4), maskBit);
                        if (sizey == 2) atomicOr((U32*)(currPtr + CR_BIN_SIZE * 4), maskBit);
                        if (sizex == 2 && sizey == 2) atomicOr((U32*)(currPtr + 4 + CR_BIN_SIZE * 4), maskBit);
                    }
                }
                else
                {
                    // Compute warp-AABB (scan-32).

                    U32 aabbMask = add_sub(2 << hix, 0x20000 << hiy, 1 << lox) - (0x10000 << loy);
                    if (triIdx == -1)
                        aabbMask = 0;

                    volatile U32* v = &s_scanTemp[threadIdx.y][threadIdx.x + 16];
                    v[0] = aabbMask; __syncwarp(); aabbMask |= v[-1]; __syncwarp();
                    v[0] = aabbMask; __syncwarp(); aabbMask |= v[-2]; __syncwarp();
                    v[0] = aabbMask; __syncwarp(); aabbMask |= v[-4]; __syncwarp();
                    v[0] = aabbMask; __syncwarp(); aabbMask |= v[-8]; __syncwarp();
                    v[0] = aabbMask; __syncwarp(); aabbMask |= v[-16]; __syncwarp();
                    v[0] = aabbMask; __syncwarp(); aabbMask = s_scanTemp[threadIdx.y][47];

                    U32 maskX = aabbMask & 0xFFFF;
                    U32 maskY = aabbMask >> 16;
                    int wlox = findLeadingOne(maskX ^ (maskX - 1));
                    int wloy = findLeadingOne(maskY ^ (maskY - 1));
                    int whix = findLeadingOne(maskX);
                    int whiy = findLeadingOne(maskY);
                    int warea = (add_sub(whix, 1, wlox)) * (add_sub(whiy, 1, wloy));

                    // Initialize edge functions.

                    S32 d12x = d02x - d01x;
                    S32 d12y = d02y - d01y;
                    v0x -= lox << tileLog;
                    v0y -= loy << tileLog;

                    S32 t01 = v0x * d01y - v0y * d01x;
                    S32 t02 = v0y * d02x - v0x * d02y;
                    S32 t12 = d01x * d12y - d01y * d12x - t01 - t02;
                    S32 b01 = add_sub(t01 >> tileLog, ::max(d01x, 0), ::min(d01y, 0));
                    S32 b02 = add_sub(t02 >> tileLog, ::max(d02y, 0), ::min(d02x, 0));
                    S32 b12 = add_sub(t12 >> tileLog, ::max(d12x, 0), ::min(d12y, 0));

                    d01x += sizex * d01y;
                    d02x += sizex * d02y;
                    d12x += sizex * d12y;

                    // Case B: Warp-AABB is not much larger than largest AABB => Check tiles in warp-AABB, record using ballots.
                    if (__any_sync(~0u, warea * 4 <= area * 8))
                    {
                        // Not sure if this is any faster than Case C after all the post-Volta ballot mask tracking.
                        bool act = (triIdx != -1);
                        U32 actMask = __ballot_sync(~0u, act);
                        if (act)
                        {
                            for (int y = wloy; y <= whiy; y++)
                            {
                                bool yIn = (y >= loy && y <= hiy);
                                U32 yMask = __ballot_sync(actMask, yIn);
                                if (yIn)
                                {
                                    for (int x = wlox; x <= whix; x++)
                                    {
                                        bool xyIn = (x >= lox && x <= hix);
                                        U32 xyMask = __ballot_sync(yMask, xyIn);
                                        if (xyIn)
                                        {
                                            U32 res = __ballot_sync(xyMask, b01 >= 0 && b02 >= 0 && b12 >= 0);
                                            if (threadIdx.x == 31 - __clz(xyMask))
                                                *(U32*)currPtr = res;
                                            currPtr += 4, b01 -= d01y, b02 += d02y, b12 -= d12y;
                                        }
                                    }
                                    currPtr += ptrYInc, b01 += d01x, b02 -= d02x, b12 += d12x;
                                }
                            }
                        }
                    }

                    // Case C: General case => Check tiles in AABB, record using atomics.

                    else
                    {
                        if (triIdx != -1)
                        {
                            U8* skipPtr = currPtr + (sizex << 2);
                            U8* endPtr  = currPtr + (sizey << (CR_BIN_LOG2 + 2));
                            do
                            {
                                if (b01 >= 0 && b02 >= 0 && b12 >= 0)
                                    atomicOr((U32*)currPtr, maskBit);
                                currPtr += 4, b01 -= d01y, b02 += d02y, b12 -= d12y;
                                if (currPtr == skipPtr)
                                    currPtr += ptrYInc, b01 += d01x, b02 -= d02x, b12 += d12x, skipPtr += CR_BIN_SIZE * 4;
                            }
                            while (currPtr != endPtr);
                        }
                    }
                }
            }

            __syncthreads();

            //------------------------------------------------------------------------
            // Count.
            //------------------------------------------------------------------------

            // Tile per thread: Initialize prefix sums.

            for (int tileInBin_base = 0; tileInBin_base < CR_BIN_SQR; tileInBin_base += CR_COARSE_WARPS * 32)
            {
                int tileInBin = tileInBin_base + thrInBlock;
                bool act = (tileInBin < CR_BIN_SQR);
                U32 actMask = __ballot_sync(~0u, act);
                if (act)
                {
                    // Compute prefix sum of emits over warps.

                    U8* srcPtr = (U8*)&s_warpEmitMask[0][tileInBin];
                    U8* dstPtr = (U8*)&s_warpEmitPrefixSum[0][tileInBin];
                    int tileEmits = 0;
                    for (int i = 0; i < CR_COARSE_WARPS; i++)
                    {
                        tileEmits += __popc(*(U32*)srcPtr);
                        *(U32*)dstPtr = tileEmits;
                        srcPtr += (CR_BIN_SQR + 1) * 4;
                        dstPtr += (CR_BIN_SQR + 1) * 4;
                    }

                    // Determine the number of segments to allocate.

                    int spaceLeft = -s_tileStreamCurrOfs[tileInBin] & (CR_TILE_SEG_SIZE - 1);
                    int tileAllocs = (tileEmits - spaceLeft + CR_TILE_SEG_SIZE - 1) >> CR_TILE_SEG_LOG2;
                    volatile U32* v = &s_tileEmitPrefixSum[tileInBin + 1];

                    // All counters within the warp are small => compute prefix sum using ballot.

                    if (!__any_sync(actMask, tileEmits >= 2))
                    {
                        U32 m = getLaneMaskLe();
                        *v = (__popc(__ballot_sync(actMask, tileEmits & 1) & m) << emitShift) | __popc(__ballot_sync(actMask, tileAllocs & 1) & m);
                    }

                    // Otherwise => scan-32 within the warp.

                    else
                    {
                        U32 sum = (tileEmits << emitShift) | tileAllocs;
                        *v = sum; __syncwarp(actMask); if (threadIdx.x >= 1)  sum += v[-1]; __syncwarp(actMask);
                        *v = sum; __syncwarp(actMask); if (threadIdx.x >= 2)  sum += v[-2]; __syncwarp(actMask);
                        *v = sum; __syncwarp(actMask); if (threadIdx.x >= 4)  sum += v[-4]; __syncwarp(actMask);
                        *v = sum; __syncwarp(actMask); if (threadIdx.x >= 8)  sum += v[-8]; __syncwarp(actMask);
                        *v = sum; __syncwarp(actMask); if (threadIdx.x >= 16) sum += v[-16]; __syncwarp(actMask);
                        *v = sum; __syncwarp(actMask);
                    }
                }
            }

            // First warp: Scan-8.

            __syncthreads();

            bool scan8 = (thrInBlock < CR_BIN_SQR / 32);
            U32 scan8Mask = __ballot_sync(~0u, scan8);
            if (scan8)
            {
                int sum = s_tileEmitPrefixSum[(thrInBlock << 5) + 32];
                volatile U32* v = &s_scanTemp[0][thrInBlock + 16];
                v[0] = sum; __syncwarp(scan8Mask);
                #if (CR_BIN_SQR > 1 * 32)
                    sum += v[-1]; __syncwarp(scan8Mask); v[0] = sum; __syncwarp(scan8Mask);
                #endif
                #if (CR_BIN_SQR > 2 * 32)
                    sum += v[-2]; __syncwarp(scan8Mask); v[0] = sum; __syncwarp(scan8Mask);
                #endif
                #if (CR_BIN_SQR > 4 * 32)
                    sum += v[-4]; __syncwarp(scan8Mask); v[0] = sum; __syncwarp(scan8Mask);
                #endif
            }

            __syncthreads();

            // Tile per thread: Finalize prefix sums.
            // Single thread: Allocate segments.

            for (int tileInBin = thrInBlock; tileInBin < CR_BIN_SQR; tileInBin += CR_COARSE_WARPS * 32)
            {
                int sum = s_tileEmitPrefixSum[tileInBin + 1] + s_scanTemp[0][(tileInBin >> 5) + 15];
                int numEmits = sum >> emitShift;
                int numAllocs = sum & ((1 << emitShift) - 1);
                s_tileEmitPrefixSum[tileInBin + 1] = numEmits;
                s_tileAllocPrefixSum[tileInBin + 1] = numAllocs;

                if (tileInBin == CR_BIN_SQR - 1 && numAllocs != 0)
                {
                    int t = atomicAdd(&atomics.numTileSegs, numAllocs);
                    s_firstAllocSeg = (t + numAllocs <= p.maxTileSegs) ? t : 0;
                }
            }

            __syncthreads();
            int firstAllocSeg   = s_firstAllocSeg;
            int totalEmits      = s_tileEmitPrefixSum[CR_BIN_SQR];
            int totalAllocs     = s_tileAllocPrefixSum[CR_BIN_SQR];

            //------------------------------------------------------------------------
            // Emit.
            //------------------------------------------------------------------------

            // Emit per thread: Write triangle index to globalmem.

            for (int emitInBin = thrInBlock; emitInBin < totalEmits; emitInBin += CR_COARSE_WARPS * 32)
            {
                // Find tile in bin.

                U8* tileBase = (U8*)&s_tileEmitPrefixSum[0];
                U8* tilePtr = tileBase;
                U8* ptr;

                #if (CR_BIN_SQR > 128)
                    ptr = tilePtr + 0x80 * 4; if (emitInBin >= *(U32*)ptr) tilePtr = ptr;
                #endif
                #if (CR_BIN_SQR > 64)
                    ptr = tilePtr + 0x40 * 4; if (emitInBin >= *(U32*)ptr) tilePtr = ptr;
                #endif
                #if (CR_BIN_SQR > 32)
                    ptr = tilePtr + 0x20 * 4; if (emitInBin >= *(U32*)ptr) tilePtr = ptr;
                #endif
                #if (CR_BIN_SQR > 16)
                    ptr = tilePtr + 0x10 * 4; if (emitInBin >= *(U32*)ptr) tilePtr = ptr;
                #endif
                #if (CR_BIN_SQR > 8)
                    ptr = tilePtr + 0x08 * 4; if (emitInBin >= *(U32*)ptr) tilePtr = ptr;
                #endif
                #if (CR_BIN_SQR > 4)
                    ptr = tilePtr + 0x04 * 4; if (emitInBin >= *(U32*)ptr) tilePtr = ptr;
                #endif
                #if (CR_BIN_SQR > 2)
                    ptr = tilePtr + 0x02 * 4; if (emitInBin >= *(U32*)ptr) tilePtr = ptr;
                #endif
                #if (CR_BIN_SQR > 1)
                    ptr = tilePtr + 0x01 * 4; if (emitInBin >= *(U32*)ptr) tilePtr = ptr;
                #endif

                int tileInBin = (tilePtr - tileBase) >> 2;
                int emitInTile = emitInBin - *(U32*)tilePtr;

                // Find warp in tile.

                int warpStep = (CR_BIN_SQR + 1) * 4;
                U8* warpBase = (U8*)&s_warpEmitPrefixSum[0][tileInBin] - warpStep;
                U8* warpPtr = warpBase;

                #if (CR_COARSE_WARPS > 8)
                    ptr = warpPtr + 0x08 * warpStep; if (emitInTile >= *(U32*)ptr) warpPtr = ptr;
                #endif
                #if (CR_COARSE_WARPS > 4)
                    ptr = warpPtr + 0x04 * warpStep; if (emitInTile >= *(U32*)ptr) warpPtr = ptr;
                #endif
                #if (CR_COARSE_WARPS > 2)
                    ptr = warpPtr + 0x02 * warpStep; if (emitInTile >= *(U32*)ptr) warpPtr = ptr;
                #endif
                #if (CR_COARSE_WARPS > 1)
                    ptr = warpPtr + 0x01 * warpStep; if (emitInTile >= *(U32*)ptr) warpPtr = ptr;
                #endif

                int warpInTile = (warpPtr - warpBase) >> (CR_BIN_LOG2 * 2 + 2);
                U32 emitMask = *(U32*)(warpPtr + warpStep + ((U8*)s_warpEmitMask - (U8*)s_warpEmitPrefixSum));
                int emitInWarp = emitInTile - *(U32*)(warpPtr + warpStep) + __popc(emitMask);

                // Find thread in warp.

                int threadInWarp = 0;
                int pop = __popc(emitMask & 0xFFFF);
                bool pred = (emitInWarp >= pop);
                if (pred) emitInWarp -= pop;
                if (pred) emitMask >>= 0x10;
                if (pred) threadInWarp += 0x10;

                pop = __popc(emitMask & 0xFF);
                pred = (emitInWarp >= pop);
                if (pred) emitInWarp -= pop;
                if (pred) emitMask >>= 0x08;
                if (pred) threadInWarp += 0x08;

                pop = __popc(emitMask & 0xF);
                pred = (emitInWarp >= pop);
                if (pred) emitInWarp -= pop;
                if (pred) emitMask >>= 0x04;
                if (pred) threadInWarp += 0x04;

                pop = __popc(emitMask & 0x3);
                pred = (emitInWarp >= pop);
                if (pred) emitInWarp -= pop;
                if (pred) emitMask >>= 0x02;
                if (pred) threadInWarp += 0x02;

                if (emitInWarp >= (emitMask & 1))
                    threadInWarp++;

                // Figure out where to write.

                int currOfs = s_tileStreamCurrOfs[tileInBin];
                int spaceLeft = -currOfs & (CR_TILE_SEG_SIZE - 1);
                int outOfs = emitInTile;

                if (outOfs < spaceLeft)
                    outOfs += currOfs;
                else
                {
                    int allocLo = firstAllocSeg + s_tileAllocPrefixSum[tileInBin];
                    outOfs += (allocLo << CR_TILE_SEG_LOG2) - spaceLeft;
                }

                // Write.

                int queueIdx = warpInTile * 32 + threadInWarp;
                int triIdx = s_triQueue[(triQueueReadPos + queueIdx) & (CR_COARSE_QUEUE_SIZE - 1)];

                tileSegData[outOfs] = triIdx;
            }

            //------------------------------------------------------------------------
            // Patch.
            //------------------------------------------------------------------------

            // Allocated segment per thread: Initialize next-pointer and count.

            for (int i = CR_COARSE_WARPS * 32 - 1 - thrInBlock; i < totalAllocs; i += CR_COARSE_WARPS * 32)
            {
                int segIdx = firstAllocSeg + i;
                tileSegNext[segIdx] = segIdx + 1;
                tileSegCount[segIdx] = CR_TILE_SEG_SIZE;
            }

            // Tile per thread: Fix previous segment's next-pointer and update s_tileStreamCurrOfs.

            __syncthreads();
            for (int tileInBin = CR_COARSE_WARPS * 32 - 1 - thrInBlock; tileInBin < CR_BIN_SQR; tileInBin += CR_COARSE_WARPS * 32)
            {
                int oldOfs = s_tileStreamCurrOfs[tileInBin];
                int newOfs = oldOfs + s_warpEmitPrefixSum[CR_COARSE_WARPS - 1][tileInBin];
                int allocLo = s_tileAllocPrefixSum[tileInBin];
                int allocHi = s_tileAllocPrefixSum[tileInBin + 1];

                if (allocLo != allocHi)
                {
                    S32* nextPtr = &tileSegNext[(oldOfs - 1) >> CR_TILE_SEG_LOG2];
                    if (oldOfs < 0)
                        nextPtr = &tileFirstSeg[binTileIdx + globalTileIdx(tileInBin, p.widthTiles)];
                    *nextPtr = firstAllocSeg + allocLo;

                    newOfs--;
                    newOfs &= CR_TILE_SEG_SIZE - 1;
                    newOfs |= (firstAllocSeg + allocHi - 1) << CR_TILE_SEG_LOG2;
                    newOfs++;
                }
                s_tileStreamCurrOfs[tileInBin] = newOfs;
            }

            // Advance queue read pointer.
            // Queue became empty => bin done.

            triQueueReadPos += CR_COARSE_WARPS * 32;
        }
        while (triQueueReadPos < triQueueWritePos);

        // Tile per thread: Fix next-pointer and count of the last segment.
        // 32 tiles per warp: Count active tiles.

        __syncthreads();

        for (int tileInBin_base = 0; tileInBin_base < CR_BIN_SQR; tileInBin_base += CR_COARSE_WARPS * 32)
        {
            int tileInBin = tileInBin_base + thrInBlock;
            bool act = (tileInBin < CR_BIN_SQR);
            U32 actMask = __ballot_sync(~0u, act);
            if (act)
            {
                int tileX = tileInBin & (CR_BIN_SIZE - 1);
                int tileY = tileInBin >> CR_BIN_LOG2;
                bool force = (p.deferredClear & tileX <= maxTileXInBin & tileY <= maxTileYInBin);

                int ofs = s_tileStreamCurrOfs[tileInBin];
                int segIdx = (ofs - 1) >> CR_TILE_SEG_LOG2;
                int segCount = ofs & (CR_TILE_SEG_SIZE - 1);

                if (ofs >= 0)
                    tileSegNext[segIdx] = -1;
                else if (force)
                {
                    s_tileStreamCurrOfs[tileInBin] = 0;
                    tileFirstSeg[binTileIdx + tileX + tileY * p.widthTiles] = -1;
                }

                if (segCount != 0)
                    tileSegCount[segIdx] = segCount;

                U32 res = __ballot_sync(actMask, ofs >= 0 | force);
                if (threadIdx.x == 0)
                    s_scanTemp[0][(tileInBin >> 5) + 16] = __popc(res);
            }
        }

        // First warp: Scan-8.
        // One thread: Allocate space for active tiles.

        __syncthreads();

        bool scan8 = (thrInBlock < CR_BIN_SQR / 32);
        U32 scan8Mask = __ballot_sync(~0u, scan8);
        if (scan8)
        {
            volatile U32* v = &s_scanTemp[0][thrInBlock + 16];
            U32 sum = v[0];
            #if (CR_BIN_SQR > 1 * 32)
                sum += v[-1]; __syncwarp(scan8Mask); v[0] = sum; __syncwarp(scan8Mask);
            #endif
            #if (CR_BIN_SQR > 2 * 32)
                sum += v[-2]; __syncwarp(scan8Mask); v[0] = sum; __syncwarp(scan8Mask);
            #endif
            #if (CR_BIN_SQR > 4 * 32)
                sum += v[-4]; __syncwarp(scan8Mask); v[0] = sum; __syncwarp(scan8Mask);
            #endif

            if (thrInBlock == CR_BIN_SQR / 32 - 1)
                s_firstActiveIdx = atomicAdd(&atomics.numActiveTiles, sum);
        }

        // Tile per thread: Output active tiles.

        __syncthreads();

        for (int tileInBin_base = 0; tileInBin_base < CR_BIN_SQR; tileInBin_base += CR_COARSE_WARPS * 32)
        {
            int tileInBin = tileInBin_base + thrInBlock;
            bool act = (tileInBin < CR_BIN_SQR) && (s_tileStreamCurrOfs[tileInBin] >= 0);
            U32 actMask = __ballot_sync(~0u, act);
            if (act)
            {
                int activeIdx = s_firstActiveIdx;
                activeIdx += s_scanTemp[0][(tileInBin >> 5) + 15];
                activeIdx += __popc(actMask & getLaneMaskLt());
                activeTiles[activeIdx] = binTileIdx + globalTileIdx(tileInBin, p.widthTiles);
            }
        }
    }
}

//------------------------------------------------------------------------
