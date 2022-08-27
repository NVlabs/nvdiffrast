// Copyright (c) 2009-2022, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

//------------------------------------------------------------------------

__device__ __inline__ void binRasterImpl(const CRParams p)
{
    __shared__ volatile U32 s_broadcast [CR_BIN_WARPS + 16];
    __shared__ volatile S32 s_outOfs    [CR_MAXBINS_SQR];
    __shared__ volatile S32 s_outTotal  [CR_MAXBINS_SQR];
    __shared__ volatile S32 s_overIndex [CR_MAXBINS_SQR];
    __shared__ volatile S32 s_outMask   [CR_BIN_WARPS][CR_MAXBINS_SQR + 1]; // +1 to avoid bank collisions
    __shared__ volatile S32 s_outCount  [CR_BIN_WARPS][CR_MAXBINS_SQR + 1]; // +1 to avoid bank collisions
    __shared__ volatile S32 s_triBuf    [CR_BIN_WARPS*32*4];                // triangle ring buffer
    __shared__ volatile U32 s_batchPos;
    __shared__ volatile U32 s_bufCount;
    __shared__ volatile U32 s_overTotal;
    __shared__ volatile U32 s_allocBase;

    const CRImageParams&    ip              = getImageParams(p, blockIdx.z);
    CRAtomics&              atomics         = p.atomics[blockIdx.z];
    const U8*               triSubtris      = (const U8*)p.triSubtris + p.maxSubtris * blockIdx.z;
    const CRTriangleHeader* triHeader       = (const CRTriangleHeader*)p.triHeader + p.maxSubtris * blockIdx.z;

    S32*                    binFirstSeg     = (S32*)p.binFirstSeg + CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * blockIdx.z;
    S32*                    binTotal        = (S32*)p.binTotal    + CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * blockIdx.z;
    S32*                    binSegData      = (S32*)p.binSegData  + p.maxBinSegs * CR_BIN_SEG_SIZE * blockIdx.z;
    S32*                    binSegNext      = (S32*)p.binSegNext  + p.maxBinSegs * blockIdx.z;
    S32*                    binSegCount     = (S32*)p.binSegCount + p.maxBinSegs * blockIdx.z;

    if (atomics.numSubtris > p.maxSubtris)
        return;

    // per-thread state
    int thrInBlock = threadIdx.x + threadIdx.y * 32;
    int batchPos = 0;

    // first 16 elements of s_broadcast are always zero
    if (thrInBlock < 16)
        s_broadcast[thrInBlock] = 0;

    // initialize output linked lists and offsets
    if (thrInBlock < p.numBins)
    {
        binFirstSeg[(thrInBlock << CR_BIN_STREAMS_LOG2) + blockIdx.x] = -1;
        s_outOfs[thrInBlock] = -CR_BIN_SEG_SIZE;
        s_outTotal[thrInBlock] = 0;
    }

    // repeat until done
    for(;;)
    {
        // get batch
        if (thrInBlock == 0)
            s_batchPos = atomicAdd(&atomics.binCounter, ip.binBatchSize);
        __syncthreads();
        batchPos = s_batchPos;

        // all batches done?
        if (batchPos >= ip.triCount)
            break;

        // per-thread state
        int bufIndex = 0;
        int bufCount = 0;
        int batchEnd = min(batchPos + ip.binBatchSize, ip.triCount);

        // loop over batch as long as we have triangles in it
        do
        {
            // read more triangles
            while (bufCount < CR_BIN_WARPS*32 && batchPos < batchEnd)
            {
                // get subtriangle count

                int triIdx = batchPos + thrInBlock;
                int num = 0;
                if (triIdx < batchEnd)
                    num = triSubtris[triIdx];

                // cumulative sum of subtriangles within each warp
                U32 myIdx = __popc(__ballot_sync(~0u, num & 1) & getLaneMaskLt());
                if (__any_sync(~0u, num > 1))
                {
                    myIdx += __popc(__ballot_sync(~0u, num & 2) & getLaneMaskLt()) * 2;
                    myIdx += __popc(__ballot_sync(~0u, num & 4) & getLaneMaskLt()) * 4;
                }
                if (threadIdx.x == 31) // Do not assume that last thread in warp wins the write.
                    s_broadcast[threadIdx.y + 16] = myIdx + num;
                __syncthreads();

                // cumulative sum of per-warp subtriangle counts
                // Note: cannot have more than 32 warps or this needs to sync between each step.
                bool act = (thrInBlock < CR_BIN_WARPS);
                U32 actMask = __ballot_sync(~0u, act);
                if (threadIdx.y == 0 && act)
                {
                    volatile U32* ptr = &s_broadcast[thrInBlock + 16];
                    U32 val = *ptr;
                    #if (CR_BIN_WARPS > 1)
                        val += ptr[-1]; __syncwarp(actMask);
                        *ptr = val;     __syncwarp(actMask);
                    #endif
                    #if (CR_BIN_WARPS > 2)
                        val += ptr[-2]; __syncwarp(actMask);
                        *ptr = val;     __syncwarp(actMask);
                    #endif
                    #if (CR_BIN_WARPS > 4)
                        val += ptr[-4]; __syncwarp(actMask);
                        *ptr = val;     __syncwarp(actMask);
                    #endif
                    #if (CR_BIN_WARPS > 8)
                        val += ptr[-8]; __syncwarp(actMask);
                        *ptr = val;     __syncwarp(actMask);
                    #endif
                    #if (CR_BIN_WARPS > 16)
                        val += ptr[-16]; __syncwarp(actMask);
                        *ptr = val;      __syncwarp(actMask);
                    #endif

                    // initially assume that we consume everything
                    // only last active thread does the writes
                    if (threadIdx.x == CR_BIN_WARPS - 1)
                    {
                        s_batchPos = batchPos + CR_BIN_WARPS * 32;
                        s_bufCount = bufCount + val;
                    }
                }
                __syncthreads();

                // skip if no subtriangles
                if (num)
                {
                    // calculate write position for first subtriangle
                    U32 pos = bufCount + myIdx + s_broadcast[threadIdx.y + 16 - 1];

                    // only write if entire triangle fits
                    if (pos + num <= CR_ARRAY_SIZE(s_triBuf))
                    {
                        pos += bufIndex; // adjust for current start position
                        pos &= CR_ARRAY_SIZE(s_triBuf)-1;
                        if (num == 1)
                            s_triBuf[pos] = triIdx * 8 + 7; // single triangle
                        else
                        {
                            for (int i=0; i < num; i++)
                            {
                                s_triBuf[pos] = triIdx * 8 + i;
                                pos++;
                                pos &= CR_ARRAY_SIZE(s_triBuf)-1;
                            }
                        }
                    } else if (pos <= CR_ARRAY_SIZE(s_triBuf))
                    {
                        // this triangle is the first that failed, overwrite total count and triangle count
                        s_batchPos = batchPos + thrInBlock;
                        s_bufCount = pos;
                    }
                }

                // update triangle counts
                __syncthreads();
                batchPos = s_batchPos;
                bufCount = s_bufCount;
            }

            // make every warp clear its output buffers
            for (int i=threadIdx.x; i < p.numBins; i += 32)
                s_outMask[threadIdx.y][i] = 0;
            __syncwarp();

            // choose our triangle
            uint4 triData = make_uint4(0, 0, 0, 0);
            if (thrInBlock < bufCount)
            {
                U32 triPos = bufIndex + thrInBlock;
                triPos &= CR_ARRAY_SIZE(s_triBuf)-1;

                // find triangle
                int triIdx = s_triBuf[triPos];
                int dataIdx = triIdx >> 3;
                int subtriIdx = triIdx & 7;
                if (subtriIdx != 7)
                    dataIdx = triHeader[dataIdx].misc + subtriIdx;

                // read triangle

                triData = *(((const uint4*)triHeader) + dataIdx);
            }

            // setup bounding box and edge functions, and rasterize
            S32 lox, loy, hix, hiy;
            bool hasTri = (thrInBlock < bufCount);
            U32 hasTriMask = __ballot_sync(~0u, hasTri);
            if (hasTri)
            {
                S32 v0x = add_s16lo_s16lo(triData.x, p.widthPixels  * (CR_SUBPIXEL_SIZE >> 1));
                S32 v0y = add_s16hi_s16lo(triData.x, p.heightPixels * (CR_SUBPIXEL_SIZE >> 1));
                S32 d01x = sub_s16lo_s16lo(triData.y, triData.x);
                S32 d01y = sub_s16hi_s16hi(triData.y, triData.x);
                S32 d02x = sub_s16lo_s16lo(triData.z, triData.x);
                S32 d02y = sub_s16hi_s16hi(triData.z, triData.x);
                int binLog = CR_BIN_LOG2 + CR_TILE_LOG2 + CR_SUBPIXEL_LOG2;
                lox = add_clamp_0_x((v0x + min_min(d01x, 0, d02x)) >> binLog, 0, p.widthBins  - 1);
                loy = add_clamp_0_x((v0y + min_min(d01y, 0, d02y)) >> binLog, 0, p.heightBins - 1);
                hix = add_clamp_0_x((v0x + max_max(d01x, 0, d02x)) >> binLog, 0, p.widthBins  - 1);
                hiy = add_clamp_0_x((v0y + max_max(d01y, 0, d02y)) >> binLog, 0, p.heightBins - 1);

                U32 bit = 1 << threadIdx.x;
#if __CUDA_ARCH__ >= 700
                bool multi = (hix != lox || hiy != loy);
                if (!__any_sync(hasTriMask, multi))
                {
                    int binIdx = lox + p.widthBins * loy;
                    U32 mask = __match_any_sync(hasTriMask, binIdx);
                    s_outMask[threadIdx.y][binIdx] = mask;
                    __syncwarp(hasTriMask);
                } else
#endif
                {
                    bool complex = (hix > lox+1 || hiy > loy+1);
                    if (!__any_sync(hasTriMask, complex))
                    {
                        int binIdx = lox + p.widthBins * loy;
                        atomicOr((U32*)&s_outMask[threadIdx.y][binIdx], bit);
                        if (hix > lox) atomicOr((U32*)&s_outMask[threadIdx.y][binIdx + 1], bit);
                        if (hiy > loy) atomicOr((U32*)&s_outMask[threadIdx.y][binIdx + p.widthBins], bit);
                        if (hix > lox && hiy > loy) atomicOr((U32*)&s_outMask[threadIdx.y][binIdx + p.widthBins + 1], bit);
                    } else
                    {
                        S32 d12x = d02x - d01x, d12y = d02y - d01y;
                        v0x -= lox << binLog, v0y -= loy << binLog;

                        S32 t01 = v0x * d01y - v0y * d01x;
                        S32 t02 = v0y * d02x - v0x * d02y;
                        S32 t12 = d01x * d12y - d01y * d12x - t01 - t02;
                        S32 b01 = add_sub(t01 >> binLog, max(d01x, 0), min(d01y, 0));
                        S32 b02 = add_sub(t02 >> binLog, max(d02y, 0), min(d02x, 0));
                        S32 b12 = add_sub(t12 >> binLog, max(d12x, 0), min(d12y, 0));

                        int width = hix - lox + 1;
                        d01x += width * d01y;
                        d02x += width * d02y;
                        d12x += width * d12y;

                        U8* currPtr = (U8*)&s_outMask[threadIdx.y][lox + loy * p.widthBins];
                        U8* skipPtr = (U8*)&s_outMask[threadIdx.y][(hix + 1) + loy * p.widthBins];
                        U8* endPtr  = (U8*)&s_outMask[threadIdx.y][lox + (hiy + 1) * p.widthBins];
                        int stride  = p.widthBins * 4;
                        int ptrYInc = stride - width * 4;

                        do
                        {
                            if (b01 >= 0 && b02 >= 0 && b12 >= 0)
                                atomicOr((U32*)currPtr, bit);
                            currPtr += 4, b01 -= d01y, b02 += d02y, b12 -= d12y;
                            if (currPtr == skipPtr)
                                currPtr += ptrYInc, b01 += d01x, b02 -= d02x, b12 += d12x, skipPtr += stride;
                        }
                        while (currPtr != endPtr);
                    }
                }
            }

            // count per-bin contributions
            if (thrInBlock == 0)
                s_overTotal = 0; // overflow counter

            // ensure that out masks are done
            __syncthreads();

            int overIndex = -1;
            bool act = (thrInBlock < p.numBins);
            U32 actMask = __ballot_sync(~0u, act);
            if (act)
            {
                U8* srcPtr = (U8*)&s_outMask[0][thrInBlock];
                U8* dstPtr = (U8*)&s_outCount[0][thrInBlock];
                int total = 0;
                for (int i = 0; i < CR_BIN_WARPS; i++)
                {
                    total += __popc(*(U32*)srcPtr);
                    *(U32*)dstPtr = total;
                    srcPtr += (CR_MAXBINS_SQR + 1) * 4;
                    dstPtr += (CR_MAXBINS_SQR + 1) * 4;
                }

                // overflow => request a new segment
                int ofs = s_outOfs[thrInBlock];
                bool ovr = (((ofs - 1) >> CR_BIN_SEG_LOG2) != (((ofs - 1) + total) >> CR_BIN_SEG_LOG2));
                U32 ovrMask = __ballot_sync(actMask, ovr);
                if (ovr)
                {
                    overIndex = __popc(ovrMask & getLaneMaskLt());
                    if (overIndex == 0)
                        s_broadcast[threadIdx.y + 16] = atomicAdd((U32*)&s_overTotal, __popc(ovrMask));
                    __syncwarp(ovrMask);
                    overIndex += s_broadcast[threadIdx.y + 16];
                    s_overIndex[thrInBlock] = overIndex;
                }
            }

            // sync after overTotal is ready
            __syncthreads();

            // at least one segment overflowed => allocate segments
            U32 overTotal = s_overTotal;
            U32 allocBase = 0;
            if (overTotal > 0)
            {
                // allocate memory
                if (thrInBlock == 0)
                {
                    U32 allocBase = atomicAdd(&atomics.numBinSegs, overTotal);
                    s_allocBase = (allocBase + overTotal <= p.maxBinSegs) ? allocBase : 0;
                }
                __syncthreads();
                allocBase = s_allocBase;

                // did my bin overflow?
                if (overIndex != -1)
                {
                    // calculate new segment index
                    int segIdx = allocBase + overIndex;

                    // add to linked list
                    if (s_outOfs[thrInBlock] < 0)
                        binFirstSeg[(thrInBlock << CR_BIN_STREAMS_LOG2) + blockIdx.x] = segIdx;
                    else
                        binSegNext[(s_outOfs[thrInBlock] - 1) >> CR_BIN_SEG_LOG2] = segIdx;

                    // defaults
                    binSegNext [segIdx] = -1;
                    binSegCount[segIdx] = CR_BIN_SEG_SIZE;
                }
            }

            // concurrent emission -- each warp handles its own triangle
            if (thrInBlock < bufCount)
            {
                int triPos  = (bufIndex + thrInBlock) & (CR_ARRAY_SIZE(s_triBuf) - 1);
                int currBin = lox + loy * p.widthBins;
                int skipBin = (hix + 1) + loy * p.widthBins;
                int endBin  = lox + (hiy + 1) * p.widthBins;
                int binYInc = p.widthBins - (hix - lox + 1);

                // loop over triangle's bins
                do
                {
                    U32 outMask = s_outMask[threadIdx.y][currBin];
                    if (outMask & (1<<threadIdx.x))
                    {
                        int idx = __popc(outMask & getLaneMaskLt());
                        if (threadIdx.y > 0)
                            idx += s_outCount[threadIdx.y-1][currBin];

                        int base = s_outOfs[currBin];
                        int free = (-base) & (CR_BIN_SEG_SIZE - 1);
                        if (idx >= free)
                            idx += ((allocBase + s_overIndex[currBin]) << CR_BIN_SEG_LOG2) - free;
                        else
                            idx += base;

                        binSegData[idx] = s_triBuf[triPos];
                    }

                    currBin++;
                    if (currBin == skipBin)
                        currBin += binYInc, skipBin += p.widthBins;
                }
                while (currBin != endBin);
            }

            // wait all triangles to finish, then replace overflown segment offsets
            __syncthreads();
            if (thrInBlock < p.numBins)
            {
                U32 total  = s_outCount[CR_BIN_WARPS - 1][thrInBlock];
                U32 oldOfs = s_outOfs[thrInBlock];
                if (overIndex == -1)
                    s_outOfs[thrInBlock] = oldOfs + total;
                else
                {
                    int addr = oldOfs + total;
                    addr = ((addr - 1) & (CR_BIN_SEG_SIZE - 1)) + 1;
                    addr += (allocBase + overIndex) << CR_BIN_SEG_LOG2;
                    s_outOfs[thrInBlock] = addr;
                }
                s_outTotal[thrInBlock] += total;
            }

            // these triangles are now done
            int count = ::min(bufCount, CR_BIN_WARPS * 32);
            bufCount -= count;
            bufIndex += count;
            bufIndex &= CR_ARRAY_SIZE(s_triBuf)-1;
        }
        while (bufCount > 0 || batchPos < batchEnd);

        // flush all bins
        if (thrInBlock < p.numBins)
        {
            int ofs = s_outOfs[thrInBlock];
            if (ofs & (CR_BIN_SEG_SIZE-1))
            {
                int seg = ofs >> CR_BIN_SEG_LOG2;
                binSegCount[seg] = ofs & (CR_BIN_SEG_SIZE-1);
                s_outOfs[thrInBlock] = (ofs + CR_BIN_SEG_SIZE - 1) & -CR_BIN_SEG_SIZE;
            }
        }
    }

    // output totals
    if (thrInBlock < p.numBins)
        binTotal[(thrInBlock << CR_BIN_STREAMS_LOG2) + blockIdx.x] = s_outTotal[thrInBlock];
}

//------------------------------------------------------------------------
