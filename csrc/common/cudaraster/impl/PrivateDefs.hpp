// Copyright (c) 2009-2022, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once
#include "Defs.hpp"
#include "Constants.hpp"

namespace CR
{
//------------------------------------------------------------------------
// Projected triangle.
//------------------------------------------------------------------------

struct CRTriangleHeader
{
    S16 v0x;    // Subpixels relative to viewport center. Valid if triSubtris = 1.
    S16 v0y;
    S16 v1x;
    S16 v1y;
    S16 v2x;
    S16 v2y;

    U32 misc;   // triSubtris=1: (zmin:20, f01:4, f12:4, f20:4), triSubtris>=2: (subtriBase)
};

//------------------------------------------------------------------------

struct CRTriangleData
{
    U32 zx;     // zx * sampleX + zy * sampleY + zb = lerp(CR_DEPTH_MIN, CR_DEPTH_MAX, (clipZ / clipW + 1) / 2)
    U32 zy;
    U32 zb;
    U32 id;     // Triangle id.
};

//------------------------------------------------------------------------
// Device-side structures.
//------------------------------------------------------------------------

struct CRAtomics
{
    // Setup.
    S32         numSubtris;         // = numTris

    // Bin.
    S32         binCounter;         // = 0
    S32         numBinSegs;         // = 0

    // Coarse.
    S32         coarseCounter;      // = 0
    S32         numTileSegs;        // = 0
    S32         numActiveTiles;     // = 0

    // Fine.
    S32         fineCounter;        // = 0
};

//------------------------------------------------------------------------

struct CRImageParams
{
    S32         triOffset;          // First triangle index to draw.
    S32         triCount;           // Number of triangles to draw.
    S32         binBatchSize;       // Number of triangles per batch.
};

//------------------------------------------------------------------------

struct CRParams
{
    // Common.

    CRAtomics*  atomics;            // Work counters. Per-image.
    S32         numImages;          // Batch size.
    S32         totalCount;         // In range mode, total number of triangles to render.
    S32         instanceMode;       // 0 = range mode, 1 = instance mode.

    S32         numVertices;        // Number of vertices in input buffer, not counting multiples in instance mode.
    S32         numTriangles;       // Number of triangles in input buffer.
    void*       vertexBuffer;       // numVertices * float4(x, y, z, w)
    void*       indexBuffer;        // numTriangles * int3(vi0, vi1, vi2)

    S32         widthPixels;        // Render buffer size in pixels. Must be multiple of tile size (8x8).
    S32         heightPixels;
    S32         widthPixelsVp;      // Viewport size in pixels.
    S32         heightPixelsVp;
    S32         widthBins;          // widthPixels / CR_BIN_SIZE
    S32         heightBins;         // heightPixels / CR_BIN_SIZE
    S32         numBins;            // widthBins * heightBins

    F32         xs;                 // Vertex position adjustments for tiled rendering.
    F32         ys;
    F32         xo;
    F32         yo;

    S32         widthTiles;         // widthPixels / CR_TILE_SIZE
    S32         heightTiles;        // heightPixels / CR_TILE_SIZE
    S32         numTiles;           // widthTiles * heightTiles

    U32         renderModeFlags;
    S32         deferredClear;      // 1 = Clear framebuffer before rendering triangles.
    U32         clearColor;
    U32         clearDepth;

    // These are uniform across batch.

    S32         maxSubtris;
    S32         maxBinSegs;
    S32         maxTileSegs;

    // Setup output / bin input.

    void*       triSubtris;         // maxSubtris * U8
    void*       triHeader;          // maxSubtris * CRTriangleHeader
    void*       triData;            // maxSubtris * CRTriangleData

    // Bin output / coarse input.

    void*       binSegData;         // maxBinSegs * CR_BIN_SEG_SIZE * S32
    void*       binSegNext;         // maxBinSegs * S32
    void*       binSegCount;        // maxBinSegs * S32
    void*       binFirstSeg;        // CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * (S32 segIdx), -1 = none
    void*       binTotal;           // CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * (S32 numTris)

    // Coarse output / fine input.

    void*       tileSegData;        // maxTileSegs * CR_TILE_SEG_SIZE * S32
    void*       tileSegNext;        // maxTileSegs * S32
    void*       tileSegCount;       // maxTileSegs * S32
    void*       activeTiles;        // CR_MAXTILES_SQR * (S32 tileIdx)
    void*       tileFirstSeg;       // CR_MAXTILES_SQR * (S32 segIdx), -1 = none

    // Surface buffers. Outer tile offset is baked into pointers.

    void*       colorBuffer;        // sizePixels.x * sizePixels.y * numImages * U32
    void*       depthBuffer;        // sizePixels.x * sizePixels.y * numImages * U32
    void*       peelBuffer;         // sizePixels.x * sizePixels.y * numImages * U32, only if peeling enabled.
    S32         strideX;            // horizontal size in pixels
    S32         strideY;            // vertical stride in pixels

    // Per-image parameters for first images are embedded here to avoid extra memcpy for small batches.

    CRImageParams imageParamsFirst[CR_EMBED_IMAGE_PARAMS];
    const CRImageParams* imageParamsExtra; // After CR_EMBED_IMAGE_PARAMS.
};

//------------------------------------------------------------------------
}
