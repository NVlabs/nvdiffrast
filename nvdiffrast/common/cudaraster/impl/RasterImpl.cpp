// Copyright (c) 2009-2022, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "../../framework.h"
#include "PrivateDefs.hpp"
#include "Constants.hpp"
#include "RasterImpl.hpp"
#include <cuda_runtime.h>

using namespace CR;
using std::min;
using std::max;

//------------------------------------------------------------------------
// Kernel prototypes and variables.

void triangleSetupKernel (const CRParams p);
void binRasterKernel     (const CRParams p);
void coarseRasterKernel  (const CRParams p);
void fineRasterKernel    (const CRParams p);

//------------------------------------------------------------------------

RasterImpl::RasterImpl(void)
:   m_renderModeFlags       (0),
    m_deferredClear         (false),
    m_clearColor            (0),
    m_vertexPtr             (NULL),
    m_indexPtr              (NULL),
    m_numVertices           (0),
    m_numTriangles          (0),
    m_bufferSizesReported   (0),

    m_numImages             (0),
    m_sizePixels            (0, 0),
    m_sizeBins              (0, 0),
    m_numBins               (0),
    m_sizeTiles             (0, 0),
    m_numTiles              (0),

    m_numSMs                (1),
    m_numCoarseBlocksPerSM  (1),
    m_numFineBlocksPerSM    (1),
    m_numFineWarpsPerBlock  (1),

    m_maxSubtris            (1),
    m_maxBinSegs            (1),
    m_maxTileSegs           (1)
{
    // Query relevant device attributes.

    int currentDevice = 0;
    NVDR_CHECK_CUDA_ERROR(cudaGetDevice(&currentDevice));
    NVDR_CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&m_numSMs, cudaDevAttrMultiProcessorCount, currentDevice));
    cudaFuncAttributes attr;
    NVDR_CHECK_CUDA_ERROR(cudaFuncGetAttributes(&attr, (void*)fineRasterKernel));
    m_numFineWarpsPerBlock = min(attr.maxThreadsPerBlock / 32, CR_FINE_MAX_WARPS);
    NVDR_CHECK_CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&m_numCoarseBlocksPerSM, (void*)coarseRasterKernel, 32 * CR_COARSE_WARPS, 0));
    NVDR_CHECK_CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&m_numFineBlocksPerSM, (void*)fineRasterKernel, 32 * m_numFineWarpsPerBlock, 0));

    // Setup functions.

    NVDR_CHECK_CUDA_ERROR(cudaFuncSetCacheConfig((void*)triangleSetupKernel, cudaFuncCachePreferShared));
    NVDR_CHECK_CUDA_ERROR(cudaFuncSetCacheConfig((void*)binRasterKernel,     cudaFuncCachePreferShared));
    NVDR_CHECK_CUDA_ERROR(cudaFuncSetCacheConfig((void*)coarseRasterKernel,  cudaFuncCachePreferShared));
    NVDR_CHECK_CUDA_ERROR(cudaFuncSetCacheConfig((void*)fineRasterKernel,    cudaFuncCachePreferShared));
}

//------------------------------------------------------------------------

RasterImpl::~RasterImpl(void)
{
    // Empty.
}

//------------------------------------------------------------------------

void RasterImpl::setViewportSize(Vec3i size)
{
    if ((size.x | size.y) & (CR_TILE_SIZE - 1))
        return; // Invalid size.

    m_numImages     = size.z;
    m_sizePixels    = Vec2i(size.x, size.y);
    m_sizeTiles.x   = m_sizePixels.x >> CR_TILE_LOG2;
    m_sizeTiles.y   = m_sizePixels.y >> CR_TILE_LOG2;
    m_numTiles      = m_sizeTiles.x * m_sizeTiles.y;
    m_sizeBins.x    = (m_sizeTiles.x + CR_BIN_SIZE - 1) >> CR_BIN_LOG2;
    m_sizeBins.y    = (m_sizeTiles.y + CR_BIN_SIZE - 1) >> CR_BIN_LOG2;
    m_numBins       = m_sizeBins.x * m_sizeBins.y;

    m_colorBuffer.reset(m_sizePixels.x * m_sizePixels.y * m_numImages * sizeof(U32));
    m_depthBuffer.reset(m_sizePixels.x * m_sizePixels.y * m_numImages * sizeof(U32));
}

void RasterImpl::swapDepthAndPeel(void)
{
    m_peelBuffer.reset(m_depthBuffer.getSize()); // Ensure equal size and valid pointer.

    void* tmp = m_depthBuffer.getPtr();
    m_depthBuffer.setPtr(m_peelBuffer.getPtr());
    m_peelBuffer.setPtr(tmp);
}

//------------------------------------------------------------------------

bool RasterImpl::drawTriangles(const Vec2i* ranges, bool peel, cudaStream_t stream)
{
    bool instanceMode = (!ranges);

    int maxSubtrisSlack     = 4096;     // x 81B    = 324KB
    int maxBinSegsSlack     = 256;      // x 2137B  = 534KB
    int maxTileSegsSlack    = 4096;     // x 136B   = 544KB

    // Resize atomics as needed.
    m_crAtomics    .grow(m_numImages * sizeof(CRAtomics));
    m_crAtomicsHost.grow(m_numImages * sizeof(CRAtomics));

    // Size of these buffers doesn't depend on input.
    m_binFirstSeg  .grow(m_numImages * CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * sizeof(S32));
    m_binTotal     .grow(m_numImages * CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * sizeof(S32));
    m_activeTiles  .grow(m_numImages * CR_MAXTILES_SQR * sizeof(S32));
    m_tileFirstSeg .grow(m_numImages * CR_MAXTILES_SQR * sizeof(S32));

    // Construct per-image parameters and determine worst-case buffer sizes.
    m_crImageParamsHost.grow(m_numImages * sizeof(CRImageParams));
    CRImageParams* imageParams = (CRImageParams*)m_crImageParamsHost.getPtr();
    for (int i=0; i < m_numImages; i++)
    {
        CRImageParams& ip = imageParams[i];

        int roundSize  = CR_BIN_WARPS * 32;
        int minBatches = CR_BIN_STREAMS_SIZE * 2;
        int maxRounds  = 32;

        ip.triOffset = instanceMode ? 0 : ranges[i].x;
        ip.triCount  = instanceMode ? m_numTriangles : ranges[i].y;
        ip.binBatchSize = min(max(ip.triCount / (roundSize * minBatches), 1), maxRounds) * roundSize;

        m_maxSubtris  = max(m_maxSubtris,  min(ip.triCount + maxSubtrisSlack, CR_MAXSUBTRIS_SIZE));
        m_maxBinSegs  = max(m_maxBinSegs,  max(m_numBins * CR_BIN_STREAMS_SIZE, (ip.triCount - 1) / CR_BIN_SEG_SIZE + 1) + maxBinSegsSlack);
        m_maxTileSegs = max(m_maxTileSegs, max(m_numTiles, (ip.triCount - 1) / CR_TILE_SEG_SIZE + 1) + maxTileSegsSlack);
    }

    // Retry until successful.

    for (;;)
    {
        // Allocate buffers.
        m_triSubtris.reset(m_numImages * m_maxSubtris * sizeof(U8));
        m_triHeader .reset(m_numImages * m_maxSubtris * sizeof(CRTriangleHeader));
        m_triData   .reset(m_numImages * m_maxSubtris * sizeof(CRTriangleData));

        m_binSegData .reset(m_numImages * m_maxBinSegs * CR_BIN_SEG_SIZE * sizeof(S32));
        m_binSegNext .reset(m_numImages * m_maxBinSegs * sizeof(S32));
        m_binSegCount.reset(m_numImages * m_maxBinSegs * sizeof(S32));

        m_tileSegData .reset(m_numImages * m_maxTileSegs * CR_TILE_SEG_SIZE * sizeof(S32));
        m_tileSegNext .reset(m_numImages * m_maxTileSegs * sizeof(S32));
        m_tileSegCount.reset(m_numImages * m_maxTileSegs * sizeof(S32));

        // Report if buffers grow from last time.
        size_t sizesTotal = getTotalBufferSizes();
        if (sizesTotal > m_bufferSizesReported)
        {
            size_t sizesMB = ((sizesTotal - 1) >> 20) + 1; // Round up.
            sizesMB = ((sizesMB + 9) / 10) * 10; // 10MB granularity enough in this day and age.
            LOG(INFO) << "Internal buffers grown to " << sizesMB << " MB";
            m_bufferSizesReported = sizesMB << 20;
        }

        // Launch stages. Blocks until everything is done.
        launchStages(instanceMode, peel, stream);

        // Peeling iteration cannot fail, so no point checking things further.
        if (peel)
            break;

        // Atomics after coarse stage are now available.
        CRAtomics* atomics = (CRAtomics*)m_crAtomicsHost.getPtr();

        // Success?
        bool failed = false;
        for (int i=0; i < m_numImages; i++)
        {
            const CRAtomics& a = atomics[i];
            failed = failed || (a.numSubtris > m_maxSubtris) || (a.numBinSegs > m_maxBinSegs) || (a.numTileSegs > m_maxTileSegs);
        }
        if (!failed)
            break; // Success!

        // If we were already at maximum capacity, no can do.
        if (m_maxSubtris == CR_MAXSUBTRIS_SIZE)
            return false;

        // Enlarge buffers and try again.
        for (int i=0; i < m_numImages; i++)
        {
            const CRAtomics& a = atomics[i];
            m_maxSubtris  = max(m_maxSubtris,  min(a.numSubtris + maxSubtrisSlack, CR_MAXSUBTRIS_SIZE));
            m_maxBinSegs  = max(m_maxBinSegs,  a.numBinSegs + maxBinSegsSlack);
            m_maxTileSegs = max(m_maxTileSegs, a.numTileSegs + maxTileSegsSlack);
        }
    }

    m_deferredClear = false;
    return true; // Success.
}

//------------------------------------------------------------------------

size_t RasterImpl::getTotalBufferSizes(void) const
{
    return
        m_colorBuffer.getSize() + m_depthBuffer.getSize() + // Don't include atomics and image params.
        m_triSubtris.getSize() + m_triHeader.getSize() + m_triData.getSize() +
        m_binFirstSeg.getSize() + m_binTotal.getSize() + m_binSegData.getSize() + m_binSegNext.getSize() + m_binSegCount.getSize() +
        m_activeTiles.getSize() + m_tileFirstSeg.getSize() + m_tileSegData.getSize() + m_tileSegNext.getSize() + m_tileSegCount.getSize();
}

//------------------------------------------------------------------------

void RasterImpl::launchStages(bool instanceMode, bool peel, cudaStream_t stream)
{
    CRImageParams* imageParams = (CRImageParams*)m_crImageParamsHost.getPtr();

    // Unless peeling, initialize atomics to mostly zero.
    CRAtomics* atomics = (CRAtomics*)m_crAtomicsHost.getPtr();
    if (!peel)
    {
        memset(atomics, 0, m_numImages * sizeof(CRAtomics));
        for (int i=0; i < m_numImages; i++)
            atomics[i].numSubtris = imageParams[i].triCount;
    }

    // Copy to device. If peeling, this is the state after coarse raster launch on first iteration.
    NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(m_crAtomics.getPtr(), atomics, m_numImages * sizeof(CRAtomics), cudaMemcpyHostToDevice, stream));

    // Copy per-image parameters if there are more than fits in launch parameter block and we haven't done it already.
    if (!peel && m_numImages > CR_EMBED_IMAGE_PARAMS)
    {
        int numImageParamsExtra = m_numImages - CR_EMBED_IMAGE_PARAMS;
        m_crImageParamsExtra.grow(numImageParamsExtra * sizeof(CRImageParams));
        NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(m_crImageParamsExtra.getPtr(), imageParams + CR_EMBED_IMAGE_PARAMS, numImageParamsExtra * sizeof(CRImageParams), cudaMemcpyHostToDevice, stream));
    }

    // Set global parameters.
    CRParams p;
    {
        p.atomics           = (CRAtomics*)m_crAtomics.getPtr();
        p.numImages         = m_numImages;
        p.totalCount        = 0; // Only relevant in range mode.
        p.instanceMode      = instanceMode ? 1 : 0;

        p.numVertices       = m_numVertices;
        p.numTriangles      = m_numTriangles;
        p.vertexBuffer      = m_vertexPtr;
        p.indexBuffer       = m_indexPtr;

        p.widthPixels       = m_sizePixels.x;
        p.heightPixels      = m_sizePixels.y;
        p.widthBins         = m_sizeBins.x;
        p.heightBins        = m_sizeBins.y;
        p.numBins           = m_numBins;

        p.widthTiles        = m_sizeTiles.x;
        p.heightTiles       = m_sizeTiles.y;
        p.numTiles          = m_numTiles;

        p.renderModeFlags   = m_renderModeFlags;
        p.deferredClear     = m_deferredClear ? 1 : 0;
        p.clearColor        = m_clearColor;
        p.clearDepth        = CR_DEPTH_MAX;

        p.maxSubtris        = m_maxSubtris;
        p.maxBinSegs        = m_maxBinSegs;
        p.maxTileSegs       = m_maxTileSegs;

        p.triSubtris        = m_triSubtris.getPtr();
        p.triHeader         = m_triHeader.getPtr();
        p.triData           = m_triData.getPtr();
        p.binSegData        = m_binSegData.getPtr();
        p.binSegNext        = m_binSegNext.getPtr();
        p.binSegCount       = m_binSegCount.getPtr();
        p.binFirstSeg       = m_binFirstSeg.getPtr();
        p.binTotal          = m_binTotal.getPtr();
        p.tileSegData       = m_tileSegData.getPtr();
        p.tileSegNext       = m_tileSegNext.getPtr();
        p.tileSegCount      = m_tileSegCount.getPtr();
        p.activeTiles       = m_activeTiles.getPtr();
        p.tileFirstSeg      = m_tileFirstSeg.getPtr();

        p.colorBuffer       = m_colorBuffer.getPtr();
        p.depthBuffer       = m_depthBuffer.getPtr();
        p.peelBuffer        = (m_renderModeFlags & CudaRaster::RenderModeFlag_EnableDepthPeeling) ? m_peelBuffer.getPtr() : 0;

        memcpy(&p.imageParamsFirst, imageParams, min(m_numImages, CR_EMBED_IMAGE_PARAMS) * sizeof(CRImageParams));
        p.imageParamsExtra  = (CRImageParams*)m_crImageParamsExtra.getPtr();
    }

    // Setup block sizes.

    dim3 brBlock(32, CR_BIN_WARPS);
    dim3 crBlock(32, CR_COARSE_WARPS);
    dim3 frBlock(32, m_numFineWarpsPerBlock);
    void* args[] = {&p};

    // Launch stages from setup to coarse and copy atomics to host only if this is not a peeling iteration.
    if (!peel)
    {
        if (instanceMode)
        {
            int setupBlocks = (m_numTriangles - 1) / (32 * CR_SETUP_WARPS) + 1;
            NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((void*)triangleSetupKernel, dim3(setupBlocks, 1, m_numImages), dim3(32, CR_SETUP_WARPS), args, 0, stream));
        }
        else
        {
            for (int i=0; i < m_numImages; i++)
                p.totalCount += imageParams[i].triCount;
            int setupBlocks = (p.totalCount - 1) / (32 * CR_SETUP_WARPS) + 1;
            NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((void*)triangleSetupKernel, dim3(setupBlocks, 1, 1), dim3(32, CR_SETUP_WARPS), args, 0, stream));
        }
        NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((void*)binRasterKernel, dim3(CR_BIN_STREAMS_SIZE, 1, m_numImages), brBlock, args, 0, stream));
        NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((void*)coarseRasterKernel, dim3(m_numSMs * m_numCoarseBlocksPerSM, 1, m_numImages), crBlock, args, 0, stream));
        NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(m_crAtomicsHost.getPtr(), m_crAtomics.getPtr(), sizeof(CRAtomics) * m_numImages, cudaMemcpyDeviceToHost, stream));
    }

    // Fine rasterizer is launched always.
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((void*)fineRasterKernel, dim3(m_numSMs * m_numFineBlocksPerSM, 1, m_numImages), frBlock, args, 0, stream));
    NVDR_CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
}

//------------------------------------------------------------------------
