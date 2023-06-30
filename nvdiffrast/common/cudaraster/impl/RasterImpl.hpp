// Copyright (c) 2009-2022, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once
#include "PrivateDefs.hpp"
#include "Buffer.hpp"
#include "../CudaRaster.hpp"

namespace CR
{
//------------------------------------------------------------------------

class RasterImpl
{
public:
					        RasterImpl				(void);
					        ~RasterImpl				(void);

    void                    setViewportSize         (Vec3i size); // Must be multiple of tile size (8x8).
    void                    setRenderModeFlags      (U32 flags) { m_renderModeFlags = flags; }
    void                    deferredClear           (U32 color) { m_deferredClear = true; m_clearColor = color; }
    void                    setVertexBuffer         (void* ptr, int numVertices) { m_vertexPtr = ptr; m_numVertices = numVertices; } // GPU pointer.
    void                    setIndexBuffer          (void* ptr, int numTriangles) { m_indexPtr = ptr; m_numTriangles = numTriangles; } // GPU pointer.
    bool                    drawTriangles           (const Vec2i* ranges, bool peel, cudaStream_t stream);
    void*                   getColorBuffer          (void) { return m_colorBuffer.getPtr(); } // GPU pointer.
    void*                   getDepthBuffer          (void) { return m_depthBuffer.getPtr(); } // GPU pointer.
    void                    swapDepthAndPeel        (void);
    size_t                  getTotalBufferSizes     (void) const;

private:
    void                    launchStages            (bool instanceMode, bool peel, cudaStream_t stream);

    // State.

    unsigned int            m_renderModeFlags;
    bool                    m_deferredClear;
    unsigned int            m_clearColor;
    void*                   m_vertexPtr;
    void*                   m_indexPtr;
    int                     m_numVertices;          // Input buffer size.
    int                     m_numTriangles;         // Input buffer size.
    size_t                  m_bufferSizesReported;  // Previously reported buffer sizes.

    // Surfaces.

    Buffer                  m_colorBuffer;
    Buffer                  m_depthBuffer;
    Buffer                  m_peelBuffer;
    int                     m_numImages;
    Vec2i                   m_sizePixels;
    Vec2i                   m_sizeBins;
    S32                     m_numBins;
    Vec2i                   m_sizeTiles;
    S32                     m_numTiles;

    // Launch sizes etc.

    S32                     m_numSMs;
    S32                     m_numCoarseBlocksPerSM;
    S32                     m_numFineBlocksPerSM;
    S32                     m_numFineWarpsPerBlock;

    // Global intermediate buffers. Individual images have offsets to these.

    Buffer                  m_crAtomics;
    HostBuffer              m_crAtomicsHost;
    HostBuffer              m_crImageParamsHost;
    Buffer                  m_crImageParamsExtra;
    Buffer                  m_triSubtris;
    Buffer                  m_triHeader;
    Buffer                  m_triData;
    Buffer                  m_binFirstSeg;
    Buffer                  m_binTotal;
    Buffer                  m_binSegData;
    Buffer                  m_binSegNext;
	Buffer                  m_binSegCount;
    Buffer                  m_activeTiles;
    Buffer                  m_tileFirstSeg;
    Buffer                  m_tileSegData;
    Buffer                  m_tileSegNext;
    Buffer                  m_tileSegCount;

    // Actual buffer sizes.

    S32                     m_maxSubtris;
    S32                     m_maxBinSegs;
    S32                     m_maxTileSegs;
};

//------------------------------------------------------------------------
} // namespace CR

