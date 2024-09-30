// Copyright (c) 2009-2022, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "Defs.hpp"
#include "../CudaRaster.hpp"
#include "RasterImpl.hpp"

using namespace CR;

//------------------------------------------------------------------------
// Stub interface implementation.
//------------------------------------------------------------------------

CudaRaster::CudaRaster()
{
    m_impl = new RasterImpl();
}

CudaRaster::~CudaRaster()
{
    delete m_impl;
}

void CudaRaster::setBufferSize(int width, int height, int numImages)
{
    m_impl->setBufferSize(Vec3i(width, height, numImages));
}

void CudaRaster::setViewport(int width, int height, int offsetX, int offsetY)
{
    m_impl->setViewport(Vec2i(width, height), Vec2i(offsetX, offsetY));
}

void CudaRaster::setRenderModeFlags(U32 flags)
{
    m_impl->setRenderModeFlags(flags);
}

void CudaRaster::deferredClear(U32 clearColor)
{
    m_impl->deferredClear(clearColor);
}

void CudaRaster::setVertexBuffer(void* vertices, int numVertices)
{
    m_impl->setVertexBuffer(vertices, numVertices);
}

void CudaRaster::setIndexBuffer(void* indices, int numTriangles)
{
    m_impl->setIndexBuffer(indices, numTriangles);
}

bool CudaRaster::drawTriangles(const int* ranges, bool peel, cudaStream_t stream)
{
    return m_impl->drawTriangles((const Vec2i*)ranges, peel, stream);
}

void* CudaRaster::getColorBuffer(void)
{
    return m_impl->getColorBuffer();
}

void* CudaRaster::getDepthBuffer(void)
{
    return m_impl->getDepthBuffer();
}

void CudaRaster::swapDepthAndPeel(void)
{
    m_impl->swapDepthAndPeel();
}

//------------------------------------------------------------------------
