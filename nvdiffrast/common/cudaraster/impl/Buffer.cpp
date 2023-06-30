// Copyright (c) 2009-2022, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "../../framework.h"
#include "Buffer.hpp"

using namespace CR;

//------------------------------------------------------------------------
// GPU buffer.
//------------------------------------------------------------------------

Buffer::Buffer(void)
:   m_gpuPtr(NULL),
    m_bytes (0)
{
    // empty
}

Buffer::~Buffer(void)
{
    if (m_gpuPtr)
        cudaFree(m_gpuPtr); // Don't throw an exception.
}

void Buffer::reset(size_t bytes)
{
    if (bytes == m_bytes)
        return;

    if (m_gpuPtr)
    {
        NVDR_CHECK_CUDA_ERROR(cudaFree(m_gpuPtr));
        m_gpuPtr = NULL;
    }

    if (bytes > 0)
        NVDR_CHECK_CUDA_ERROR(cudaMalloc(&m_gpuPtr, bytes));

    m_bytes = bytes;
}

void Buffer::grow(size_t bytes)
{
    if (bytes > m_bytes)
        reset(bytes);
}

//------------------------------------------------------------------------
// Host buffer with page-locked memory.
//------------------------------------------------------------------------

HostBuffer::HostBuffer(void)
:   m_hostPtr(NULL),
    m_bytes  (0)
{
    // empty
}

HostBuffer::~HostBuffer(void)
{
    if (m_hostPtr)
        cudaFreeHost(m_hostPtr); // Don't throw an exception.
}

void HostBuffer::reset(size_t bytes)
{
    if (bytes == m_bytes)
        return;

    if (m_hostPtr)
    {
        NVDR_CHECK_CUDA_ERROR(cudaFreeHost(m_hostPtr));
        m_hostPtr = NULL;
    }

    if (bytes > 0)
        NVDR_CHECK_CUDA_ERROR(cudaMallocHost(&m_hostPtr, bytes));

    m_bytes = bytes;
}

void HostBuffer::grow(size_t bytes)
{
    if (bytes > m_bytes)
        reset(bytes);
}

//------------------------------------------------------------------------
