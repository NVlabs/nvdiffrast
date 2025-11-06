// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <cuda_runtime.h>

//------------------------------------------------------------------------
// Block and grid size calculators for kernel launches.

dim3 getLaunchBlockSize(int maxWidth, int maxHeight, int width, int height)
{
    int maxThreads = maxWidth * maxHeight;
    if (maxThreads <= 1 || (width * height) <= 1)
        return dim3(1, 1, 1); // Degenerate.

    // Start from max size.
    int bw = maxWidth;
    int bh = maxHeight;

    // Optimizations for weirdly sized buffers.
    if (width < bw)
    {
        // Decrease block width to smallest power of two that covers the buffer width.
        while ((bw >> 1) >= width)
            bw >>= 1;

        // Maximize height.
        bh = maxThreads / bw;
        if (bh > height)
            bh = height;
    }
    else if (height < bh)
    {
        // Halve height and double width until fits completely inside buffer vertically.
        while (bh > height)
        {
            bh >>= 1;
            if (bw < width)
                bw <<= 1;
        }
    }

    // Done.
    return dim3(bw, bh, 1);
}

dim3 getLaunchGridSize(dim3 blockSize, int width, int height, int depth)
{
    dim3 gridSize;
    gridSize.x = (width  - 1) / blockSize.x + 1;
    gridSize.y = (height - 1) / blockSize.y + 1;
    gridSize.z = (depth  - 1) / blockSize.z + 1;
    return gridSize;
}

//------------------------------------------------------------------------
