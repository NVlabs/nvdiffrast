// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "framework.h"
#include "texture.h"

//------------------------------------------------------------------------
// Mip stack construction and access helpers.

void raiseMipSizeError(NVDR_CTX_ARGS, const TextureKernelParams& p)
{
    char buf[1024];
    int bufsz = 1024;

    std::string msg = "Mip-map size error - cannot downsample an odd extent greater than 1. Resize the texture so that both spatial extents are powers of two, or limit the number of mip maps using max_mip_level argument.\n";

    int w = p.texWidth;
    int h = p.texHeight;
    bool ew = false;
    bool eh = false;

    msg += "Attempted mip stack construction:\n";
    msg +=               "level  width height\n";
    msg +=               "-----  ----- ------\n";
    snprintf(buf, bufsz, "base   %5d  %5d\n", w, h);
    msg += buf;

    int mipTotal = 0;
    int level = 0;
    while ((w|h) > 1 && !(ew || eh)) // Stop at first impossible size.
    {
        // Current level.
        level += 1;

        // Determine if downsampling fails.
        ew = ew || (w > 1 && (w & 1));
        eh = eh || (h > 1 && (h & 1));

        // Downsample.
        if (w > 1) w >>= 1;
        if (h > 1) h >>= 1;

        // Append level size to error message.
        snprintf(buf, bufsz, "mip %-2d ", level);
        msg += buf; 
        if (ew) snprintf(buf, bufsz, "  err  ");
        else    snprintf(buf, bufsz, "%5d  ", w);
        msg += buf;
        if (eh) snprintf(buf, bufsz, "  err\n");
        else    snprintf(buf, bufsz, "%5d\n", h);
        msg += buf;
    }

    NVDR_CHECK(0, msg);
}

int calculateMipInfo(NVDR_CTX_ARGS, TextureKernelParams& p, int* mipOffsets)
{
    // No levels at all?
    if (p.mipLevelLimit == 0)
    {
        p.mipLevelMax = 0;
        return 0;
    }

    // Current level size.
    int w = p.texWidth;
    int h = p.texHeight;

    int mipTotal = 0;
    int level = 0;
    int c = (p.boundaryMode == TEX_BOUNDARY_MODE_CUBE) ? (p.channels * 6) : p.channels;
    mipOffsets[0] = 0;
    while ((w|h) > 1)
    {
        // Current level.
        level += 1;

        // Quit if cannot downsample.
        if ((w > 1 && (w & 1)) || (h > 1 && (h & 1)))
            raiseMipSizeError(NVDR_CTX_PARAMS, p);

        // Downsample.
        if (w > 1) w >>= 1;
        if (h > 1) h >>= 1;

        mipOffsets[level] = mipTotal; // Store the mip offset (#floats).
        mipTotal += w * h * p.texDepth * c;

        // Hit the level limit?
        if (p.mipLevelLimit >= 0 && level == p.mipLevelLimit)
            break;
    }

    p.mipLevelMax = level;
    return mipTotal;
}

//------------------------------------------------------------------------
