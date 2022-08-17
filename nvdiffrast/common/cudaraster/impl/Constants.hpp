// Copyright (c) 2009-2022, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

//------------------------------------------------------------------------

#define CR_MAXVIEWPORT_LOG2     11      // ViewportSize / PixelSize.
#define CR_SUBPIXEL_LOG2        4       // PixelSize / SubpixelSize.

#define CR_MAXBINS_LOG2         4       // ViewportSize / BinSize.
#define CR_BIN_LOG2             4       // BinSize / TileSize.
#define CR_TILE_LOG2            3       // TileSize / PixelSize.

#define CR_COVER8X8_LUT_SIZE    768     // 64-bit entries.
#define CR_FLIPBIT_FLIP_Y       2
#define CR_FLIPBIT_FLIP_X       3
#define CR_FLIPBIT_SWAP_XY      4
#define CR_FLIPBIT_COMPL        5

#define CR_BIN_STREAMS_LOG2     4
#define CR_BIN_SEG_LOG2         9       // 32-bit entries.
#define CR_TILE_SEG_LOG2        5       // 32-bit entries.

#define CR_MAXSUBTRIS_LOG2      24      // Triangle structs. Dictated by CoarseRaster.
#define CR_COARSE_QUEUE_LOG2    10      // Triangles.

#define CR_SETUP_WARPS          2
#define CR_SETUP_OPT_BLOCKS     8
#define CR_BIN_WARPS            16
#define CR_COARSE_WARPS         16      // Must be a power of two.
#define CR_FINE_MAX_WARPS       20

#define CR_EMBED_IMAGE_PARAMS   32      // Number of per-image parameter structs embedded in kernel launch parameter block.

//------------------------------------------------------------------------

#define CR_MAXVIEWPORT_SIZE     (1 << CR_MAXVIEWPORT_LOG2)
#define CR_SUBPIXEL_SIZE        (1 << CR_SUBPIXEL_LOG2)
#define CR_SUBPIXEL_SQR         (1 << (CR_SUBPIXEL_LOG2 * 2))

#define CR_MAXBINS_SIZE         (1 << CR_MAXBINS_LOG2)
#define CR_MAXBINS_SQR          (1 << (CR_MAXBINS_LOG2 * 2))
#define CR_BIN_SIZE             (1 << CR_BIN_LOG2)
#define CR_BIN_SQR              (1 << (CR_BIN_LOG2 * 2))

#define CR_MAXTILES_LOG2        (CR_MAXBINS_LOG2 + CR_BIN_LOG2)
#define CR_MAXTILES_SIZE        (1 << CR_MAXTILES_LOG2)
#define CR_MAXTILES_SQR         (1 << (CR_MAXTILES_LOG2 * 2))
#define CR_TILE_SIZE            (1 << CR_TILE_LOG2)
#define CR_TILE_SQR             (1 << (CR_TILE_LOG2 * 2))

#define CR_BIN_STREAMS_SIZE     (1 << CR_BIN_STREAMS_LOG2)
#define CR_BIN_SEG_SIZE         (1 << CR_BIN_SEG_LOG2)
#define CR_TILE_SEG_SIZE        (1 << CR_TILE_SEG_LOG2)

#define CR_MAXSUBTRIS_SIZE      (1 << CR_MAXSUBTRIS_LOG2)
#define CR_COARSE_QUEUE_SIZE    (1 << CR_COARSE_QUEUE_LOG2)

//------------------------------------------------------------------------
// When evaluating interpolated Z pixel centers, we may introduce an error
// of (+-CR_LERP_ERROR) ULPs.

#define CR_LERP_ERROR(SAMPLES_LOG2) (2200u << (SAMPLES_LOG2))
#define CR_DEPTH_MIN                CR_LERP_ERROR(3)
#define CR_DEPTH_MAX                (CR_U32_MAX - CR_LERP_ERROR(3))

//------------------------------------------------------------------------
