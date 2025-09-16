// Copyright (c) 2009-2022, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "../CudaRaster.hpp"
#include "PrivateDefs.hpp"
#include "Constants.hpp"
#include "Util.inl"

namespace CR
{

//------------------------------------------------------------------------
// Stage implementations.
//------------------------------------------------------------------------

#include "TriangleSetup.inl"
#include "BinRaster.inl"
#include "CoarseRaster.inl"
#include "FineRaster.inl"

}

//------------------------------------------------------------------------
// Stage entry points.
//------------------------------------------------------------------------

__global__ void __launch_bounds__(CR_SETUP_WARPS * 32, CR_SETUP_OPT_BLOCKS)  triangleSetupKernel (const CR::CRParams p)  { CR::triangleSetupImpl(p); }
__global__ void __launch_bounds__(CR_BIN_WARPS * 32, 1)                      binRasterKernel     (const CR::CRParams p)  { CR::binRasterImpl(p); }
__global__ void __launch_bounds__(CR_COARSE_WARPS * 32, 1)                   coarseRasterKernel  (const CR::CRParams p)  { CR::coarseRasterImpl(p); }
__global__ void __launch_bounds__(CR_FINE_MAX_WARPS * 32, 1)                 fineRasterKernel    (const CR::CRParams p)  { CR::fineRasterImpl(p); }

//------------------------------------------------------------------------
