// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "torch_common.inl"

//------------------------------------------------------------------------
// Python GL state wrapper.

class RasterizeGLState;
class RasterizeGLStateWrapper
{
public:
    RasterizeGLStateWrapper     (bool enableDB, bool automatic, int cudaDeviceIdx);
    ~RasterizeGLStateWrapper    (void);

    void setContext             (void);
    void releaseContext         (void);

    RasterizeGLState*           pState;
    bool                        automatic;
    int                         cudaDeviceIdx;
};

//------------------------------------------------------------------------
// Python CudaRaster state wrapper.

namespace CR { class CudaRaster; }
class RasterizeCRStateWrapper
{
public:
    RasterizeCRStateWrapper     (int cudaDeviceIdx);
    ~RasterizeCRStateWrapper    (void);

    CR::CudaRaster*             cr;
    int                         cudaDeviceIdx;
};

//------------------------------------------------------------------------
// Mipmap wrapper to prevent intrusion from Python side.

class TextureMipWrapper
{
public:
    torch::Tensor               mip;
    int                         max_mip_level;
    std::vector<int64_t>        texture_size;   // For error checking.
    bool                        cube_mode;      // For error checking.
};


//------------------------------------------------------------------------
// Antialias topology hash wrapper to prevent intrusion from Python side.

class TopologyHashWrapper
{
public:
    torch::Tensor               ev_hash;
};

//------------------------------------------------------------------------
