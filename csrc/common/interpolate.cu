// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "common.h"
#include "interpolate.h"

//------------------------------------------------------------------------
// Forward kernel.

template <bool ENABLE_DA>
static __forceinline__ __device__ void InterpolateFwdKernelTemplate(const InterpolateKernelParams p)
{
    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)
        return;

    // Pixel index.
    int pidx = px + p.width * (py + p.height * pz);

    // Output ptrs.
    float* out = p.out + pidx * p.numAttr;
    float2* outDA = ENABLE_DA ? (((float2*)p.outDA) + pidx * p.numDiffAttr) : 0;

    // Fetch rasterizer output.
    float4 r = ((float4*)p.rast)[pidx];
    int triIdx = float_to_triidx(r.w) - 1;
    bool triValid = (triIdx >= 0 && triIdx < p.numTriangles);

    // If no geometry in entire warp, zero the output and exit.
    // Otherwise force barys to zero and output with live threads.
    if (__all_sync(0xffffffffu, !triValid))
    {
        for (int i=0; i < p.numAttr; i++)
            out[i] = 0.f;
        if (ENABLE_DA)
            for (int i=0; i < p.numDiffAttr; i++)
                outDA[i] = make_float2(0.f, 0.f);
        return;
    }

    // Fetch vertex indices.
    int vi0 = triValid ? p.tri[triIdx * 3 + 0] : 0;
    int vi1 = triValid ? p.tri[triIdx * 3 + 1] : 0;
    int vi2 = triValid ? p.tri[triIdx * 3 + 2] : 0;

    // Bail out if corrupt indices.
    if (vi0 < 0 || vi0 >= p.numVertices ||
        vi1 < 0 || vi1 >= p.numVertices ||
        vi2 < 0 || vi2 >= p.numVertices)
        return;

    // In instance mode, adjust vertex indices by minibatch index unless broadcasting.
    if (p.instance_mode && !p.attrBC)
    {
        vi0 += pz * p.numVertices;
        vi1 += pz * p.numVertices;
        vi2 += pz * p.numVertices;
    }

    // Pointers to attributes.
    const float* a0 = p.attr + vi0 * p.numAttr;
    const float* a1 = p.attr + vi1 * p.numAttr;
    const float* a2 = p.attr + vi2 * p.numAttr;

    // Barys. If no triangle, force all to zero -> output is zero.
    float b0 = triValid ? r.x : 0.f;
    float b1 = triValid ? r.y : 0.f;
    float b2 = triValid ? (1.f - r.x - r.y) : 0.f;

    // Interpolate and write attributes.
    for (int i=0; i < p.numAttr; i++)
        out[i] = b0*a0[i] + b1*a1[i] + b2*a2[i];

    // No diff attrs? Exit.
    if (!ENABLE_DA)
        return;

    // Read bary pixel differentials if we have a triangle.
    float4 db = make_float4(0.f, 0.f, 0.f, 0.f);
    if (triValid)
        db = ((float4*)p.rastDB)[pidx];

    // Unpack a bit.
    float dudx = db.x;
    float dudy = db.y;
    float dvdx = db.z;
    float dvdy = db.w;

    // Calculate the pixel differentials of chosen attributes.    
    for (int i=0; i < p.numDiffAttr; i++)
    {   
        // Input attribute index.
        int j = p.diff_attrs_all ? i : p.diffAttrs[i];
        if (j < 0)
            j += p.numAttr; // Python-style negative indices.

        // Zero output if invalid index.
        float dsdx = 0.f;
        float dsdy = 0.f;
        if (j >= 0 && j < p.numAttr)
        {
            float s0 = a0[j];
            float s1 = a1[j];
            float s2 = a2[j];
            float dsdu = s0 - s2;
            float dsdv = s1 - s2;
            dsdx = dudx*dsdu + dvdx*dsdv;
            dsdy = dudy*dsdu + dvdy*dsdv;
        }

        // Write.
        outDA[i] = make_float2(dsdx, dsdy);
    }
}

// Template specializations.
__global__ void InterpolateFwdKernel  (const InterpolateKernelParams p) { InterpolateFwdKernelTemplate<false>(p); }
__global__ void InterpolateFwdKernelDa(const InterpolateKernelParams p) { InterpolateFwdKernelTemplate<true>(p); }

//------------------------------------------------------------------------
// Gradient kernel.

template <bool ENABLE_DA>
static __forceinline__ __device__ void InterpolateGradKernelTemplate(const InterpolateKernelParams p)
{
    // Temporary space for coalesced atomics.
    CA_DECLARE_TEMP(IP_GRAD_MAX_KERNEL_BLOCK_WIDTH * IP_GRAD_MAX_KERNEL_BLOCK_HEIGHT);

    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)
        return;

    // Pixel index.
    int pidx = px + p.width * (py + p.height * pz);

    // Fetch triangle ID. If none, output zero bary/db gradients and exit.
    float4 r = ((float4*)p.rast)[pidx];
    int triIdx = float_to_triidx(r.w) - 1;
    if (triIdx < 0 || triIdx >= p.numTriangles)
    {
        ((float4*)p.gradRaster)[pidx] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (ENABLE_DA)
            ((float4*)p.gradRasterDB)[pidx] = make_float4(0.f, 0.f, 0.f, 0.f);
        return;
    }

    // Fetch vertex indices.
    int vi0 = p.tri[triIdx * 3 + 0];
    int vi1 = p.tri[triIdx * 3 + 1];
    int vi2 = p.tri[triIdx * 3 + 2];

    // Bail out if corrupt indices.
    if (vi0 < 0 || vi0 >= p.numVertices ||
        vi1 < 0 || vi1 >= p.numVertices ||
        vi2 < 0 || vi2 >= p.numVertices)
        return;

    // In instance mode, adjust vertex indices by minibatch index unless broadcasting.
    if (p.instance_mode && !p.attrBC)
    {
        vi0 += pz * p.numVertices;
        vi1 += pz * p.numVertices;
        vi2 += pz * p.numVertices;
    }

    // Initialize coalesced atomics.
    CA_SET_GROUP(triIdx);

    // Pointers to inputs.
    const float* a0 = p.attr + vi0 * p.numAttr;
    const float* a1 = p.attr + vi1 * p.numAttr;
    const float* a2 = p.attr + vi2 * p.numAttr;
    const float* pdy = p.dy + pidx * p.numAttr;

    // Pointers to outputs.
    float* ga0 = p.gradAttr + vi0 * p.numAttr;
    float* ga1 = p.gradAttr + vi1 * p.numAttr;
    float* ga2 = p.gradAttr + vi2 * p.numAttr;

    // Barys and bary gradient accumulators.
    float b0 = r.x;
    float b1 = r.y;
    float b2 = 1.f - r.x - r.y;
    float gb0 = 0.f;
    float gb1 = 0.f;

    // Loop over attributes and accumulate attribute gradients.
    for (int i=0; i < p.numAttr; i++)
    {
        float y = pdy[i];
        float s0 = a0[i];
        float s1 = a1[i];
        float s2 = a2[i];
        gb0 += y * (s0 - s2);
        gb1 += y * (s1 - s2);
        caAtomicAdd(ga0 + i, b0 * y);
        caAtomicAdd(ga1 + i, b1 * y);
        caAtomicAdd(ga2 + i, b2 * y);
    }

    // Write the bary gradients.
    ((float4*)p.gradRaster)[pidx] = make_float4(gb0, gb1, 0.f, 0.f);

    // If pixel differentials disabled, we're done.
    if (!ENABLE_DA)
        return;

    // Calculate gradients based on attribute pixel differentials.
    const float2* dda = ((float2*)p.dda) + pidx * p.numDiffAttr;
    float gdudx = 0.f;
    float gdudy = 0.f;
    float gdvdx = 0.f;
    float gdvdy = 0.f;

    // Read bary pixel differentials.
    float4 db = ((float4*)p.rastDB)[pidx];
    float dudx = db.x;
    float dudy = db.y;
    float dvdx = db.z;
    float dvdy = db.w;

    for (int i=0; i < p.numDiffAttr; i++)
    {
        // Input attribute index.
        int j = p.diff_attrs_all ? i : p.diffAttrs[i];
        if (j < 0)
            j += p.numAttr; // Python-style negative indices.

        // Check that index is valid.
        if (j >= 0 && j < p.numAttr)
        {
            float2 dsdxy = dda[i];
            float dsdx = dsdxy.x;
            float dsdy = dsdxy.y;

            float s0 = a0[j];
            float s1 = a1[j];
            float s2 = a2[j];

            // Gradients of db.
            float dsdu = s0 - s2;
            float dsdv = s1 - s2;
            gdudx += dsdu * dsdx;
            gdudy += dsdu * dsdy;
            gdvdx += dsdv * dsdx;
            gdvdy += dsdv * dsdy;

            // Gradients of attributes.
            float du = dsdx*dudx + dsdy*dudy;
            float dv = dsdx*dvdx + dsdy*dvdy;
            caAtomicAdd(ga0 + j, du);
            caAtomicAdd(ga1 + j, dv);
            caAtomicAdd(ga2 + j, -du - dv);
        }
    }

    // Write.
    ((float4*)p.gradRasterDB)[pidx] = make_float4(gdudx, gdudy, gdvdx, gdvdy);
}

// Template specializations.
__global__ void InterpolateGradKernel  (const InterpolateKernelParams p) { InterpolateGradKernelTemplate<false>(p); }
__global__ void InterpolateGradKernelDa(const InterpolateKernelParams p) { InterpolateGradKernelTemplate<true>(p); }

//------------------------------------------------------------------------
