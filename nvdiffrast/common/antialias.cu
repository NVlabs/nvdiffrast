// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "antialias.h"

//------------------------------------------------------------------------
// Helpers.

#define F32_MAX (3.402823466e+38f)
static __forceinline__ __device__ bool same_sign(float a, float b) { return (__float_as_int(a) ^ __float_as_int(b)) >= 0; }
static __forceinline__ __device__ bool rational_gt(float n0, float n1, float d0, float d1) { return (n0*d1 > n1*d0) == same_sign(d0, d1); }
static __forceinline__ __device__ int max_idx3(float n0, float n1, float n2, float d0, float d1, float d2)
{
    bool g10 = rational_gt(n1, n0, d1, d0);
    bool g20 = rational_gt(n2, n0, d2, d0);
    bool g21 = rational_gt(n2, n1, d2, d1);
    if (g20 && g21) return 2;
    if (g10) return 1;
    return 0;
}

//------------------------------------------------------------------------
// Format of antialiasing work items stored in work buffer. Usually accessed directly as int4.

struct AAWorkItem
{
    enum
    {
        EDGE_MASK       = 3,    // Edge index in lowest bits.
        FLAG_DOWN_BIT   = 2,    // Down instead of right.
        FLAG_TRI1_BIT   = 3,    // Edge is from other pixel's triangle.
    };

    int             px, py;         // Pixel x, y.
    unsigned int    pz_flags;       // High 16 bits = pixel z, low 16 bits = edge index and flags.
    float           alpha;          // Antialiasing alpha value. Zero if no AA.
};

//------------------------------------------------------------------------
// Hash functions. Adapted from public-domain code at http://www.burtleburtle.net/bob/hash/doobs.html

#define JENKINS_MAGIC (0x9e3779b9u)
static __device__ __forceinline__ void jenkins_mix(unsigned int& a, unsigned int& b, unsigned int& c)
{
    a -= b; a -= c; a ^= (c>>13);
    b -= c; b -= a; b ^= (a<<8);
    c -= a; c -= b; c ^= (b>>13);
    a -= b; a -= c; a ^= (c>>12);
    b -= c; b -= a; b ^= (a<<16);
    c -= a; c -= b; c ^= (b>>5);
    a -= b; a -= c; a ^= (c>>3);
    b -= c; b -= a; b ^= (a<<10);
    c -= a; c -= b; c ^= (b>>15);
}

// Helper class for hash index iteration. Implements simple odd-skip linear probing with a key-dependent skip.
class HashIndex
{
public:
    __device__ __forceinline__ HashIndex(const AntialiasKernelParams& p, uint64_t key)
    {
        m_mask = (p.allocTriangles << AA_LOG_HASH_ELEMENTS_PER_TRIANGLE(p.allocTriangles)) - 1; // This should work until triangle count exceeds 1073741824.
        m_idx  = (uint32_t)(key & 0xffffffffu);
        m_skip = (uint32_t)(key >> 32);
        uint32_t dummy = JENKINS_MAGIC;
        jenkins_mix(m_idx, m_skip, dummy);
        m_idx &= m_mask;
        m_skip &= m_mask;
        m_skip |= 1;
    }
    __device__ __forceinline__ int get(void) const { return m_idx; }
    __device__ __forceinline__ void next(void) { m_idx = (m_idx + m_skip) & m_mask; }
private:
    uint32_t m_idx, m_skip, m_mask;
};

static __device__ __forceinline__ void hash_insert(const AntialiasKernelParams& p, uint64_t key, int v)
{
    HashIndex idx(p, key);
    while(1)
    {
        uint64_t prev = atomicCAS((unsigned long long*)&p.evHash[idx.get()], 0, (unsigned long long)key);
        if (prev == 0 || prev == key)
            break;
        idx.next();
    }
    int* q = (int*)&p.evHash[idx.get()];
    int a = atomicCAS(q+2, 0, v);
    if (a != 0 && a != v)
        atomicCAS(q+3, 0, v);
}

static __device__ __forceinline__ int2 hash_find(const AntialiasKernelParams& p, uint64_t key)
{
    HashIndex idx(p, key);
    while(1)
    {
        uint4 entry = p.evHash[idx.get()];
        uint64_t k = ((uint64_t)entry.x) | (((uint64_t)entry.y) << 32);
        if (k == key || k == 0)
            return make_int2((int)entry.z, (int)entry.w);
        idx.next();
    }
}

static __device__ __forceinline__ void evhash_insert_vertex(const AntialiasKernelParams& p, int va, int vb, int vn)
{
    if (va == vb)
        return;
    
    uint64_t v0 = (uint32_t)min(va, vb) + 1; // canonical vertex order
    uint64_t v1 = (uint32_t)max(va, vb) + 1;
    uint64_t vk = v0 | (v1 << 32); // hash key
    hash_insert(p, vk, vn + 1);
}

static __forceinline__ __device__ int evhash_find_vertex(const AntialiasKernelParams& p, int va, int vb, int vr)
{
    if (va == vb)
        return -1;

    uint64_t v0 = (uint32_t)min(va, vb) + 1; // canonical vertex order
    uint64_t v1 = (uint32_t)max(va, vb) + 1;
    uint64_t vk = v0 | (v1 << 32); // hash key
    int2 vn = hash_find(p, vk) - 1;
    if (vn.x == vr) return vn.y;
    if (vn.y == vr) return vn.x;
    return -1;
}

//------------------------------------------------------------------------
// Mesh analysis kernel.

__global__ void AntialiasFwdMeshKernel(const AntialiasKernelParams p)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= p.numTriangles)
        return;

    int v0 = p.tri[idx * 3 + 0];
    int v1 = p.tri[idx * 3 + 1];
    int v2 = p.tri[idx * 3 + 2];

    if (v0 < 0 || v0 >= p.numVertices ||
        v1 < 0 || v1 >= p.numVertices ||
        v2 < 0 || v2 >= p.numVertices)
        return;

    if (v0 == v1 || v1 == v2 || v2 == v0)
        return;

    evhash_insert_vertex(p, v1, v2, v0);
    evhash_insert_vertex(p, v2, v0, v1);
    evhash_insert_vertex(p, v0, v1, v2);
}

//------------------------------------------------------------------------
// Discontinuity finder kernel.

__global__ void AntialiasFwdDiscontinuityKernel(const AntialiasKernelParams p)
{
    // Calculate pixel position.
    int px = blockIdx.x * AA_DISCONTINUITY_KERNEL_BLOCK_WIDTH + threadIdx.x;
    int py = blockIdx.y * AA_DISCONTINUITY_KERNEL_BLOCK_HEIGHT + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.n)
        return;

    // Pointer to our TriIdx and fetch.
    int pidx0 = ((px + p.width * (py + p.height * pz)) << 2) + 3;
    float tri0 = p.rasterOut[pidx0]; // These can stay as float, as we only compare them against each other.

    // Look right, clamp at edge.
    int pidx1 = pidx0;
    if (px < p.width - 1)
        pidx1 += 4;
    float tri1 = p.rasterOut[pidx1];

    // Look down, clamp at edge.
    int pidx2 = pidx0;
    if (py < p.height - 1)
        pidx2 += p.width << 2;
    float tri2 = p.rasterOut[pidx2];

    // Determine amount of work.
    int count = 0;
    if (tri1 != tri0) count  = 1;
    if (tri2 != tri0) count += 1;
    if (!count)
        return; // Exit warp.

    // Coalesce work counter update to once per CTA.
    __shared__ int s_temp;
    s_temp = 0;
    __syncthreads();
    int idx = atomicAdd(&s_temp, count);
    __syncthreads();
    if (idx == 0)
    {
        int base = atomicAdd(&p.workBuffer[0].x, s_temp);
        s_temp = base + 1; // don't clobber the counters in first slot.
    }
    __syncthreads();
    idx += s_temp;

    // Write to memory.
    if (tri1 != tri0) p.workBuffer[idx++] = make_int4(px, py, (pz << 16), 0);
    if (tri2 != tri0) p.workBuffer[idx]   = make_int4(px, py, (pz << 16) + (1 << AAWorkItem::FLAG_DOWN_BIT), 0);
}

//------------------------------------------------------------------------
// Forward analysis kernel.

__global__ void AntialiasFwdAnalysisKernel(const AntialiasKernelParams p)
{
    __shared__ int s_base;
    int workCount = p.workBuffer[0].x;
    for(;;)
    {
        // Persistent threads work fetcher.
        __syncthreads();
        if (threadIdx.x == 0)
            s_base = atomicAdd(&p.workBuffer[0].y, AA_ANALYSIS_KERNEL_THREADS_PER_BLOCK);
        __syncthreads();
        int thread_idx = s_base + threadIdx.x;
        if (thread_idx >= workCount)
            return;

        int4* pItem = p.workBuffer + thread_idx + 1;
        int4 item = *pItem;
        int px = item.x;
        int py = item.y;
        int pz = (int)(((unsigned int)item.z) >> 16);
        int d  = (item.z >> AAWorkItem::FLAG_DOWN_BIT) & 1;

        int pixel0 = px + p.width * (py + p.height * pz);
        int pixel1 = pixel0 + (d ? p.width : 1);
        float2 zt0 = ((float2*)p.rasterOut)[(pixel0 << 1) + 1];
        float2 zt1 = ((float2*)p.rasterOut)[(pixel1 << 1) + 1];
        int tri0 = float_to_triidx(zt0.y) - 1;
        int tri1 = float_to_triidx(zt1.y) - 1;

        // Select triangle based on background / depth.
        int tri = (tri0 >= 0) ? tri0 : tri1;
        if (tri0 >= 0 && tri1 >= 0)
            tri = (zt0.x < zt1.x) ? tri0 : tri1;
        if (tri == tri1)
        {
            // Calculate with respect to neighbor pixel if chose that triangle.
            px += 1 - d;
            py += d;
        }

        // Bail out if triangle index is corrupt.
        if (tri < 0 || tri >= p.numTriangles)
            continue;

        // Fetch vertex indices.
        int vi0 = p.tri[tri * 3 + 0];
        int vi1 = p.tri[tri * 3 + 1];
        int vi2 = p.tri[tri * 3 + 2];

        // Bail out if vertex indices are corrupt.
        if (vi0 < 0 || vi0 >= p.numVertices ||
            vi1 < 0 || vi1 >= p.numVertices ||
            vi2 < 0 || vi2 >= p.numVertices)
            continue;

        // Fetch opposite vertex indices. Use vertex itself (always silhouette) if no opposite vertex exists.
        int op0 = evhash_find_vertex(p, vi2, vi1, vi0);
        int op1 = evhash_find_vertex(p, vi0, vi2, vi1);
        int op2 = evhash_find_vertex(p, vi1, vi0, vi2);

        // Instance mode: Adjust vertex indices based on minibatch index.
        if (p.instance_mode)
        {
            int vbase = pz * p.numVertices;
            vi0 += vbase;
            vi1 += vbase;
            vi2 += vbase;
            if (op0 >= 0) op0 += vbase;
            if (op1 >= 0) op1 += vbase;
            if (op2 >= 0) op2 += vbase;
        }

        // Fetch vertex positions.
        float4 p0 = ((float4*)p.pos)[vi0];
        float4 p1 = ((float4*)p.pos)[vi1];
        float4 p2 = ((float4*)p.pos)[vi2];
        float4 o0 = (op0 < 0) ? p0 : ((float4*)p.pos)[op0];
        float4 o1 = (op1 < 0) ? p1 : ((float4*)p.pos)[op1];
        float4 o2 = (op2 < 0) ? p2 : ((float4*)p.pos)[op2];

        // Project vertices to pixel space.
        float w0  = 1.f / p0.w;
        float w1  = 1.f / p1.w;
        float w2  = 1.f / p2.w;
        float ow0 = 1.f / o0.w;
        float ow1 = 1.f / o1.w;
        float ow2 = 1.f / o2.w;
        float fx  = (float)px + .5f - p.xh;
        float fy  = (float)py + .5f - p.yh;
        float x0  = p0.x * w0 * p.xh - fx;
        float y0  = p0.y * w0 * p.yh - fy;
        float x1  = p1.x * w1 * p.xh - fx;
        float y1  = p1.y * w1 * p.yh - fy;
        float x2  = p2.x * w2 * p.xh - fx;
        float y2  = p2.y * w2 * p.yh - fy;
        float ox0 = o0.x * ow0 * p.xh - fx;
        float oy0 = o0.y * ow0 * p.yh - fy;
        float ox1 = o1.x * ow1 * p.xh - fx;
        float oy1 = o1.y * ow1 * p.yh - fy;
        float ox2 = o2.x * ow2 * p.xh - fx;
        float oy2 = o2.y * ow2 * p.yh - fy;

        // Signs to kill non-silhouette edges.
        float bb = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0); // Triangle itself.
        float a0 = (x1-ox0)*(y2-oy0) - (x2-ox0)*(y1-oy0); // Wings.
        float a1 = (x2-ox1)*(y0-oy1) - (x0-ox1)*(y2-oy1);
        float a2 = (x0-ox2)*(y1-oy2) - (x1-ox2)*(y0-oy2);

        // If no matching signs anywhere, skip the rest.
        if (same_sign(a0, bb) || same_sign(a1, bb) || same_sign(a2, bb))
        {
            // XY flip for horizontal edges.
            if (d)
            {
                swap(x0, y0);
                swap(x1, y1);
                swap(x2, y2);
            }

            float dx0 = x2 - x1;
            float dx1 = x0 - x2;
            float dx2 = x1 - x0;
            float dy0 = y2 - y1;
            float dy1 = y0 - y2;
            float dy2 = y1 - y0;

            // Check if an edge crosses between us and the neighbor pixel.
            float dc = -F32_MAX;
            float ds = (tri == tri0) ? 1.f : -1.f;
            float d0 = ds * (x1*dy0 - y1*dx0);
            float d1 = ds * (x2*dy1 - y2*dx1);
            float d2 = ds * (x0*dy2 - y0*dx2);

            if (same_sign(y1, y2)) d0 = -F32_MAX, dy0 = 1.f;
            if (same_sign(y2, y0)) d1 = -F32_MAX, dy1 = 1.f;
            if (same_sign(y0, y1)) d2 = -F32_MAX, dy2 = 1.f;

            int di = max_idx3(d0, d1, d2, dy0, dy1, dy2);
            if (di == 0 && same_sign(a0, bb) && fabsf(dy0) >= fabsf(dx0)) dc = d0 / dy0;
            if (di == 1 && same_sign(a1, bb) && fabsf(dy1) >= fabsf(dx1)) dc = d1 / dy1;
            if (di == 2 && same_sign(a2, bb) && fabsf(dy2) >= fabsf(dx2)) dc = d2 / dy2;
            float eps = .0625f; // Expect no more than 1/16 pixel inaccuracy.

            // Adjust output image if a suitable edge was found.
            if (dc > -eps && dc < 1.f + eps)
            {
                dc = fminf(fmaxf(dc, 0.f), 1.f);
                float alpha = ds * (.5f - dc);
                const float* pColor0 = p.color + pixel0 * p.channels;
                const float* pColor1 = p.color + pixel1 * p.channels;
                float* pOutput = p.output + (alpha > 0.f ? pixel0 : pixel1) * p.channels;
                for (int i=0; i < p.channels; i++)
                    atomicAdd(&pOutput[i], alpha * (pColor1[i] - pColor0[i]));

                // Rewrite the work item's flags and alpha. Keep original px, py.
                unsigned int flags = pz << 16;
                flags |= di;
                flags |= d << AAWorkItem::FLAG_DOWN_BIT;
                flags |= (__float_as_uint(ds) >> 31) << AAWorkItem::FLAG_TRI1_BIT;
                ((int2*)pItem)[1] = make_int2(flags, __float_as_int(alpha));
            }
        }
    }
}

//------------------------------------------------------------------------
// Gradient kernel.

__global__ void AntialiasGradKernel(const AntialiasKernelParams p)
{
    // Temporary space for coalesced atomics.
    CA_DECLARE_TEMP(AA_GRAD_KERNEL_THREADS_PER_BLOCK);
    __shared__ int s_base; // Work counter communication across entire CTA.

    int workCount = p.workBuffer[0].x;

    for(;;)
    {
        // Persistent threads work fetcher.
        __syncthreads();
        if (threadIdx.x == 0)
            s_base = atomicAdd(&p.workBuffer[0].y, AA_GRAD_KERNEL_THREADS_PER_BLOCK);
        __syncthreads();
        int thread_idx = s_base + threadIdx.x;
        if (thread_idx >= workCount)
            return;

        // Read work item filled out by forward kernel.
        int4 item = p.workBuffer[thread_idx + 1];
        unsigned int amask = __ballot_sync(0xffffffffu, item.w);
        if (item.w == 0)
            continue; // No effect.

        // Unpack work item and replicate setup from forward analysis kernel.
        int px = item.x;
        int py = item.y;
        int pz = (int)(((unsigned int)item.z) >> 16);
        int d = (item.z >> AAWorkItem::FLAG_DOWN_BIT) & 1;
        float alpha = __int_as_float(item.w);
        int tri1 = (item.z >> AAWorkItem::FLAG_TRI1_BIT) & 1;
        int di = item.z & AAWorkItem::EDGE_MASK;
        float ds = __int_as_float(__float_as_int(1.0) | (tri1 << 31));
        int pixel0 = px + p.width * (py + p.height * pz);
        int pixel1 = pixel0 + (d ? p.width : 1);
        int tri = float_to_triidx(p.rasterOut[((tri1 ? pixel1 : pixel0) << 2) + 3]) - 1;
        if (tri1)
        {
            px += 1 - d;
            py += d;
        }

        // Bail out if triangle index is corrupt.
        bool triFail = (tri < 0 || tri >= p.numTriangles);
        amask = __ballot_sync(amask, !triFail);
        if (triFail)
            continue;

        // Outgoing color gradients.
        float* pGrad0 = p.gradColor + pixel0 * p.channels;
        float* pGrad1 = p.gradColor + pixel1 * p.channels;

        // Incoming color gradients.
        const float* pDy = p.dy + (alpha > 0.f ? pixel0 : pixel1) * p.channels;

        // Position gradient weight based on colors and incoming gradients.
        float dd = 0.f;
        const float* pColor0 = p.color + pixel0 * p.channels;
        const float* pColor1 = p.color + pixel1 * p.channels;

        // Loop over channels and accumulate.
        for (int i=0; i < p.channels; i++)
        {
            float dy = pDy[i];
            if (dy != 0.f)
            {
                // Update position gradient weight.
                dd += dy * (pColor1[i] - pColor0[i]);

                // Update color gradients. No coalescing because all have different targets.
                float v = alpha * dy;
                atomicAdd(&pGrad0[i], -v);
                atomicAdd(&pGrad1[i], v);
            }
        }

        // If position weight is zero, skip the rest.
        bool noGrad = (dd == 0.f);
        amask = __ballot_sync(amask, !noGrad);
        if (noGrad)
            continue;

        // Fetch vertex indices of the active edge and their positions.
        int i1 = (di < 2) ? (di + 1) : 0;
        int i2 = (i1 < 2) ? (i1 + 1) : 0;
        int vi1 = p.tri[3 * tri + i1];
        int vi2 = p.tri[3 * tri + i2];

        // Bail out if vertex indices are corrupt.
        bool vtxFail = (vi1 < 0 || vi1 >= p.numVertices || vi2 < 0 || vi2 >= p.numVertices);
        amask = __ballot_sync(amask, !vtxFail);
        if (vtxFail)
            continue;

        // Instance mode: Adjust vertex indices based on minibatch index.
        if (p.instance_mode)
        {
            vi1 += pz * p.numVertices;
            vi2 += pz * p.numVertices;
        }

        // Fetch vertex positions.
        float4 p1 = ((float4*)p.pos)[vi1];
        float4 p2 = ((float4*)p.pos)[vi2];

        // Project vertices to pixel space.
        float pxh = p.xh;
        float pyh = p.yh;
        float fx = (float)px + .5f - pxh;
        float fy = (float)py + .5f - pyh;

        // XY flip for horizontal edges.
        if (d)
        {
            swap(p1.x, p1.y);
            swap(p2.x, p2.y);
            swap(pxh, pyh);
            swap(fx, fy);
        }

        // Gradient calculation setup.
        float w1 = 1.f / p1.w;
        float w2 = 1.f / p2.w;
        float x1 = p1.x * w1 * pxh - fx;
        float y1 = p1.y * w1 * pyh - fy;
        float x2 = p2.x * w2 * pxh - fx;
        float y2 = p2.y * w2 * pyh - fy;
        float dx = x2 - x1;
        float dy = y2 - y1;
        float db = x1*dy - y1*dx;

        // Compute inverse delta-y with epsilon.
        float ep = copysignf(1e-3f, dy); // ~1/1000 pixel.
        float iy = 1.f / (dy + ep);

        // Compute position gradients.
        float dby = db * iy;
        float iw1 = -w1 * iy * dd;
        float iw2 =  w2 * iy * dd;
        float gp1x = iw1 * pxh * y2;
        float gp2x = iw2 * pxh * y1;
        float gp1y = iw1 * pyh * (dby - x2);
        float gp2y = iw2 * pyh * (dby - x1);
        float gp1w = -(p1.x * gp1x + p1.y * gp1y) * w1;
        float gp2w = -(p2.x * gp2x + p2.y * gp2y) * w2;

        // XY flip the gradients.
        if (d)
        {
            swap(gp1x, gp1y);
            swap(gp2x, gp2y);
        }

        // Kill position gradients if alpha was saturated.
        if (fabsf(alpha) >= 0.5f)
        {
            gp1x = gp1y = gp1w = 0.f;
            gp2x = gp2y = gp2w = 0.f;
        }

        // Initialize coalesced atomics. Match both triangle ID and edge index.
        // Also note that some threads may be inactive.
        CA_SET_GROUP_MASK(tri ^ (di << 30), amask);

        // Accumulate gradients.
        caAtomicAdd3_xyw(p.gradPos + 4 * vi1, gp1x, gp1y, gp1w);
        caAtomicAdd3_xyw(p.gradPos + 4 * vi2, gp2x, gp2y, gp2w);
    }
}

//------------------------------------------------------------------------
