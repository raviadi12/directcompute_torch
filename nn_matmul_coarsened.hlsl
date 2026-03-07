// Optimized matmul: TS=64, WPT=4, K_STEP=32
// Best config for Intel Xe:
//   - 16 accumulators (no register spilling on 128 GRFs)
//   - Transposed tileA with +1 padding for bank-conflict-free shared mem
//   - Register prefetching: load next tile from global while computing current
//   - K_STEP=32: 512 FMAs per thread per tile, 8 loads per thread
//   - ~16KB shared memory (good occupancy: 3-4 workgroups per subslice)

cbuffer Params : register(b0) { 
    uint M, K, N, flags; 
};
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);

#define TS 64
#define DIM 16
#define WPT 4
#define K_STEP 32
#define NLD 8       // loads per thread = TS * K_STEP / (DIM*DIM) = 64*32/256

groupshared float tileA_T[K_STEP][TS + 1];  // +1 pad: stride 65 breaks bank alignment
groupshared float tileB[K_STEP][TS];

[numthreads(DIM, DIM, 1)]
void CSMain(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint tx = lid.x, ty = lid.y;
    const uint tid = ty * DIM + tx;

    float acc[WPT][WPT];
    [unroll] for (uint r = 0; r < WPT; ++r)
        [unroll] for (uint c = 0; c < WPT; ++c)
            acc[r][c] = 0.0f;

    const uint numTiles = (K + K_STEP - 1) / K_STEP;

    // ── Prologue: load first tile into shared memory ──
    [unroll] for (uint i = 0; i < NLD; ++i) {
        uint idx = i * 256 + tid;
        uint rA = idx / K_STEP, cA = idx % K_STEP;
        uint gR = gid.y * TS + rA, gC = cA;
        tileA_T[cA][rA] = (gR < M && gC < K)
            ? A[(flags & 1) ? gC * M + gR : gR * K + gC] : 0.0f;
    }
    [unroll] for (uint i = 0; i < NLD; ++i) {
        uint idx = i * 256 + tid;
        uint rB = idx / TS, cB = idx % TS;
        uint gR = rB, gC = gid.x * TS + cB;
        tileB[rB][cB] = (gR < K && gC < N)
            ? B[(flags & 2) ? gC * K + gR : gR * N + gC] : 0.0f;
    }
    GroupMemoryBarrierWithGroupSync();

    // ── Main loop: prefetch next tile → compute current → writeback ──
    for (uint t = 0; t < numTiles; ++t) {
        float pA[NLD], pB[NLD];
        const bool more = (t + 1 < numTiles);
        if (more) {
            const uint base = (t + 1) * K_STEP;
            [unroll] for (uint i = 0; i < NLD; ++i) {
                uint idx = i * 256 + tid;
                uint rA = idx / K_STEP, cA = idx % K_STEP;
                uint gR = gid.y * TS + rA, gC = base + cA;
                pA[i] = (gR < M && gC < K)
                    ? A[(flags & 1) ? gC * M + gR : gR * K + gC] : 0.0f;
            }
            [unroll] for (uint i = 0; i < NLD; ++i) {
                uint idx = i * 256 + tid;
                uint rB = idx / TS, cB = idx % TS;
                uint gR = base + rB, gC = gid.x * TS + cB;
                pB[i] = (gR < K && gC < N)
                    ? B[(flags & 2) ? gC * K + gR : gR * N + gC] : 0.0f;
            }
        }

        for (uint k = 0; k < K_STEP; ++k) {
            float a[WPT], b[WPT];
            [unroll] for (uint r = 0; r < WPT; ++r)
                a[r] = tileA_T[k][ty + r * DIM];
            [unroll] for (uint c = 0; c < WPT; ++c)
                b[c] = tileB[k][tx + c * DIM];
            [unroll] for (uint r = 0; r < WPT; ++r)
                [unroll] for (uint c = 0; c < WPT; ++c)
                    acc[r][c] += a[r] * b[c];
        }

        GroupMemoryBarrierWithGroupSync();

        if (more) {
            [unroll] for (uint i = 0; i < NLD; ++i) {
                uint idx = i * 256 + tid;
                tileA_T[idx % K_STEP][idx / K_STEP] = pA[i];
            }
            [unroll] for (uint i = 0; i < NLD; ++i) {
                uint idx = i * 256 + tid;
                tileB[idx / TS][idx % TS] = pB[i];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // ── Store results ──
    [unroll] for (uint r = 0; r < WPT; ++r) {
        [unroll] for (uint c = 0; c < WPT; ++c) {
            uint gR = gid.y * TS + ty + r * DIM;
            uint gC = gid.x * TS + tx + c * DIM;
            if (gR < M && gC < N)
                C[gR * N + gC] = acc[r][c];
        }
    }
}
