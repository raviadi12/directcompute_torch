// Medium matmul: TS=32, WPT=2, K_STEP=32
// Bridges gap between universal (TS=16, 1 elem/thread) and coarsened (TS=64, 16 elems/thread)
// For moderate dimensions (32 <= M,N < 64): 4 elems/thread, 8KB shared memory
// Thread group: 16x16 = 256 threads, each computes 2x2 output elements

cbuffer Params : register(b0) { 
    uint M, K, N, flags; 
};
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);

#define TS 32
#define DIM 16
#define WPT 2
#define K_STEP 32

groupshared float tileA_T[K_STEP][TS];  // Transposed for conflict-free reads
groupshared float tileB[K_STEP][TS];

[numthreads(DIM, DIM, 1)]
void CSMain(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint tx = lid.x;
    uint ty = lid.y;
    uint tid = ty * DIM + tx;

    float sum[WPT][WPT];
    [unroll] for (uint wy = 0; wy < WPT; ++wy)
        [unroll] for (uint wx = 0; wx < WPT; ++wx)
            sum[wy][wx] = 0.0f;

    uint numTiles = (K + K_STEP - 1) / K_STEP;

    for (uint t = 0; t < numTiles; ++t) {
        uint tBase = t * K_STEP;

        // Load tileA transposed: 4 loads per thread (TS*K_STEP / 256 = 32*32/256 = 4)
        [unroll]
        for (uint i = 0; i < 4; ++i) {
            uint idx = i * 256 + tid;
            uint rowA = idx / K_STEP;
            uint colA = idx % K_STEP;
            uint globalRow = gid.y * TS + rowA;
            uint globalCol = tBase + colA;
            float val = 0.0f;
            if (globalRow < M && globalCol < K) {
                uint idxA = (flags & 1) ? (globalCol * M + globalRow) : (globalRow * K + globalCol);
                val = A[idxA];
            }
            tileA_T[colA][rowA] = val;
        }

        // Load tileB: 4 loads per thread
        [unroll]
        for (uint i = 0; i < 4; ++i) {
            uint idx = i * 256 + tid;
            uint rowB = idx / TS;
            uint colB = idx % TS;
            uint globalRow = tBase + rowB;
            uint globalCol = gid.x * TS + colB;
            float val = 0.0f;
            if (globalRow < K && globalCol < N) {
                uint idxB = (flags & 2) ? (globalCol * K + globalRow) : (globalRow * N + globalCol);
                val = B[idxB];
            }
            tileB[rowB][colB] = val;
        }

        GroupMemoryBarrierWithGroupSync();

        for (uint k = 0; k < K_STEP; ++k) {
            float a[WPT], b[WPT];
            [unroll] for (uint wy = 0; wy < WPT; ++wy)
                a[wy] = tileA_T[k][ty + wy * DIM];
            [unroll] for (uint wx = 0; wx < WPT; ++wx)
                b[wx] = tileB[k][tx + wx * DIM];
            [unroll] for (uint wy = 0; wy < WPT; ++wy)
                [unroll] for (uint wx = 0; wx < WPT; ++wx)
                    sum[wy][wx] += a[wy] * b[wx];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    [unroll]
    for (uint wy = 0; wy < WPT; ++wy) {
        [unroll]
        for (uint wx = 0; wx < WPT; ++wx) {
            uint globalRow = gid.y * TS + ty + wy * DIM;
            uint globalCol = gid.x * TS + tx + wx * DIM;
            if (globalRow < M && globalCol < N) {
                C[globalRow * N + globalCol] = sum[wy][wx];
            }
        }
    }
}
