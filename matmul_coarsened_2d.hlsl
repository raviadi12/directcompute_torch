cbuffer Constants : register(b0) {
    uint widthA;
    uint heightA;
    uint widthB;
    uint heightB;
};

StructuredBuffer<float> BufferA : register(t0);
StructuredBuffer<float> BufferB : register(t1);
RWStructuredBuffer<float> BufferOut : register(u0);

// TS is the Output Tile Size (64x64)
#define TS 64
// DIM is the Thread Group dimension (16x16)
#define DIM 16
// WPT is Work Per Thread (4x4)
#define WPT 4
// K_STEP is the inner loop step and tile dimension along the K axis
#define K_STEP 16

groupshared float tileA[TS][K_STEP];
groupshared float tileB[K_STEP][TS];

[numthreads(DIM, DIM, 1)]
void CSMain(uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID) {
    uint tx = groupThreadID.x;
    uint ty = groupThreadID.y;
    uint tid = ty * DIM + tx; // Linear thread ID (0 to 255)

    // Register accumulation
    float sum[WPT][WPT];
    [unroll]
    for (uint wy = 0; wy < WPT; ++wy) {
        [unroll]
        for (uint wx = 0; wx < WPT; ++wx) {
            sum[wy][wx] = 0.0f;
        }
    }

    uint numTiles = (widthA + K_STEP - 1) / K_STEP;

    for (uint t = 0; t < numTiles; ++t) {
        // 1. Load tileA (64 rows, 16 cols)
        // 1024 elements total / 256 threads = 4 elements per thread
        [unroll]
        for (uint i = 0; i < (TS * K_STEP) / (DIM * DIM); ++i) {
            uint idx = i * (DIM * DIM) + tid;
            uint rowA = idx / K_STEP;
            uint colA = idx % K_STEP;
            
            uint globalRowA = groupID.y * TS + rowA;
            uint globalColA = t * K_STEP + colA;
            
            if (globalRowA < heightA && globalColA < widthA)
                tileA[rowA][colA] = BufferA[globalRowA * widthA + globalColA];
            else
                tileA[rowA][colA] = 0.0f;
        }

        // 2. Load tileB (16 rows, 64 cols)
        // 1024 elements total / 256 threads = 4 elements per thread
        [unroll]
        for (uint i = 0; i < (K_STEP * TS) / (DIM * DIM); ++i) {
            uint idx = i * (DIM * DIM) + tid;
            uint rowB = idx / TS;
            uint colB = idx % TS;
            
            uint globalRowB = t * K_STEP + rowB;
            uint globalColB = groupID.x * TS + colB;
            
            if (globalRowB < heightB && globalColB < widthB)
                tileB[rowB][colB] = BufferB[globalRowB * widthB + globalColB];
            else
                tileB[rowB][colB] = 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        // 3. Compute 2D Block
        [unroll]
        for (uint k = 0; k < K_STEP; ++k) {
            float a[WPT];
            float b[WPT];
            
            // Load 4 values from shared memory to registers
            [unroll]
            for (uint wy = 0; wy < WPT; ++wy) {
                a[wy] = tileA[ty + wy * DIM][k];
            }
            
            [unroll]
            for (uint wx = 0; wx < WPT; ++wx) {
                b[wx] = tileB[k][tx + wx * DIM];
            }
            
            // 16 Fused Multiply-Adds (MADs)
            [unroll]
            for (uint wy = 0; wy < WPT; ++wy) {
                [unroll]
                for (uint wx = 0; wx < WPT; ++wx) {
                    sum[wy][wx] += a[wy] * b[wx];
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // 4. Store Results back to Global Memory
    [unroll]
    for (uint wy = 0; wy < WPT; ++wy) {
        [unroll]
        for (uint wx = 0; wx < WPT; ++wx) {
            // Using strided mapping to allow coalesced global writes
            uint globalRow = groupID.y * TS + ty + wy * DIM;
            uint globalCol = groupID.x * TS + tx + wx * DIM;
            
            if (globalRow < heightA && globalCol < widthB) {
                BufferOut[globalRow * widthB + globalCol] = sum[wy][wx];
            }
        }
    }
}
