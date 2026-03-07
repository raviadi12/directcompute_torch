cbuffer Constants : register(b0) {
    uint widthA;
    uint heightA;
    uint widthB;
    uint heightB;
};

StructuredBuffer<float> BufferA : register(t0);
StructuredBuffer<float> BufferB : register(t1);
RWStructuredBuffer<float> BufferOut : register(u0);

// Block size for the output matrix
#define TS_Y 16
#define TS_X 64

// Work per thread (each thread computes a 1x4 block)
#define WPT 4 

groupshared float tileA[TS_Y][TS_Y];
groupshared float tileB[TS_Y][TS_X];

[numthreads(16, 16, 1)]
void CSMain(uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID) {
    uint row = groupID.y * TS_Y + groupThreadID.y;
    
    // Each thread group processes 64 columns. 
    // Thread x computes 4 consecutive columns.
    uint colBase = groupID.x * TS_X + groupThreadID.x * WPT;
    
    // Register arrays to accumulate the results (Thread Coarsening)
    float sum[WPT] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint numTiles = (widthA + TS_Y - 1) / TS_Y;

    for (uint t = 0; t < numTiles; ++t) {
        
        // 1. Load tileA (16x16 elements using 16x16 threads = 1 element per thread)
        uint tiledColA = t * TS_Y + groupThreadID.x;
        if (row < heightA && tiledColA < widthA)
            tileA[groupThreadID.y][groupThreadID.x] = BufferA[row * widthA + tiledColA];
        else
            tileA[groupThreadID.y][groupThreadID.x] = 0.0f;

        // 2. Load tileB (16x64 elements using 16x16 threads = 4 elements per thread)
        uint linearTID = groupThreadID.y * 16 + groupThreadID.x; 
        
        [unroll]
        for (uint i = 0; i < 4; ++i) {
            uint linearLoadIdx = linearTID + i * 256;
            uint loadY = linearLoadIdx / TS_X; 
            uint loadX = linearLoadIdx % TS_X; 
            
            uint tiledRowB = t * TS_Y + loadY;
            uint tiledColB = groupID.x * TS_X + loadX;
            
            if (tiledRowB < heightB && tiledColB < widthB)
                tileB[loadY][loadX] = BufferB[tiledRowB * widthB + tiledColB];
            else
                tileB[loadY][loadX] = 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        // 3. Compute
        [unroll]
        for (uint k = 0; k < TS_Y; ++k) {
            float a = tileA[groupThreadID.y][k];
            
            [unroll]
            for (uint w = 0; w < WPT; ++w) {
                // Fused Multiply-Add (MAD) in registers
                sum[w] += a * tileB[k][groupThreadID.x * WPT + w];
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // 4. Store Results
    [unroll]
    for (uint w = 0; w < WPT; ++w) {
        uint col = colBase + w;
        if (row < heightA && col < widthB) {
            BufferOut[row * widthB + col] = sum[w];
        }
    }
}
