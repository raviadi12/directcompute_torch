cbuffer Constants : register(b0) {
    uint widthA;
    uint heightA;
    uint widthB;
    uint heightB;
};

StructuredBuffer<float> BufferA : register(t0);
StructuredBuffer<float> BufferB : register(t1);
RWStructuredBuffer<float> BufferOut : register(u0);

#define TILE_SIZE 16

groupshared float tileA[TILE_SIZE][TILE_SIZE];
groupshared float tileB[TILE_SIZE][TILE_SIZE];

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void CSMain(uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID) {
    uint row = groupID.y * TILE_SIZE + groupThreadID.y;
    uint col = groupID.x * TILE_SIZE + groupThreadID.x;

    float sum = 0.0f;
    uint numTiles = (widthA + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; ++t) {
        uint tiledColA = t * TILE_SIZE + groupThreadID.x;
        uint tiledRowB = t * TILE_SIZE + groupThreadID.y;

        if (row < heightA && tiledColA < widthA)
            tileA[groupThreadID.y][groupThreadID.x] = BufferA[row * widthA + tiledColA];
        else
            tileA[groupThreadID.y][groupThreadID.x] = 0.0f;

        if (tiledRowB < heightB && col < widthB)
            tileB[groupThreadID.y][groupThreadID.x] = BufferB[tiledRowB * widthB + col];
        else
            tileB[groupThreadID.y][groupThreadID.x] = 0.0f;

        GroupMemoryBarrierWithGroupSync();

        // Force complete loop unrolling
        [unroll]
        for (uint k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[groupThreadID.y][k] * tileB[k][groupThreadID.x];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < heightA && col < widthB) {
        BufferOut[row * widthB + col] = sum;
    }
}
