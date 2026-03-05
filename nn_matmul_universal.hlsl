cbuffer Params : register(b0) { 
    uint M, K, N, flags; 
};
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);

#define TILE_SIZE 16
groupshared float tileA[TILE_SIZE][TILE_SIZE];
groupshared float tileB[TILE_SIZE][TILE_SIZE];

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint row = gid.y * TILE_SIZE + lid.y;
    uint col = gid.x * TILE_SIZE + lid.x;
    
    float acc = 0.0;
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < numTiles; ++t) {
        // Load tileA
        uint tA_col = t * TILE_SIZE + lid.x;
        if (row < M && tA_col < K) {
            uint idxA = (flags & 1) ? (tA_col * M + row) : (row * K + tA_col);
            tileA[lid.y][lid.x] = A[idxA];
        } else {
            tileA[lid.y][lid.x] = 0.0;
        }
        
        // Load tileB
        uint tB_row = t * TILE_SIZE + lid.y;
        if (tB_row < K && col < N) {
            uint idxB = (flags & 2) ? (col * K + tB_row) : (tB_row * N + col);
            tileB[lid.y][lid.x] = B[idxB];
        } else {
            tileB[lid.y][lid.x] = 0.0;
        }
        
        GroupMemoryBarrierWithGroupSync();
        
        [unroll]
        for (uint k = 0; k < TILE_SIZE; ++k) {
            acc += tileA[lid.y][k] * tileB[k][lid.x];
        }
        
        GroupMemoryBarrierWithGroupSync();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
