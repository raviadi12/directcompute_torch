cbuffer ConvParams : register(b0) {
    uint batch;
    uint inC;
    uint inH;
    uint inW;
    uint outC;
    uint ks;
    uint outH;
    uint outW;
};

StructuredBuffer<float> dZ : register(t0);
StructuredBuffer<float> A_prev : register(t1);
RWStructuredBuffer<float> dF : register(u0);
RWStructuredBuffer<float> dB : register(u1);

#define TILE_SIZE 16
groupshared float subTileDZ[TILE_SIZE * TILE_SIZE];
groupshared float subTileAP[TILE_SIZE * TILE_SIZE];

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint M = outC;
    uint N = batch * outH * outW;
    uint K = N;
    uint tr = inC * ks * ks;

    uint row = gid.y * TILE_SIZE + lid.y;
    uint col = gid.x * TILE_SIZE + lid.x;

    float acc = 0.0;
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; ++t) {
        uint k_idx = t * TILE_SIZE + lid.x;
        if (row < M && k_idx < K) {
            uint b = k_idx / (outH * outW);
            uint p = k_idx % (outH * outW);
            subTileDZ[lid.y * TILE_SIZE + lid.x] = dZ[b * (outC * outH * outW) + row * (outH * outW) + p];
        } else {
            subTileDZ[lid.y * TILE_SIZE + lid.x] = 0.0;
        }

        uint k_idx2 = t * TILE_SIZE + lid.y;
        if (col < tr && k_idx2 < K) {
            uint b = k_idx2 / (outH * outW);
            uint p = k_idx2 % (outH * outW);
            uint oh = p / outW;
            uint ow = p % outW;
            
            uint ic = col / (ks * ks);
            uint ks_idx = col % (ks * ks);
            uint kh = ks_idx / ks;
            uint kw = ks_idx % ks;
            
            subTileAP[lid.y * TILE_SIZE + lid.x] = A_prev[b * (inC * inH * inW) + ic * (inH * inW) + (oh + kh) * inW + (ow + kw)];
        } else {
            subTileAP[lid.y * TILE_SIZE + lid.x] = 0.0;
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < TILE_SIZE; ++k) {
            acc += subTileDZ[lid.y * TILE_SIZE + k] * subTileAP[k * TILE_SIZE + lid.x];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < M && col < tr) {
        dF[row * tr + col] = acc;
    }
}
