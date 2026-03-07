cbuffer ConvParams : register(b0) {
    uint batch;
    uint inC;
    uint inH;
    uint inW;
    uint outC;
    uint ks;
    uint outH;
    uint outW;
    uint actType;
};

StructuredBuffer<float> input : register(t0);
StructuredBuffer<float> filters : register(t1);
StructuredBuffer<float> bias : register(t2);
RWStructuredBuffer<float> output : register(u0);

#define TILE_SIZE 16
groupshared float subTileF[TILE_SIZE * TILE_SIZE];
groupshared float subTileI[TILE_SIZE * TILE_SIZE];

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint M = outC;
    uint N = batch * outH * outW;
    uint K = inC * ks * ks;

    uint row = gid.y * TILE_SIZE + lid.y;
    uint col = gid.x * TILE_SIZE + lid.x;

    uint b = col / (outH * outW);
    uint pixelIdx = col % (outH * outW);
    uint oh = pixelIdx / outW;
    uint ow = pixelIdx % outW;

    float acc = 0.0;
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; ++t) {
        uint fCol = t * TILE_SIZE + lid.x;
        if (row < M && fCol < K) {
            subTileF[lid.y * TILE_SIZE + lid.x] = filters[row * K + fCol];
        } else {
            subTileF[lid.y * TILE_SIZE + lid.x] = 0.0;
        }

        uint iRow = t * TILE_SIZE + lid.y;
        if (iRow < K && col < N) {
            uint ic = iRow / (ks * ks);
            uint kIdx = iRow % (ks * ks);
            uint kh = kIdx / ks;
            uint kw = kIdx % ks;
            int ih = (int)oh + (int)kh;
            int iw_val = (int)ow + (int)kw;
            subTileI[lid.y * TILE_SIZE + lid.x] = input[b * (inC * inH * inW) + ic * (inH * inW) + ih * inW + iw_val];
        } else {
            subTileI[lid.y * TILE_SIZE + lid.x] = 0.0;
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < TILE_SIZE; ++k) {
            acc += subTileF[lid.y * TILE_SIZE + k] * subTileI[k * TILE_SIZE + lid.x];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < M && col < N) {
        float z = acc + bias[row];
        float a = z;
        if (actType == 1) a = max(0.0, z);
        else if (actType == 2) a = tanh(z);
        
        output[b * (outC * outH * outW) + row * (outH * outW) + pixelIdx] = a;
    }
}
