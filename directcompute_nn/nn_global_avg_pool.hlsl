// Global Average Pooling: (N, C, H, W) -> (N, C, 1, 1)
// Dispatched with (C * N) groups, 256 threads each reduce over H*W
cbuffer Params : register(b0) {
    uint batch;
    uint C;
    uint S;  // H * W
    uint pad1;
};

StructuredBuffer<float> input   : register(t0);
RWStructuredBuffer<float> output : register(u0);

groupshared float s_sum[256];

[numthreads(256, 1, 1)]
void CSMain(uint3 tid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint nc = gid.x;  // linear index into (N, C)
    float my_sum = 0.0f;

    uint base = nc * S;
    for (uint i = tid.x; i < S; i += 256) {
        my_sum += input[base + i];
    }

    s_sum[tid.x] = my_sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid.x < s) s_sum[tid.x] += s_sum[tid.x + s];
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid.x == 0) {
        output[nc] = s_sum[0] / (float)S;
    }
}
