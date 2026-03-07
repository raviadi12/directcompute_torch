cbuffer Params : register(b0) { uint count; uint pad[3]; };
StructuredBuffer<float> A : register(t0);
RWStructuredBuffer<float> out : register(u0);

groupshared float s_sum[256];

[numthreads(256, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID, uint ti : SV_GroupIndex) {
    float sum = 0;
    for (uint i = id.x; i < count; i += 256) {
        float v = A[i];
        sum += v * v;
    }
    s_sum[ti] = sum;
    GroupMemoryBarrierWithGroupSync();
    
    for (uint s = 128; s > 0; s >>= 1) {
        if (ti < s) s_sum[ti] += s_sum[ti + s];
        GroupMemoryBarrierWithGroupSync();
    }
    
    if (ti == 0) {
        out[0] = sqrt(s_sum[0] / count);
    }
}
