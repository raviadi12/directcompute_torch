// Per-parameter gradient squared-sum reduction.
// Each thread group of 256 reduces 256 grad elements to 1 partial sum.
// Dispatch: ((count+255)/256, 1, 1) per parameter, writing at globalOffset.

StructuredBuffer<float> grad : register(t0);
RWStructuredBuffer<float> partials : register(u0);

cbuffer Params : register(b0) {
    uint count;        // number of elements in this grad buffer
    uint globalOffset; // write offset into partials buffer
};

groupshared float sdata[256];

[numthreads(256, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID, uint3 dtid : SV_DispatchThreadID) {
    float val = 0.0f;
    if (dtid.x < count) val = grad[dtid.x];
    sdata[lid.x] = val * val;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128; s > 0; s >>= 1) {
        if (lid.x < s) sdata[lid.x] += sdata[lid.x + s];
        GroupMemoryBarrierWithGroupSync();
    }

    if (lid.x == 0) {
        partials[globalOffset + gid.x] = sdata[0];
    }
}
