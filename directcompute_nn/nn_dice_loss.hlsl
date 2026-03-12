// Parallel dice loss using shared-memory tree reduction.
// Each thread group (256 threads) handles one batch element.
// Dispatch: (batch, 1, 1)

RWStructuredBuffer<float> Out : register(u0);
StructuredBuffer<float> Pred : register(t0);
StructuredBuffer<float> Target : register(t1);

cbuffer Params : register(b0) {
    uint batch;
    uint size; // C * H * W per sample
};

groupshared float s_inter[256];
groupshared float s_pred[256];
groupshared float s_tgt[256];

[numthreads(256, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint b = gid.x;
    if (b >= batch) return;

    uint offset = b * size;
    float inter = 0.0f, sp = 0.0f, st = 0.0f;

    // Each of 256 threads strides through its portion
    for (uint i = lid.x; i < size; i += 256) {
        float p = Pred[offset + i];
        float t = Target[offset + i];
        inter += p * t;
        sp += p;
        st += t;
    }

    s_inter[lid.x] = inter;
    s_pred[lid.x] = sp;
    s_tgt[lid.x] = st;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128; s > 0; s >>= 1) {
        if (lid.x < s) {
            s_inter[lid.x] += s_inter[lid.x + s];
            s_pred[lid.x] += s_pred[lid.x + s];
            s_tgt[lid.x] += s_tgt[lid.x + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (lid.x == 0) {
        float dice = (2.0f * s_inter[0]) / (s_pred[0] + s_tgt[0] + 1e-6f);
        Out[b] = 1.0f - dice;
    }
}
