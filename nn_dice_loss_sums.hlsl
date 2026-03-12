// Two-pass dice loss backward: Pass 1
// Computes per-batch (intersection, sum_pred, sum_target) using
// shared-memory parallel reduction. 256 threads per batch element.
// Dispatch: (batch, 1, 1)

RWStructuredBuffer<float> Sums : register(u0);  // output: batch * 3 floats
StructuredBuffer<float> Pred : register(t0);
StructuredBuffer<float> Target : register(t1);

cbuffer Params : register(b0) {
    uint batch;
    uint size;  // C * H * W per sample
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
        Sums[b * 3 + 0] = s_inter[0];
        Sums[b * 3 + 1] = s_pred[0];
        Sums[b * 3 + 2] = s_tgt[0];
    }
}
