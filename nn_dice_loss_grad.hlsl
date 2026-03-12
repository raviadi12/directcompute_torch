// Two-pass dice loss backward: Pass 2
// Reads precomputed per-batch sums from Sums buffer (computed by dice_loss_sums).
// Each thread handles one element — O(1) work per thread, no loops.
// Dispatch: ((batch*size+255)/256, 1, 1)

RWStructuredBuffer<float> DX : register(u0);
StructuredBuffer<float> Pred : register(t0);
StructuredBuffer<float> Target : register(t1);
StructuredBuffer<float> Sums : register(t2);  // batch * 3 floats: {inter, sum_p, sum_t} per batch

cbuffer Params : register(b0) {
    uint batch;
    uint size; // C * H * W
};

[numthreads(256, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint index = dtid.x;
    uint total = batch * size;
    if (index >= total) return;

    uint b = index / size;

    float intersection = Sums[b * 3 + 0];
    float sum_pred     = Sums[b * 3 + 1];
    float sum_target   = Sums[b * 3 + 2];

    float denom = sum_pred + sum_target + 1e-6f;
    float p_i = Pred[index];
    float t_i = Target[index];

    float dL_dp = -2.0f * ((t_i * denom) - intersection) / (denom * denom);
    DX[index] = dL_dp / (float)batch;
}
