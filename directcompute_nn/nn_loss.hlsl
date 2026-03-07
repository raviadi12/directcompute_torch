// Parallel cross-entropy loss with shared memory reduction
// Each thread processes one or more batch items, then tree-reduce
cbuffer Params : register(b0) { uint batch_size, num_classes, pad1, pad2; };
StructuredBuffer<float> logits : register(t0);
StructuredBuffer<float> labels : register(t1);
RWStructuredBuffer<float> loss : register(u0);

groupshared float partial[256];

[numthreads(256, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    float my_loss = 0.0f;

    // Each thread handles batch items at stride 256
    for (uint i = lid.x; i < batch_size; i += 256) {
        uint label = (uint)labels[i];
        uint base = i * num_classes;

        float max_val = -1e30f;
        for (uint j = 0; j < num_classes; ++j)
            max_val = max(max_val, logits[base + j]);

        float sum_exp = 0.0f;
        for (uint k = 0; k < num_classes; ++k)
            sum_exp += exp(logits[base + k] - max_val);

        my_loss += -(logits[base + label] - max_val - log(sum_exp));
    }

    partial[lid.x] = my_loss;
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction
    [unroll]
    for (uint s = 128; s > 0; s >>= 1) {
        if (lid.x < s)
            partial[lid.x] += partial[lid.x + s];
        GroupMemoryBarrierWithGroupSync();
    }

    if (lid.x == 0) {
        loss[0] = partial[0] / (float)batch_size;
    }
}
