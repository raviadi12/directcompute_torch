// Counts correct predictions entirely on GPU (no CPU readback needed per batch)
// Input: logits (batch_size * num_classes), labels (batch_size)
// Output: correct_count[0] (atomically incremented)
cbuffer Params : register(b0) { uint batch_size, num_classes; };
StructuredBuffer<float> logits : register(t0);
StructuredBuffer<float> labels : register(t1);
RWStructuredBuffer<uint> correct_count : register(u0);

[numthreads(256, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= batch_size) return;
    
    uint label = (uint)labels[i];
    float max_val = logits[i * num_classes];
    uint max_idx = 0;
    for (uint j = 1; j < num_classes; j++) {
        float v = logits[i * num_classes + j];
        if (v > max_val) {
            max_val = v;
            max_idx = j;
        }
    }
    if (max_idx == label) {
        InterlockedAdd(correct_count[0], 1);
    }
}
