cbuffer Params : register(b0) { uint batch_size, num_classes, pad1, pad2; };
StructuredBuffer<float> input : register(t0);
RWStructuredBuffer<float> output : register(u0);
[numthreads(16, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x < batch_size) {
        uint base = id.x * num_classes;
        float max_val = -1e30;
        for (uint i = 0; i < num_classes; ++i) max_val = max(max_val, input[base + i]);
        float sum = 0;
        for (uint j = 0; j < num_classes; ++j) sum += exp(input[base + j] - max_val);
        for (uint k = 0; k < num_classes; ++k) output[base + k] = exp(input[base + k] - max_val) / sum;
    }
}
