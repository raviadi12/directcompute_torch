cbuffer Params : register(b0) { uint batch_size, num_classes, pad1, pad2; };
StructuredBuffer<float> softmax_out : register(t0);
StructuredBuffer<float> labels : register(t1);
RWStructuredBuffer<float> grad_in : register(u0);
[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x < num_classes && id.y < batch_size) {
        uint idx = id.y * num_classes + id.x;
        float label = (labels[id.y] == (float)id.x) ? 1.0f : 0.0f;
        grad_in[idx] = (softmax_out[idx] - label) / (float)batch_size;
    }
}
