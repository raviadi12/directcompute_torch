StructuredBuffer<float> input : register(t0);
RWStructuredBuffer<float> output : register(u0);
cbuffer Params : register(b0) { uint count; uint pad1; uint pad2; uint pad3; };

[numthreads(256, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x < count) {
        float x = input[id.x];
        output[id.x] = min(max(x, 0.0f), 6.0f);
    }
}
