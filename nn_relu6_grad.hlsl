StructuredBuffer<float> grad_output : register(t0);
StructuredBuffer<float> fwd_input  : register(t1);
RWStructuredBuffer<float> grad_input : register(u0);
cbuffer Params : register(b0) { uint count; uint pad1; uint pad2; uint pad3; };

[numthreads(256, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x < count) {
        float x = fwd_input[id.x];
        grad_input[id.x] = (x > 0.0f && x < 6.0f) ? grad_output[id.x] : 0.0f;
    }
}
