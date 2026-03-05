cbuffer Params : register(b0) { uint count, pad1, pad2, pad3; };
StructuredBuffer<float> grad_out : register(t0);
StructuredBuffer<float> input : register(t1);
RWStructuredBuffer<float> grad_in : register(u0);
[numthreads(256, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x < count) {
        grad_in[id.x] = input[id.x] > 0 ? grad_out[id.x] : 0;
    }
}
