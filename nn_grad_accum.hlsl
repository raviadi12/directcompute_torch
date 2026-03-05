cbuffer Params : register(b0) { uint count; uint pad1, pad2, pad3; };
StructuredBuffer<float> grad : register(t0);
RWStructuredBuffer<float> accum : register(u0);
[numthreads(256, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x < count) {
        accum[id.x] = accum[id.x] + grad[id.x];
    }
}
