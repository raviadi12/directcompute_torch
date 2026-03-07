cbuffer Params : register(b0) { uint count; float alpha; float beta; uint pad; };
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> out : register(u0);

[numthreads(256, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x < count) {
        out[id.x] = alpha * A[id.x] + beta * B[id.x];
    }
}
