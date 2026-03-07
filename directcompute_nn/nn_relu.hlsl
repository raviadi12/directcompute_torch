StructuredBuffer<float> A : register(t0);
RWStructuredBuffer<float> B : register(u0);
[numthreads(256, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    B[id.x] = max(0.0f, A[id.x]);
}
