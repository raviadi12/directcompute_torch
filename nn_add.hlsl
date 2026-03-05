cbuffer Params : register(b0) { uint M, N, pad1, pad2; };
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);
[numthreads(256, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x < M * N) C[id.x] = A[id.x] + B[id.x];
}
