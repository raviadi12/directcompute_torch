// Fused add_bias + ReLU: C[i] = max(0, A[i] + B[col])
// Saves one dispatch + one buffer allocation vs separate add_bias → relu
cbuffer Params : register(b0) { uint M, N, pad1, pad2; };
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);
[numthreads(256, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x < M * N) {
        uint col = id.x % N;
        C[id.x] = max(0.0f, A[id.x] + B[col]);
    }
}
