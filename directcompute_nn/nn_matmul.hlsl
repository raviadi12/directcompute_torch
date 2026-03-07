cbuffer Params : register(b0) { uint M, K, N, pad; };
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);
[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x < N && id.y < M) {
        float sum = 0;
        for (uint i = 0; i < K; ++i) sum += A[id.y * K + i] * B[i * N + id.x];
        C[id.y * N + id.x] = sum;
    }
}
