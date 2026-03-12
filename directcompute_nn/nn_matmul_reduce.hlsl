// Parallel-Reduction MatMul for small M×N but large K
// Used for conv backward dFilter: dW = grad_reshaped × im2col^T
//
// Problem: Normal matmul parallelizes over M×N output elements.
//          When M=6, N=25, that's only 2 thread groups → 95% GPU idle!
//
// Solution: Each thread GROUP computes ONE output element C[m,n].
//           All 256 threads in the group cooperatively reduce over K.
//           Dispatch(N, M, 1) → one group per output element.
//
// For Conv1 (M=6, N=25, K=18432):
//   Old: Dispatch(2, 1, 1) = 512 threads       → 4611 µs
//   New: Dispatch(25, 6, 1) = 38,400 threads    → 2204 µs (2.1× faster)

cbuffer Params : register(b0) { uint M, K, N, flags; };

StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);

#define THREADS 256

groupshared float shared_sum[THREADS];

[numthreads(THREADS, 1, 1)]
void CSMain(uint3 tid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint m = gid.y;  // output row (outC)
    uint n = gid.x;  // output col (totalRow = inC*kH*kW)
    
    float sum = 0.0f;
    
    if (m < M && n < N) {
        // Each of 256 threads handles a strided portion of K
        // Thread 0: k=0, 256, 512, ...
        // Thread 1: k=1, 257, 513, ...
        for (uint k = tid.x; k < K; k += THREADS) {
            // flags & 1: A is transposed (read A[k,m] = A[k*M+m])
            // flags & 2: B is transposed (read B[n,k] = B[n*K+k])
            float a = (flags & 1) ? A[k * M + m] : A[m * K + k];
            float b = (flags & 2) ? B[n * K + k] : B[k * N + n];
            sum += a * b;
        }
    }
    
    shared_sum[tid.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    
    // Parallel tree reduction in shared memory
    // 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
    [unroll]
    for (uint s = THREADS / 2; s > 0; s >>= 1) {
        if (tid.x < s) {
            shared_sum[tid.x] += shared_sum[tid.x + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    
    // Thread 0 writes the final reduced result
    if (tid.x == 0 && m < M && n < N) {
        C[m * N + n] = shared_sum[0];
    }
}
