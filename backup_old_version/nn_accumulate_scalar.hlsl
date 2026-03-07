// Adds a scalar (float) from src[0] to dst[0]
// Used to accumulate loss on GPU across batches without CPU readback
StructuredBuffer<float> src : register(t0);
RWStructuredBuffer<float> dst : register(u0);

[numthreads(1, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    dst[0] = dst[0] + src[0];
}
