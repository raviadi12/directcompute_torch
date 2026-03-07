cbuffer ConvParams : register(b0) {
    uint inC;
    uint outC;
    uint ks;
    uint batch;
};

StructuredBuffer<float> partial_dF : register(t0);
StructuredBuffer<float> partial_dB : register(t1);
RWStructuredBuffer<float> dF : register(u0);
RWStructuredBuffer<float> dB : register(u1);

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint totalFilters = outC * inC * ks * ks;
    if (id.x < totalFilters) {
        float sumF = 0.0;
        for (uint m = 0; m < batch; ++m) {
            sumF += partial_dF[m * totalFilters + id.x];
        }
        dF[id.x] = sumF;
    }

    if (id.x < outC) {
        float sumB = 0.0;
        for (uint m = 0; m < batch; ++m) {
            sumB += partial_dB[m * outC + id.x];
        }
        dB[id.x] = sumB;
    }
}
