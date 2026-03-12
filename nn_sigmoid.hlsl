RWStructuredBuffer<float> Out : register(u0);
StructuredBuffer<float> X : register(t0);

cbuffer Params : register(b0) {
    uint count;
};

[numthreads(256, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= count) return;
    float x = X[dtid.x];
    Out[dtid.x] = 1.0f / (1.0f + exp(-x));
}
