RWStructuredBuffer<float> Out : register(u0);
StructuredBuffer<float> X : register(t0);

cbuffer Params : register(b0) {
    uint count;
    uint alpha_as_uint;
};

[numthreads(256, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= count) return;
    float x = X[dtid.x];
    float alpha = asfloat(alpha_as_uint);
    Out[dtid.x] = x > 0.0f ? x : x * alpha;
}
