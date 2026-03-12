RWStructuredBuffer<float> DX : register(u0);
StructuredBuffer<float> DY : register(t0);
StructuredBuffer<float> X : register(t1);

cbuffer Params : register(b0) {
    uint count;
    uint alpha_as_uint;
};

[numthreads(256, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= count) return;
    float x = X[dtid.x];
    float dy = DY[dtid.x];
    float alpha = asfloat(alpha_as_uint);
    DX[dtid.x] = dy * (x > 0.0f ? 1.0f : alpha);
}
