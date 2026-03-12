RWStructuredBuffer<float> DX : register(u0);
StructuredBuffer<float> DY : register(t0);
StructuredBuffer<float> OutVal : register(t1);

cbuffer Params : register(b0) {
    uint count;
};

[numthreads(256, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= count) return;
    float dy = DY[dtid.x];
    float y = OutVal[dtid.x];
    DX[dtid.x] = dy * y * (1.0f - y);
}
