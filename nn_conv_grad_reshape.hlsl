cbuffer Params : register(b0) {
    uint batch;
    uint inC;
    uint inH;
    uint inW;
    uint ks;
    uint stride;
    uint padding;
    uint outH;
    uint outW;
};

StructuredBuffer<float> dZ : register(t0);
RWStructuredBuffer<float> output : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 gid : SV_DispatchThreadID) {
    uint oc = gid.y;
    uint b_pixel = gid.x;
    uint B_OH_OW = batch * outH * outW;
    
    if (oc >= ks || b_pixel >= B_OH_OW) return; // Note: ks is reused as outC here for simplicity in CB
    
    uint b = b_pixel / (outH * outW);
    uint pixel = b_pixel % (outH * outW);
    
    output[oc * B_OH_OW + b_pixel] = dZ[b * (ks * outH * outW) + oc * (outH * outW) + pixel];
}
