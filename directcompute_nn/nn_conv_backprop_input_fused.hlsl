cbuffer ConvParams : register(b0) {
    uint batch;
    uint inC;
    uint inH;
    uint inW;
    uint outC;
    uint ks;
    uint outH;
    uint outW;
    uint prevActType;
};

StructuredBuffer<float> dZ : register(t0);
StructuredBuffer<float> filters : register(t1);
StructuredBuffer<float> A_prev : register(t2);
RWStructuredBuffer<float> dInput : register(u0);

[numthreads(256, 1, 1)]
void CSMain(uint3 gid : SV_DispatchThreadID) {
    uint totalIn = batch * inC * inH * inW;
    if (gid.x >= totalIn) return;

    uint iw = gid.x % inW;
    uint ih = (gid.x / inW) % inH;
    uint ic = (gid.x / (inW * inH)) % inC;
    uint b = gid.x / (inC * inH * inW);

    float acc = 0.0;
    
    for (uint oc = 0; oc < outC; ++oc) {
        for (uint kh = 0; kh < ks; ++kh) {
            int oh = (int)ih - (int)kh;
            if (oh >= 0 && oh < (int)outH) {
                for (uint kw = 0; kw < ks; ++kw) {
                    int ow = (int)iw - (int)kw;
                    if (ow >= 0 && ow < (int)outW) {
                        acc += dZ[b * (outC * outH * outW) + oc * (outH * outW) + (uint)oh * outW + (uint)ow] * 
                               filters[oc * (inC * ks * ks) + ic * (ks * ks) + kh * ks + kw];
                    }
                }
            }
        }
    }

    float a = A_prev[gid.x];
    float dA = 1.0;
    if (prevActType == 1) dA = (a > 0.0) ? 1.0 : 0.0;
    else if (prevActType == 2) dA = 1.0 - (a * a);
    
    dInput[gid.x] = acc * dA;
}
