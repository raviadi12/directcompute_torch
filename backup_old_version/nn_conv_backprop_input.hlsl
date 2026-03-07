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

[numthreads(64, 1, 1)]
void CSMain(uint3 gid : SV_DispatchThreadID) {
    uint totalIn = batch * inC * inH * inW;
    if (gid.x >= totalIn) return;

    uint iw = gid.x % inW;
    uint ih = (gid.x / inW) % inH;
    uint ic = (gid.x / (inW * inH)) % inC;
    uint m = gid.x / (inC * inH * inW);

    float acc = 0.0;
    uint m_dz_offset = m * (outC * outH * outW);
    uint ic_f_offset = ic * (ks * ks);

    for (uint oc = 0; oc < outC; ++oc) {
        uint oc_dz_offset = m_dz_offset + oc * (outH * outW);
        uint oc_f_offset = oc * (inC * ks * ks) + ic_f_offset;
        for (uint kh = 0; kh < ks; ++kh) {
            uint oh_i = ih - kh;
            if (oh_i < outH) {
                for (uint kw = 0; kw < ks; ++kw) {
                    uint ow_i = iw - kw;
                    if (ow_i < outW) {
                        acc += dZ[oc_dz_offset + oh_i * outW + ow_i] * filters[oc_f_offset + kh * ks + kw];
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
