cbuffer ConvParams : register(b0) {
    uint batch;
    uint inC;
    uint inH;
    uint inW;
    uint outC;
    uint ks;
    uint outH;
    uint outW;
};

StructuredBuffer<float> dZ : register(t0);
StructuredBuffer<float> A_prev : register(t1);
RWStructuredBuffer<float> partial_dF : register(u0);
RWStructuredBuffer<float> partial_dB : register(u1);

[numthreads(64, 1, 1)]
void CSMain(uint3 gid : SV_DispatchThreadID) {
    uint totalFilters = outC * inC * ks * ks;
    if (gid.x >= batch * totalFilters) return;

    uint m = gid.x / totalFilters;
    uint fIdx = gid.x % totalFilters;

    uint kw = fIdx % ks;
    uint kh = (fIdx / ks) % ks;
    uint ic = (fIdx / (ks * ks)) % inC;
    uint oc = fIdx / (inC * ks * ks);

    float acc = 0.0;
    uint m_dz_offset = m * (outC * outH * outW) + oc * (outH * outW);
    uint m_in_offset = m * (inC * inH * inW) + ic * (inH * inW);

    for (uint oh = 0; oh < outH; ++oh) {
        for (uint ow = 0; ow < outW; ++ow) {
            acc += dZ[m_dz_offset + oh * outW + ow] * A_prev[m_in_offset + (oh + kh) * inW + (ow + kw)];
        }
    }
    partial_dF[gid.x] = acc;

    if (ic == 0 && kh == 0 && kw == 0) {
        float bAcc = 0.0;
        for (uint bh = 0; bh < outH; ++bh) {
            for (uint bw = 0; bw < outW; ++bw) {
                bAcc += dZ[m_dz_offset + bh * outW + bw];
            }
        }
        partial_dB[m * outC + oc] = bAcc;
    }
}
