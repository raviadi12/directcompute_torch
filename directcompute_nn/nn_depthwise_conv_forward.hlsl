// Depthwise Conv2d Forward: each channel has its own filter
// Input: (N, C, H, W), Filter: (C, 1, kH, kW), Bias: (C)
// Output: (N, C, outH, outW)
cbuffer Params : register(b0) {
    uint batch;
    uint C;
    uint inH;
    uint inW;
    uint kH;
    uint kW;
    uint stride;
    uint padding;
    uint outH;
};

StructuredBuffer<float> input   : register(t0);
StructuredBuffer<float> filters : register(t1);
StructuredBuffer<float> bias    : register(t2);
RWStructuredBuffer<float> output : register(u0);

[numthreads(256, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint outW_ = (inW + 2 * padding - kW) / stride + 1;
    uint total = batch * C * outH * outW_;
    if (dtid.x >= total) return;

    uint ow = dtid.x % outW_;
    uint oh = (dtid.x / outW_) % outH;
    uint c  = (dtid.x / (outW_ * outH)) % C;
    uint n  = dtid.x / (outW_ * outH * C);

    float sum = 0.0f;
    uint filterBase = c * kH * kW;
    uint inputBase  = n * (C * inH * inW) + c * (inH * inW);

    for (uint fh = 0; fh < kH; ++fh) {
        int ih = (int)(oh * stride + fh) - (int)padding;
        if (ih < 0 || ih >= (int)inH) continue;
        for (uint fw = 0; fw < kW; ++fw) {
            int iw = (int)(ow * stride + fw) - (int)padding;
            if (iw < 0 || iw >= (int)inW) continue;
            sum += input[inputBase + ih * inW + iw] * filters[filterBase + fh * kW + fw];
        }
    }

    output[dtid.x] = sum + bias[c];
}
