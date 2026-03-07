// Depthwise Conv2d Backward w.r.t. Input
// grad_output: (N, C, outH, outW), filters: (C, 1, kH, kW)
// dx: (N, C, inH, inW)
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

StructuredBuffer<float> grad_output : register(t0);
StructuredBuffer<float> filters     : register(t1);
RWStructuredBuffer<float> dx        : register(u0);

[numthreads(256, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint outW_ = (inW + 2 * padding - kW) / stride + 1;
    uint total = batch * C * inH * inW;
    if (dtid.x >= total) return;

    uint iw = dtid.x % inW;
    uint ih = (dtid.x / inW) % inH;
    uint c  = (dtid.x / (inW * inH)) % C;
    uint n  = dtid.x / (inW * inH * C);

    float sum = 0.0f;
    uint filterBase = c * kH * kW;
    uint gradBase   = n * (C * outH * outW_) + c * (outH * outW_);

    for (uint fh = 0; fh < kH; ++fh) {
        int oh_s = (int)ih - (int)fh + (int)padding;
        if (oh_s < 0 || oh_s % (int)stride != 0) continue;
        uint oh = (uint)(oh_s / (int)stride);
        if (oh >= outH) continue;

        for (uint fw = 0; fw < kW; ++fw) {
            int ow_s = (int)iw - (int)fw + (int)padding;
            if (ow_s < 0 || ow_s % (int)stride != 0) continue;
            uint ow = (uint)(ow_s / (int)stride);
            if (ow >= outW_) continue;

            sum += grad_output[gradBase + oh * outW_ + ow] * filters[filterBase + fh * kW + fw];
        }
    }

    dx[dtid.x] = sum;
}
