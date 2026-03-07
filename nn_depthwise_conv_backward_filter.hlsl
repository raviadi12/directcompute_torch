// Depthwise Conv2d Backward w.r.t. Filter and Bias
// Dispatched with (C) thread groups, 256 threads each reduce over N*outH*outW
// grad_output: (N, C, outH, outW), input: (N, C, inH, inW)
// dfilter: (C, 1, kH, kW), dbias: (C)
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
StructuredBuffer<float> input       : register(t1);
RWStructuredBuffer<float> dfilter   : register(u0);
RWStructuredBuffer<float> dbias     : register(u1);

groupshared float s_sum[256];

[numthreads(256, 1, 1)]
void CSMain(uint3 tid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint c = gid.x;
    uint outW_ = (inW + 2 * padding - kW) / stride + 1;
    uint numOut = batch * outH * outW_;

    // Accumulate bias gradient
    float my_dbias = 0.0f;
    for (uint i = tid.x; i < numOut; i += 256) {
        uint n  = i / (outH * outW_);
        uint rem = i % (outH * outW_);
        my_dbias += grad_output[n * (C * outH * outW_) + c * (outH * outW_) + rem];
    }
    s_sum[tid.x] = my_dbias;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid.x < s) s_sum[tid.x] += s_sum[tid.x + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid.x == 0) dbias[c] = s_sum[0];
    GroupMemoryBarrierWithGroupSync();

    // Accumulate filter gradients — one kernel element at a time
    for (uint fh = 0; fh < kH; ++fh) {
        for (uint fw = 0; fw < kW; ++fw) {
            float my_df = 0.0f;
            for (uint i = tid.x; i < numOut; i += 256) {
                uint n   = i / (outH * outW_);
                uint rem = i % (outH * outW_);
                uint oh  = rem / outW_;
                uint ow  = rem % outW_;

                int ih = (int)(oh * stride + fh) - (int)padding;
                int iw = (int)(ow * stride + fw) - (int)padding;

                float g = grad_output[n * (C * outH * outW_) + c * (outH * outW_) + rem];
                float x = 0.0f;
                if (ih >= 0 && ih < (int)inH && iw >= 0 && iw < (int)inW)
                    x = input[n * (C * inH * inW) + c * (inH * inW) + ih * inW + iw];

                my_df += g * x;
            }
            s_sum[tid.x] = my_df;
            GroupMemoryBarrierWithGroupSync();
            for (uint s2 = 128; s2 > 0; s2 >>= 1) {
                if (tid.x < s2) s_sum[tid.x] += s_sum[tid.x + s2];
                GroupMemoryBarrierWithGroupSync();
            }
            if (tid.x == 0) dfilter[c * kH * kW + fh * kW + fw] = s_sum[0];
            GroupMemoryBarrierWithGroupSync();
        }
    }
}
