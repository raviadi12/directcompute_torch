cbuffer ConvParams : register(b0) {
    uint batch;
    uint inC;
    uint inH;
    uint inW;
    uint outC;
    uint ks;
    uint outH;
    uint outW;
    uint actType;
};

StructuredBuffer<float> input : register(t0);
StructuredBuffer<float> filters : register(t1);
StructuredBuffer<float> bias : register(t2);
RWStructuredBuffer<float> output : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 gid : SV_DispatchThreadID) {
    uint ow = gid.x % outW;
    uint oh = (gid.x / outW) % outH;
    uint b = gid.x / (outW * outH);
    uint oc = gid.y;

    if (b >= batch || oc >= outC) return;

    float acc = 0.0;
    uint filter_offset = oc * inC * ks * ks;
    uint input_batch_offset = b * inC * inH * inW;

    for (uint ic = 0; ic < inC; ++ic) {
        uint input_chan_offset = input_batch_offset + ic * inH * inW;
        uint filter_chan_offset = filter_offset + ic * ks * ks;
        for (uint kh = 0; kh < ks; ++kh) {
            for (uint kw = 0; kw < ks; ++kw) {
                acc += input[input_chan_offset + (oh + kh) * inW + (ow + kw)] * filters[filter_chan_offset + kh * ks + kw];
            }
        }
    }

    float z = acc + bias[oc];
    float a = z;
    if (actType == 1) a = max(0.0, z);
    else if (actType == 2) a = tanh(z);
    
    output[b * (outC * outH * outW) + oc * (outH * outW) + oh * outW + ow] = a;
}
