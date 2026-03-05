cbuffer PoolParams : register(b0) {
    uint batch;
    uint inC;
    uint inH;
    uint inW;
    uint p_size;
    uint stride;
    uint outH;
    uint outW;
};

StructuredBuffer<float> input : register(t0);
RWStructuredBuffer<float> output : register(u0);
RWStructuredBuffer<float> indices : register(u1);  // stores uint indices as float bits

[numthreads(16, 16, 1)]
void CSMain(uint3 gid : SV_DispatchThreadID) {
    uint ow = gid.x;
    uint oh = gid.y;
    uint bc = gid.z;

    if (ow >= outW || oh >= outH || bc >= batch * inC) return;

    uint b = bc / inC;
    uint c = bc % inC;

    float maxVal = -1e30;
    uint maxIdx = 0xFFFFFFFF;

    uint in_chan_offset = b * (inC * inH * inW) + c * (inH * inW);
    uint start_h = oh * stride;
    uint start_w = ow * stride;

    for (uint ph = 0; ph < p_size; ++ph) {
        for (uint pw = 0; pw < p_size; ++pw) {
            uint ih = start_h + ph;
            uint iw = start_w + pw;
            if (ih < inH && iw < inW) {
                uint idx = in_chan_offset + ih * inW + iw;
                float val = input[idx];
                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = idx;
                }
            }
        }
    }

    uint out_idx = bc * (outH * outW) + oh * outW + ow;
    output[out_idx] = maxVal;
    indices[out_idx] = asfloat(maxIdx);  // store uint index as float bits
}
