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

StructuredBuffer<float> input : register(t0);
RWStructuredBuffer<float> output : register(u0);

[numthreads(256, 1, 1)]
void CSMain(uint3 gid : SV_DispatchThreadID) {
    uint totalIn = batch * inC * inH * inW;
    if (gid.x >= totalIn) return;
    
    uint iw = gid.x % inW;
    uint ih = (gid.x / inW) % inH;
    uint ic = (gid.x / (inW * inH)) % inC;
    uint b = gid.x / (inC * inH * inW);
    
    float sum = 0.0f;
    uint totalCol = batch * outH * outW;
    
    for (uint kh = 0; kh < ks; ++kh) {
        for (uint kw = 0; kw < ks; ++kw) {
            int oh_s = (int)ih - (int)kh + (int)padding;
            int ow_s = (int)iw - (int)kw + (int)padding;
            
            if (oh_s % (int)stride == 0 && ow_s % (int)stride == 0) {
                uint oh = (uint)(oh_s / (int)stride);
                uint ow = (uint)(ow_s / (int)stride);
                
                if (oh < outH && ow < outW) {
                    uint row = ic * (ks * ks) + kh * ks + kw;
                    uint col = b * (outH * outW) + oh * outW + ow;
                    sum += input[row * totalCol + col];
                }
            }
        }
    }
    
    output[gid.x] = sum;
}
