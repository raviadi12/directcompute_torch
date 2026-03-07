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

[numthreads(16, 16, 1)]
void CSMain(uint3 gid : SV_DispatchThreadID) {
    uint totalCol = batch * outH * outW;
    uint totalRow = inC * ks * ks;
    
    if (gid.x >= totalCol || gid.y >= totalRow) return;
    
    uint b = gid.x / (outH * outW);
    uint pixelIdx = gid.x % (outH * outW);
    uint oh = pixelIdx / outW;
    uint ow = pixelIdx % outW;
    
    uint ic = gid.y / (ks * ks);
    uint kIdx = gid.y % (ks * ks);
    uint kh = kIdx / ks;
    uint kw = kIdx % ks;
    
    int ih = (int)(oh * stride + kh) - (int)padding;
    int iw = (int)(ow * stride + kw) - (int)padding;
    
    float val = 0.0f;
    if (ih >= 0 && ih < (int)inH && iw >= 0 && iw < (int)inW) {
        val = input[b * (inC * inH * inW) + ic * (inH * inW) + ih * inW + iw];
    }
    
    output[gid.y * totalCol + gid.x] = val;
}
