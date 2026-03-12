RWStructuredBuffer<float> Out : register(u0);
StructuredBuffer<float> X : register(t0);

cbuffer Params : register(b0) {
    uint batch;
    uint C;
    uint inH;
    uint inW;
    uint outH;
    uint outW;
    uint scaleH;
    uint scaleW;
};

[numthreads(256, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint index = dtid.x;
    uint total = batch * C * outH * outW;
    if (index >= total) return;
    
    uint w_out = index % outW;
    uint temp = index / outW;
    uint h_out = temp % outH;
    temp = temp / outH;
    uint c = temp % C;
    uint b = temp / C;
    
    uint h_in = h_out / scaleH;
    uint w_in = w_out / scaleW;
    
    uint in_idx = ((b * C + c) * inH + h_in) * inW + w_in;
    Out[index] = X[in_idx];
}
