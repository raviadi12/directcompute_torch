RWStructuredBuffer<float> DX : register(u0);
StructuredBuffer<float> DY : register(t0);

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
    uint total = batch * C * inH * inW;
    if (index >= total) return;
    
    uint w_in = index % inW;
    uint temp = index / inW;
    uint h_in = temp % inH;
    temp = temp / inH;
    uint c = temp % C;
    uint b = temp / C;
    
    float sum = 0.0f;
    uint h_start = h_in * scaleH;
    uint w_start = w_in * scaleW;
    
    for (uint sh = 0; sh < scaleH; ++sh) {
        for (uint sw = 0; sw < scaleW; ++sw) {
            uint h_out = h_start + sh;
            uint w_out = w_start + sw;
            
            if (h_out < outH && w_out < outW) {
                uint out_idx = ((b * C + c) * outH + h_out) * outW + w_out;
                sum += DY[out_idx];
            }
        }
    }
    
    DX[index] = sum;
}
