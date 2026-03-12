RWStructuredBuffer<float> DX : register(u0);
StructuredBuffer<float> DY : register(t0);

cbuffer Params : register(b0) {
    uint batch;
    uint c_in;
    uint spatial_size;
    uint c_out;
    uint c_offset;
};

[numthreads(256, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint index = dtid.x;
    uint total = batch * c_in * spatial_size;
    if (index >= total) return;
    
    uint s = index % spatial_size;
    uint temp = index / spatial_size;
    uint c = temp % c_in;
    uint b = temp / c_in;
    
    uint in_c = c + c_offset;
    uint in_idx = (b * c_out + in_c) * spatial_size + s;
    
    DX[index] = DY[in_idx];
}
