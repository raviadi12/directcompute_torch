RWStructuredBuffer<float> Out : register(u0);
StructuredBuffer<float> X : register(t0);

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
    
    uint out_c = c + c_offset;
    uint out_idx = (b * c_out + out_c) * spatial_size + s;
    
    Out[out_idx] = X[index];
}
