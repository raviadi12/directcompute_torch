StructuredBuffer<float> x : register(t0);
RWStructuredBuffer<float> batch_mean : register(u0);
RWStructuredBuffer<float> batch_var : register(u1);
RWStructuredBuffer<float> running_mean : register(u2);
RWStructuredBuffer<float> running_var : register(u3);

cbuffer Params : register(b0) {
    uint N;
    uint C;
    uint S;
    uint is_training; // 1 or 0
    uint momentum_bits;
    uint dummy1; uint dummy2; uint dummy3; uint dummy4;
};

groupshared float s_sum[256];
groupshared float s_sum_sq[256];

[numthreads(256, 1, 1)]
void CSMain(uint3 tid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint c = gid.x;
    float my_x = 0;
    float my_x2 = 0;
    uint num_elements = N * S;
    
    for(uint i = tid.x; i < num_elements; i += 256) {
        uint n = i / S;
        uint spatial = i % S;
        float v = x[n * C * S + c * S + spatial];
        my_x += v;
        my_x2 += v * v;
    }
    
    s_sum[tid.x] = my_x;
    s_sum_sq[tid.x] = my_x2;
    GroupMemoryBarrierWithGroupSync();
    
    for(uint s = 128; s > 0; s >>= 1) {
        if(tid.x < s) {
            s_sum[tid.x] += s_sum[tid.x + s];
            s_sum_sq[tid.x] += s_sum_sq[tid.x + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    
    if(tid.x == 0) {
        float mean = s_sum[0] / num_elements;
        float x2_mean = s_sum_sq[0] / num_elements;
        float var = max(0.0, x2_mean - mean * mean);
        
        batch_mean[c] = mean;
        batch_var[c] = var;
        
        if (is_training == 1) {
            float momentum = asfloat(momentum_bits);
            float unbiased_var = var * ((float)num_elements / max(1.0, (float)(num_elements - 1)));
            running_mean[c] = (1.0 - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1.0 - momentum) * running_var[c] + momentum * unbiased_var;
        }
    }
}
