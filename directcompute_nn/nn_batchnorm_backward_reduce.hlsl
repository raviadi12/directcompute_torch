StructuredBuffer<float> x : register(t0);
StructuredBuffer<float> dy : register(t1);
StructuredBuffer<float> batch_mean : register(t2);
StructuredBuffer<float> batch_var : register(t3);

RWStructuredBuffer<float> dgamma : register(u0);
RWStructuredBuffer<float> dbeta : register(u1);

cbuffer Params : register(b0) {
    uint N;
    uint C;
    uint S;
    uint eps_bits;
    uint dummy1; uint dummy2; uint dummy3; uint dummy4; uint dummy5;
};

groupshared float s_dgamma[256];
groupshared float s_dbeta[256];

[numthreads(256, 1, 1)]
void CSMain(uint3 tid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint c = gid.x;
    float my_dg = 0;
    float my_db = 0;
    uint num_elements = N * S;
    
    float mean = batch_mean[c];
    float var = batch_var[c];
    float eps = asfloat(eps_bits);
    float rstd = 1.0 / sqrt(var + eps);
    
    for(uint i = tid.x; i < num_elements; i += 256) {
        uint n = i / S;
        uint spatial = i % S;
        uint idx = n * C * S + c * S + spatial;
        
        float val_x = x[idx];
        float val_dy = dy[idx];
        
        float x_hat = (val_x - mean) * rstd;
        my_dg += val_dy * x_hat;
        my_db += val_dy;
    }
    
    s_dgamma[tid.x] = my_dg;
    s_dbeta[tid.x] = my_db;
    GroupMemoryBarrierWithGroupSync();
    
    for(uint s = 128; s > 0; s >>= 1) {
        if(tid.x < s) {
            s_dgamma[tid.x] += s_dgamma[tid.x + s];
            s_dbeta[tid.x] += s_dbeta[tid.x + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    
    if(tid.x == 0) {
        dgamma[c] = s_dgamma[0];
        dbeta[c] = s_dbeta[0];
    }
}
