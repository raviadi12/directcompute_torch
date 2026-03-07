StructuredBuffer<float> x : register(t0);
StructuredBuffer<float> dy : register(t1);
StructuredBuffer<float> batch_mean : register(t2);
StructuredBuffer<float> batch_var : register(t3);
StructuredBuffer<float> gamma : register(t4);
StructuredBuffer<float> dgamma : register(t5);
StructuredBuffer<float> dbeta : register(t6);

RWStructuredBuffer<float> dx : register(u0);

cbuffer Params : register(b0) {
    uint N;
    uint C;
    uint S;
    uint eps_bits;
    uint dummy1; uint dummy2; uint dummy3; uint dummy4; uint dummy5;
};

[numthreads(256, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint total_elements = N * C * S;
    uint idx = dtid.x;
    if (idx >= total_elements) return;
    
    uint c = (idx / S) % C;
    
    float mean = batch_mean[c];
    float var = batch_var[c];
    float eps = asfloat(eps_bits);
    float rstd = 1.0 / sqrt(var + eps);
    
    float x_hat = (x[idx] - mean) * rstd;
    float g = gamma[c];
    float dg = dgamma[c];
    float db = dbeta[c];
    float m = (float)(N * S);
    
    dx[idx] = (g * rstd) * (dy[idx] - (db + x_hat * dg) / m);
}
