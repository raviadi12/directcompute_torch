StructuredBuffer<float> x : register(t0);
StructuredBuffer<float> mean : register(t1);
StructuredBuffer<float> var : register(t2);
StructuredBuffer<float> gamma : register(t3);
StructuredBuffer<float> beta : register(t4);

RWStructuredBuffer<float> y : register(u0);

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
    
    float m = mean[c];
    float v = var[c];
    float g = gamma[c];
    float b = beta[c];
    float eps = asfloat(eps_bits);
    
    float rstd = 1.0 / sqrt(v + eps);
    float x_hat = (x[idx] - m) * rstd;
    
    y[idx] = x_hat * g + b;
}
