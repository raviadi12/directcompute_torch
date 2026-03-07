// Global Average Pooling Backward: (N, C, 1, 1) -> (N, C, H, W)
// Each output gradient element = grad_output[n,c] / (H*W)
cbuffer Params : register(b0) {
    uint batch;
    uint C;
    uint S;  // H * W
    uint pad1;
};

StructuredBuffer<float> grad_output : register(t0);
RWStructuredBuffer<float> dx        : register(u0);

[numthreads(256, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint total = batch * C * S;
    if (dtid.x >= total) return;

    uint nc = dtid.x / S;
    dx[dtid.x] = grad_output[nc] / (float)S;
}
