// Fused conv_grad_reshape + ReLU gradient
// Combines: (1) reshape grad from NCHW to (outC, B*OH*OW) for matmul
//           (2) apply ReLU gradient mask from forward output
// Saves one dispatch + buffer vs separate relu_grad → conv_grad_reshape
cbuffer Params : register(b0) {
    uint batch;
    uint inC;
    uint inH;
    uint inW;
    uint outC;      // stored in ks slot (u5) for CB layout compat
    uint stride;
    uint padding;
    uint outH;
    uint outW;
};

StructuredBuffer<float> dZ : register(t0);          // gradient in NCHW layout
StructuredBuffer<float> fwd_output : register(t1);  // post-relu forward output (NCHW)
RWStructuredBuffer<float> output : register(u0);    // reshaped gradient (outC, B*OH*OW)

[numthreads(16, 16, 1)]
void CSMain(uint3 gid : SV_DispatchThreadID) {
    uint oc = gid.y;
    uint b_pixel = gid.x;
    uint B_OH_OW = batch * outH * outW;

    if (oc >= outC || b_pixel >= B_OH_OW) return;

    uint b = b_pixel / (outH * outW);
    uint pixel = b_pixel % (outH * outW);

    uint src_idx = b * (outC * outH * outW) + oc * (outH * outW) + pixel;
    float grad = dZ[src_idx];

    // ReLU gradient: zero out where forward output was <= 0
    if (fwd_output[src_idx] <= 0.0f) grad = 0.0f;

    output[oc * B_OH_OW + b_pixel] = grad;
}
