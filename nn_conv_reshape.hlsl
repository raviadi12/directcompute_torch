cbuffer Params : register(b0) {
    uint batch;
    uint OC;
    uint OH;
    uint OW;
    uint actType;
};

StructuredBuffer<float> matmul_res : register(t0);
StructuredBuffer<float> bias : register(t1);
RWStructuredBuffer<float> output : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 gid : SV_DispatchThreadID) {
    uint oc = gid.y;
    uint b_pixel = gid.x;
    uint B_OH_OW = batch * OH * OW;
    
    if (oc >= OC || b_pixel >= B_OH_OW) return;
    
    float z = matmul_res[oc * B_OH_OW + b_pixel] + bias[oc];
    float a = z;
    if (actType == 1) a = max(0.0f, z);
    else if (actType == 2) a = tanh(z);
    
    uint b = b_pixel / (OH * OW);
    uint pixel = b_pixel % (OH * OW);
    
    output[b * (OC * OH * OW) + oc * (OH * OW) + pixel] = a;
}
