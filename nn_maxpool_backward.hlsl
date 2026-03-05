cbuffer PoolParams : register(b0) {
    uint batch;
    uint inC;
    uint outH;
    uint outW;
};

StructuredBuffer<float> dZ : register(t0);
StructuredBuffer<float> indices_f : register(t1);  // uint indices stored as float bits
RWStructuredBuffer<float> dInput : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 gid : SV_DispatchThreadID) {
    uint ow = gid.x;
    uint oh = gid.y;
    uint bc = gid.z;

    if (ow >= outW || oh >= outH || bc >= batch * inC) return;

    uint out_idx = bc * (outH * outW) + oh * outW + ow;
    uint maxIdx = asuint(indices_f[out_idx]);  // read float bits as uint index
    if (maxIdx != 0xFFFFFFFF) {
        dInput[maxIdx] = dZ[out_idx];
    }
}
