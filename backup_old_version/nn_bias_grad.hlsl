cbuffer Params : register(b0) { uint batch_size, num_classes, stride_i, stride_id; };
StructuredBuffer<float> grad_out : register(t0);
RWStructuredBuffer<float> grad_bias : register(u0);

groupshared float sdata[256];

[numthreads(256, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint class_id = gid.x;
    if (class_id >= num_classes) return;

    float sum = 0;
    for (uint i = lid.x; i < batch_size; i += 256) {
        sum += grad_out[i * stride_i + class_id * stride_id];
    }
    sdata[lid.x] = sum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint offset = 128; offset > 0; offset /= 2) {
        if (lid.x < offset) {
            sdata[lid.x] += sdata[lid.x + offset];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    
    if (lid.x == 0) {
        grad_bias[class_id] = sdata[0];
    }
}
