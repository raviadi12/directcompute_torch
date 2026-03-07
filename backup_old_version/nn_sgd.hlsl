cbuffer Params : register(b0) { uint count; float lr; float clip; uint pad; };
RWStructuredBuffer<float> weight : register(u0);
StructuredBuffer<float> grad : register(t0);
[numthreads(256, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x < count) {
        float g = grad[id.x];
        if (g > clip) g = clip;
        if (g < -clip) g = -clip;
        weight[id.x] = weight[id.x] - lr * g;
    }
}
