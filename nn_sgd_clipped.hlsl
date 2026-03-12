// SGD with clip scale applied from a GPU buffer.
// Reads scale from normScale[1], applies: weight -= lr * grad * scale
// Dispatch: ((count+255)/256, 1, 1) per parameter.

RWStructuredBuffer<float> weight : register(u0);
StructuredBuffer<float> grad : register(t0);
StructuredBuffer<float> normScale : register(t1);  // [0]=norm, [1]=clip_scale

cbuffer Params : register(b0) {
    uint count;
    uint lrBits;  // lr as uint bits (reinterpret as float)
};

[numthreads(256, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x < count) {
        float lr = asfloat(lrBits);
        float scale = normScale[1];
        weight[id.x] -= lr * grad[id.x] * scale;
    }
}
