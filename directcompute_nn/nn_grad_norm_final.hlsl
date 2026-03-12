// Final reduction of all partial squared-sums into total grad norm + clip scale.
// Dispatch: (1, 1, 1) — single thread group of 256 reduces all partials.
// Output: normScale[0] = total_norm, normScale[1] = clip_scale

StructuredBuffer<float> partials : register(t0);
RWStructuredBuffer<float> normScale : register(u0);

cbuffer Params : register(b0) {
    uint totalPartials;  // total number of partial sums across all parameters
    uint maxNormBits;    // max_norm as uint bits (reinterpret as float)
};

groupshared float sdata[256];

[numthreads(256, 1, 1)]
void CSMain(uint3 lid : SV_GroupThreadID) {
    float sum = 0.0f;
    for (uint i = lid.x; i < totalPartials; i += 256) {
        sum += partials[i];
    }
    sdata[lid.x] = sum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128; s > 0; s >>= 1) {
        if (lid.x < s) sdata[lid.x] += sdata[lid.x + s];
        GroupMemoryBarrierWithGroupSync();
    }

    if (lid.x == 0) {
        float norm = sqrt(sdata[0]);
        float maxNorm = asfloat(maxNormBits);
        normScale[0] = norm;
        normScale[1] = (norm > maxNorm) ? (maxNorm / (norm + 1e-6f)) : 1.0f;
    }
}
