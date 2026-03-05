cbuffer BlurParams : register(b0) {
    int radius;
    float sigma;
    float dirX;
    float dirY;
    uint width;
    uint height;
    float2 pad;
};

Texture2D<float4> InputTex : register(t0);
RWTexture2D<float4> OutputTex : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;

    float4 color = float4(0, 0, 0, 0);
    float weightSum = 0.0f;
    float2 dir = float2(dirX, dirY);

    for (int i = -radius; i <= radius; ++i) {
        int2 offset = (int2)(dir * i);
        int2 samplePos = clamp((int2)tid.xy + offset, int2(0, 0), int2(width - 1, height - 1));
        
        float weight = exp(-0.5f * (float)(i * i) / (sigma * sigma));
        color += InputTex[samplePos] * weight;
        weightSum += weight;
    }

    // Preserve alpha (though not strictly necessary for webcam)
    color.a = weightSum; 
    OutputTex[tid.xy] = color / weightSum;
}
