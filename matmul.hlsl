cbuffer Constants : register(b0) {
    uint widthA;
    uint heightA;
    uint widthB;
    uint heightB;
};

StructuredBuffer<float> BufferA : register(t0);
StructuredBuffer<float> BufferB : register(t1);
RWStructuredBuffer<float> BufferOut : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint row = dispatchThreadID.y;
    uint col = dispatchThreadID.x;

    if (row < heightA && col < widthB) {
        float sum = 0.0f;
        for (uint i = 0; i < widthA; ++i) {
            sum += BufferA[row * widthA + i] * BufferB[i * widthB + col];
        }
        BufferOut[row * widthB + col] = sum;
    }
}
