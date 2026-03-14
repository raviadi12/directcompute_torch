import numpy as np

from nn_engine import CustomShader, CustomUnaryShader, Tensor


# IMPORTANT BINDING NOTE:
# - inputs=[...] are bound as SRVs: StructuredBuffer<T> at t0, t1, ...
# - outputs=[...] are bound as UAVs: RWStructuredBuffer<T> at u0, u1, ...
# So an input tensor should usually be StructuredBuffer (read-only), not RWStructuredBuffer.

SWISH_FWD_HLSL = r'''
StructuredBuffer<float> X : register(t0);
RWStructuredBuffer<float> Y : register(u0);

[numthreads(256, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    float x = X[i];
    float s = 1.0f / (1.0f + exp(-x));
    Y[i] = x * s;
}
'''

# One source with two entry points, to demonstrate non-CSMain support.
SWISH_FWD_BWD_HLSL = r'''
StructuredBuffer<float> X    : register(t0);
StructuredBuffer<float> DOUT : register(t1);
RWStructuredBuffer<float> Y  : register(u0);

[numthreads(256, 1, 1)]
void ForwardMain(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    float x = X[i];
    float s = 1.0f / (1.0f + exp(-x));
    Y[i] = x * s;
}

[numthreads(256, 1, 1)]
void BackwardMain(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    float x = X[i];
    float s = 1.0f / (1.0f + exp(-x));
    // d/dx [x*sigmoid(x)] = s + x*s*(1-s)
    float dswish = s + x * s * (1.0f - s);
    Y[i] = DOUT[i] * dswish;
}
'''

MATMUL_HLSL = r'''
cbuffer MatMulConfig : register(b0) {
    uint M;
    uint K;
    uint N;
    uint pad;
};

StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    uint col = tid.x;
    uint row = tid.y;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint i = 0; i < K; ++i) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}
'''


def demo_swish_forward_only():
    x_np = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    x = Tensor(x_np)
    y = Tensor.empty_like(x)

    swish = CustomShader(SWISH_FWD_HLSL, entry_point="CSMain", debug_name="swish_fwd")
    groups = (x.size + 255) // 256
    swish.dispatch(grid=(groups, 1, 1), inputs=[x], outputs=[y])

    y_np = y.sync()
    ref = x_np * (1.0 / (1.0 + np.exp(-x_np)))
    print("forward y:", np.round(y_np, 6))
    print("forward max abs err:", float(np.max(np.abs(y_np - ref))))


def demo_swish_with_custom_backward():
    swish = CustomUnaryShader(
        forward_hlsl=SWISH_FWD_BWD_HLSL,
        backward_hlsl=SWISH_FWD_BWD_HLSL,
        forward_entry="ForwardMain",
        backward_entry="BackwardMain",
        threads_per_group=256,
        debug_name="swish_pair",
    )

    x_np = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
    x = Tensor(x_np, requires_grad=True)
    y = swish(x)
    y.backward(Tensor(np.ones_like(x_np, dtype=np.float32), track=False))

    grad_np = x.grad.sync()
    s = 1.0 / (1.0 + np.exp(-x_np))
    ref_grad = s + x_np * s * (1.0 - s)
    print("backward dx:", np.round(grad_np, 6))
    print("backward max abs err:", float(np.max(np.abs(grad_np - ref_grad))))


def demo_custom_matmul():
    M, K, N = 4, 8, 3
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)

    a = Tensor(a_np)
    b = Tensor(b_np)
    c = Tensor.empty((M, N))

    mm = CustomShader(MATMUL_HLSL, entry_point="CSMain", debug_name="naive_mm")
    gx = (N + 16 - 1) // 16
    gy = (M + 16 - 1) // 16
    mm.dispatch(
        grid=(gx, gy, 1),
        inputs=[a, b],
        outputs=[c],
        constants=[M, K, N, 0],
    )

    c_np = c.sync()
    ref = a_np @ b_np
    print("matmul max abs err:", float(np.max(np.abs(c_np - ref))))


if __name__ == "__main__":
    demo_swish_forward_only()
    demo_swish_with_custom_backward()
    demo_custom_matmul()
