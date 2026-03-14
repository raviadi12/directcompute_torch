import numpy as np

from nn_engine import (
    CustomLayerOp,
    CustomUnaryShader,
    Tensor,
    grid_1d,
)


SWISH_HLSL = r'''
StructuredBuffer<float> X : register(t0);
StructuredBuffer<float> DOUT : register(t1);
RWStructuredBuffer<float> Y : register(u0);

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
    float dswish = s + x * s * (1.0f - s);
    Y[i] = DOUT[i] * dswish;
}
'''

ADD_HLSL = r'''
cbuffer Cfg : register(b0) {
    uint N;
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
StructuredBuffer<float> DOUT : register(t2);
RWStructuredBuffer<float> Y : register(u0);

[numthreads(256, 1, 1)]
void AddForward(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= N) return;
    Y[i] = A[i] + B[i];
}

[numthreads(256, 1, 1)]
void AddBackwardA(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= N) return;
    Y[i] = DOUT[i];
}

[numthreads(256, 1, 1)]
void AddBackwardB(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= N) return;
    Y[i] = DOUT[i];
}
'''


def assert_close(name, got, ref, atol=1e-5):
    err = float(np.max(np.abs(got - ref)))
    print(f"{name} max abs err: {err}")
    if err > atol:
        raise AssertionError(f"{name} failed: err={err} > atol={atol}")


def test_custom_unary_swish():
    swish = CustomUnaryShader(
        forward_hlsl=SWISH_HLSL,
        backward_hlsl=SWISH_HLSL,
        forward_entry="ForwardMain",
        backward_entry="BackwardMain",
        threads_per_group=256,
        debug_name="test_swish",
    )

    x_np = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    x = Tensor(x_np, requires_grad=True)
    y = swish(x)

    y_np = y.sync()
    s = 1.0 / (1.0 + np.exp(-x_np))
    ref_y = x_np * s
    assert_close("swish forward", y_np, ref_y, atol=1e-6)

    y.backward(Tensor(np.ones_like(x_np, dtype=np.float32), track=False))
    dx = x.grad.sync()
    ref_dx = s + x_np * s * (1.0 - s)
    assert_close("swish backward", dx, ref_dx, atol=1e-6)


def test_custom_layer_add():
    layer = CustomLayerOp(
        forward_hlsl=ADD_HLSL,
        backward_hlsl_list=[ADD_HLSL, ADD_HLSL],
        forward_entry="AddForward",
        backward_entry_list=["AddBackwardA", "AddBackwardB"],
        debug_name="test_add",
    )

    a_np = np.random.randn(32).astype(np.float32)
    b_np = np.random.randn(32).astype(np.float32)
    a = Tensor(a_np, requires_grad=True)
    b = Tensor(b_np, requires_grad=True)

    const = [a.size, 0, 0, 0]
    y = layer(
        inputs=[a, b],
        output_shape=a.shape,
        forward_grid=grid_1d(a.size, 256),
        forward_constants=const,
        backward_grid=grid_1d(a.size, 256),
        backward_constants=[const, const],
    )

    y_np = y.sync()
    assert_close("add forward", y_np, a_np + b_np, atol=1e-6)

    y.backward(Tensor(np.ones_like(a_np, dtype=np.float32), track=False))
    da = a.grad.sync()
    db = b.grad.sync()
    assert_close("add backward dA", da, np.ones_like(a_np), atol=1e-6)
    assert_close("add backward dB", db, np.ones_like(b_np), atol=1e-6)


def main():
    test_custom_unary_swish()
    test_custom_layer_add()
    print("All custom shader tool tests passed.")


if __name__ == "__main__":
    main()
