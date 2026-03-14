import numpy as np

from nn_engine import CustomLayerOp, Tensor, grid_1d


# Transformer-style residual add block as a fully custom layer.
# Binding reminder:
# - inputs=[x, residual] -> t0, t1 as StructuredBuffer<float>
# - outputs=[y] -> u0 as RWStructuredBuffer<float>
RESIDUAL_HLSL = r'''
cbuffer Cfg : register(b0) {
    uint N;
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<float> X : register(t0);
StructuredBuffer<float> R : register(t1);
StructuredBuffer<float> DOUT : register(t2);
RWStructuredBuffer<float> Y : register(u0);

[numthreads(256, 1, 1)]
void ResidualForward(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= N) return;
    Y[i] = X[i] + R[i];
}

[numthreads(256, 1, 1)]
void ResidualBackwardX(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= N) return;
    Y[i] = DOUT[i];
}

[numthreads(256, 1, 1)]
void ResidualBackwardR(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= N) return;
    Y[i] = DOUT[i];
}
'''


class ResidualAddLayer:
    """Reusable custom layer suitable for Transformer residual paths."""

    def __init__(self):
        self.op = CustomLayerOp(
            forward_hlsl=RESIDUAL_HLSL,
            backward_hlsl_list=[RESIDUAL_HLSL, RESIDUAL_HLSL],
            forward_entry="ResidualForward",
            backward_entry_list=["ResidualBackwardX", "ResidualBackwardR"],
            debug_name="transformer_residual",
        )

    def __call__(self, x, residual):
        if x.shape != residual.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {residual.shape}")
        n = x.size
        const = [n, 0, 0, 0]
        return self.op(
            inputs=[x, residual],
            output_shape=x.shape,
            forward_grid=grid_1d(n, 256),
            forward_constants=const,
            backward_grid=grid_1d(n, 256),
            backward_constants=[const, const],
        )


def demo_residual_add_layer():
    batch, seq, hidden = 2, 4, 8
    x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
    r_np = np.random.randn(batch, seq, hidden).astype(np.float32)

    x = Tensor(x_np, requires_grad=True)
    r = Tensor(r_np, requires_grad=True)

    layer = ResidualAddLayer()
    y = layer(x, r)
    y_np = y.sync()

    ref = x_np + r_np
    max_err = float(np.max(np.abs(y_np - ref)))
    print("residual forward max abs err:", max_err)

    y.backward(Tensor(np.ones_like(y_np, dtype=np.float32), track=False))
    dx = x.grad.sync()
    dr = r.grad.sync()
    print("residual backward dX err:", float(np.max(np.abs(dx - 1.0))))
    print("residual backward dR err:", float(np.max(np.abs(dr - 1.0))))


if __name__ == "__main__":
    demo_residual_add_layer()
