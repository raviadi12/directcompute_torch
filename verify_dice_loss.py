import numpy as np
import nn_engine as nn

def check_close(a, b, name, rtol=1e-3, atol=1e-4):
    diff = np.abs(a - b)
    max_diff = np.max(diff)
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        print(f"FAILED: {name} | Max diff: {max_diff}")
    else:
        print(f"PASSED: {name} | Max diff: {max_diff}")

# Dice Loss Test
np.random.seed(42)
pred_np = np.random.rand(2, 3, 4, 4).astype(np.float32)
target_np = np.random.randint(0, 2, (2, 3, 4, 4)).astype(np.float32)

def np_dice_loss(p, t):
    batch = p.shape[0]
    p_flat = p.reshape(batch, -1)
    t_flat = t.reshape(batch, -1)
    intersection = np.sum(p_flat * t_flat, axis=1)
    sum_p = np.sum(p_flat, axis=1)
    sum_t = np.sum(t_flat, axis=1)
    dice = (2. * intersection) / (sum_p + sum_t + 1e-6)
    return 1. - dice

def np_dice_grad(p, t):
    batch = p.shape[0]
    p_flat = p.reshape(batch, -1)
    t_flat = t.reshape(batch, -1)
    intersection = np.sum(p_flat * t_flat, axis=1, keepdims=True)
    sum_p = np.sum(p_flat, axis=1, keepdims=True)
    sum_t = np.sum(t_flat, axis=1, keepdims=True)
    denom = sum_p + sum_t + 1e-6
    
    # dL/dp_i = -2 * [ (t_i * denom) - (intersection * 1) ] / (denom^2)
    # Then scaled by 1/batch
    grad = -2.0 * (t_flat * denom - intersection) / (denom ** 2)
    grad = grad / batch
    return grad.reshape(p.shape)

loss_np = np_dice_loss(pred_np, target_np)
dpred_np = np_dice_grad(pred_np, target_np)

pred = nn.Tensor(pred_np, requires_grad=True)
target = nn.Tensor(target_np, requires_grad=False)

loss = nn.dice_loss(pred, target)
loss_out = loss.sync()
check_close(loss_out, loss_np, "Dice Loss Forward")

loss.backward(nn.Tensor(np.ones_like(loss_out)))
dpred = pred.grad.sync()
check_close(dpred, dpred_np, "Dice Loss Backward")

nn.release_all_buffers()
