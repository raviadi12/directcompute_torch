import numpy as np
import torch
import torch.nn as nn
from nn_engine import Tensor, BatchNorm2d

def test_batchnorm():
    np.random.seed(42)
    x_np = np.random.randn(2, 3, 4, 4).astype(np.float32)
    dy_np = np.random.randn(2, 3, 4, 4).astype(np.float32)

    # PyTorch
    x_pt = torch.tensor(x_np, requires_grad=True)
    bn_pt = nn.BatchNorm2d(3, eps=1e-5, momentum=0.1)
    bn_pt.train()
    
    # Initialize weights
    nn.init.uniform_(bn_pt.weight)
    nn.init.uniform_(bn_pt.bias)
    
    y_pt = bn_pt(x_pt)
    y_pt.backward(torch.tensor(dy_np))
    
    # DirectCompute
    x_dc = Tensor(x_np, requires_grad=True)
    bn_dc = BatchNorm2d(3)
    bn_dc.gamma.upload(bn_pt.weight.detach().numpy())
    bn_dc.beta.upload(bn_pt.bias.detach().numpy())
    
    y_dc = bn_dc(x_dc)
    y_dc.backward(Tensor(dy_np))
    
    y_dc_np = y_dc.sync()
    dx_dc = x_dc.grad.sync()
    dg_dc = bn_dc.gamma.grad.sync()
    db_dc = bn_dc.beta.grad.sync()
    
    print("Testing BatchNorm...")
    
    y_diff = np.abs(y_pt.detach().numpy() - y_dc_np).max()
    print(f"y max diff: {y_diff}")
    
    dx_diff = np.abs(x_pt.grad.numpy() - dx_dc).max()
    print(f"dx max diff: {dx_diff}")
    
    dg_diff = np.abs(bn_pt.weight.grad.numpy() - dg_dc).max()
    print(f"dgamma max diff: {dg_diff}")
    
    db_diff = np.abs(bn_pt.bias.grad.numpy() - db_dc).max()
    print(f"dbeta max diff: {db_diff}")
    
    rm_diff = np.abs(bn_pt.running_mean.numpy() - bn_dc.running_mean.sync()).max()
    print(f"running_mean max diff: {rm_diff}")
    
    rv_diff = np.abs(bn_pt.running_var.numpy() - bn_dc.running_var.sync()).max()
    print(f"running_var max diff: {rv_diff}")
    
    assert y_diff < 1e-4
    assert dx_diff < 1e-3
    assert dg_diff < 1e-3
    assert db_diff < 1e-3
    print("PT rm:", bn_pt.running_mean.numpy())
    print("DC rm:", bn_dc.running_mean.sync())
    print("PT rv:", bn_pt.running_var.numpy())
    print("DC rv:", bn_dc.running_var.sync())

if __name__ == '__main__':
    test_batchnorm()
