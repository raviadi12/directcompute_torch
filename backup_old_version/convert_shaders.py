"""Convert HLSL shaders from StructuredBuffer<float> to ByteAddressBuffer.
Handles access pattern conversion:
  READ:  buf[idx]       -> asfloat(buf.Load(idx * 4))  
  WRITE: buf[idx] = val -> buf.Store(idx * 4, asuint(val))
"""
import re, os

shader_dir = r"c:\Users\user\Documents\directx_exp"

# Only convert shaders that are actually compiled
used_shaders = [
    "nn_matmul_universal.hlsl", "nn_matmul_coarsened.hlsl",
    "nn_add_bias.hlsl", "nn_relu.hlsl", "nn_relu_grad.hlsl",
    "nn_softmax.hlsl", "nn_softmax_ce_grad.hlsl",
    "nn_bias_grad.hlsl", "nn_sgd.hlsl", "nn_loss.hlsl",
    "nn_conv_forward_tiled.hlsl", "nn_conv_backprop_filters_tiled.hlsl",
    "nn_conv_backprop_input_fused.hlsl",
    "nn_maxpool_forward.hlsl", "nn_maxpool_backward.hlsl",
    "nn_grad_accum.hlsl", "nn_im2col.hlsl", "nn_conv_reshape.hlsl",
    "nn_conv_grad_reshape.hlsl", "nn_col2im.hlsl",
    "nn_argmax_correct.hlsl", "nn_accumulate_scalar.hlsl",
]

def convert_shader(code):
    """Convert a shader from StructuredBuffer to ByteAddressBuffer."""
    
    # Step 1: Find all buffer variable names and their types
    read_bufs = set()   # ByteAddressBuffer (read-only)
    write_bufs = set()  # RWByteAddressBuffer (read-write)
    uint_bufs = set()   # buffers used with uint (e.g., argmax_correct)
    
    # Find StructuredBuffer<float> NAME : register(tN)
    for m in re.finditer(r'StructuredBuffer<float>\s+(\w+)', code):
        read_bufs.add(m.group(1))
    # Find RWStructuredBuffer<float> NAME : register(uN)
    for m in re.finditer(r'RWStructuredBuffer<float>\s+(\w+)', code):
        write_bufs.add(m.group(1))
    # Find RWStructuredBuffer<uint> NAME : register(uN)
    for m in re.finditer(r'RWStructuredBuffer<uint>\s+(\w+)', code):
        write_bufs.add(m.group(1))
        uint_bufs.add(m.group(1))
    
    if not read_bufs and not write_bufs:
        return code, False
    
    # Step 2: Replace buffer declarations
    code = re.sub(r'StructuredBuffer<float>', 'ByteAddressBuffer', code)
    code = re.sub(r'RWStructuredBuffer<float>', 'RWByteAddressBuffer', code)
    code = re.sub(r'RWStructuredBuffer<uint>', 'RWByteAddressBuffer', code)
    
    # Step 3: Replace write patterns: buf[expr] = val -> buf.Store((expr) * 4, asuint(val))
    for buf in write_bufs:
        if buf in uint_bufs:
            # For uint buffers, don't convert InterlockedAdd patterns
            # Just convert regular writes: buf[expr] = val
            # UINT write: buf[idx] = val -> buf.Store(idx * 4, val) (already uint)
            code = re.sub(
                rf'{buf}\[([^\]]+)\]\s*=\s*([^;]+);',
                lambda m: f'{buf}.Store(({m.group(1)}) * 4, {m.group(2)});',
                code
            )
            # InterlockedAdd(buf[idx], val) -> buf.InterlockedAdd(idx * 4, val, _dummy)
            code = re.sub(
                rf'InterlockedAdd\({buf}\[([^\]]+)\],\s*([^)]+)\)',
                lambda m: f'{buf}.InterlockedAdd(({m.group(1)}) * 4, {m.group(2)}, _dummy)',
                code
            )
        else:
            # Float write: buf[expr] = val -> buf.Store((expr) * 4, asuint(val))
            code = re.sub(
                rf'{buf}\[([^\]]+)\]\s*=\s*([^;]+);',
                lambda m: f'{buf}.Store(({m.group(1)}) * 4, asuint({m.group(2)}));',
                code
            )
    
    # Step 4: Replace read patterns: buf[expr] -> asfloat(buf.Load((expr) * 4))
    for buf in read_bufs:
        code = re.sub(
            rf'{buf}\[([^\]]+)\]',
            lambda m: f'asfloat({buf}.Load(({m.group(1)}) * 4))',
            code
        )
    
    # Step 5: Replace RW read patterns (when RW buffer is read from)
    for buf in write_bufs:
        if buf in uint_bufs:
            continue  # handled above
        # RW float read: buf[expr] -> asfloat(buf.Load((expr) * 4))
        # But only for reads, not writes (writes already converted)
        code = re.sub(
            rf'(?<!\.Store\()(?<!\.Load\(){buf}\[([^\]]+)\]',
            lambda m: f'asfloat({buf}.Load(({m.group(1)}) * 4))',
            code
        )
    
    return code, True

# Process all shaders
for fname in used_shaders:
    path = os.path.join(shader_dir, fname)
    if not os.path.exists(path):
        print(f"SKIP {fname}")
        continue
    
    with open(path, 'r') as f:
        code = f.read()
    
    new_code, changed = convert_shader(code)
    
    if changed:
        with open(path, 'w') as f:
            f.write(new_code)
        print(f"OK {fname}")
    else:
        print(f"NO CHANGE {fname}")

print("\nAll shaders converted. Run bench_matmul.py to test.")
