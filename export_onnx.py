"""Export trained DirectCompute models to ONNX format."""
import numpy as np

def export_lenet_onnx(c1, c2, l1, l2, l3, path="lenet.onnx"):
    """Export a trained LeNet model to ONNX.

    Args:
        c1, c2: ConvLayer instances
        l1, l2, l3: Linear instances
        path: output .onnx file path
    """
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    # Sync weights from GPU → CPU
    c1_w = c1.filters.sync()           # (6, 1, 5, 5)
    c1_b = c1.bias.sync()              # (6,)
    c2_w = c2.filters.sync()           # (16, 6, 5, 5)
    c2_b = c2.bias.sync()              # (16,)
    l1_w = l1.w.sync().T               # (in, out) → (out, in) for ONNX Gemm
    l1_b = l1.b.sync()                 # (120,)
    l2_w = l2.w.sync().T               # (120, 84) → (84, 120)
    l2_b = l2.b.sync()                 # (84,)
    l3_w = l3.w.sync().T               # (84, 10) → (10, 84)
    l3_b = l3.b.sync()                 # (10,)

    # Initializers (weight tensors embedded in the model)
    initializers = [
        numpy_helper.from_array(c1_w, "c1_w"),
        numpy_helper.from_array(c1_b, "c1_b"),
        numpy_helper.from_array(c2_w, "c2_w"),
        numpy_helper.from_array(c2_b, "c2_b"),
        numpy_helper.from_array(l1_w, "l1_w"),
        numpy_helper.from_array(l1_b, "l1_b"),
        numpy_helper.from_array(l2_w, "l2_w"),
        numpy_helper.from_array(l2_b, "l2_b"),
        numpy_helper.from_array(l3_w, "l3_w"),
        numpy_helper.from_array(l3_b, "l3_b"),
    ]

    # Build the graph: Conv→Relu→MaxPool→Conv→Relu→MaxPool→Flatten→FC→Relu→FC→Relu→FC
    nodes = [
        helper.make_node("Conv", ["input", "c1_w", "c1_b"], ["conv1"], kernel_shape=[5, 5]),
        helper.make_node("Relu", ["conv1"], ["relu1"]),
        helper.make_node("MaxPool", ["relu1"], ["pool1"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Conv", ["pool1", "c2_w", "c2_b"], ["conv2"], kernel_shape=[5, 5]),
        helper.make_node("Relu", ["conv2"], ["relu2"]),
        helper.make_node("MaxPool", ["relu2"], ["pool2"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Flatten", ["pool2"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "l1_w", "l1_b"], ["fc1"], transB=1),
        helper.make_node("Relu", ["fc1"], ["relu3"]),
        helper.make_node("Gemm", ["relu3", "l2_w", "l2_b"], ["fc2"], transB=1),
        helper.make_node("Relu", ["fc2"], ["relu4"]),
        helper.make_node("Gemm", ["relu4", "l3_w", "l3_b"], ["output"], transB=1),
    ]

    graph = helper.make_graph(
        nodes,
        "LeNet",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 1, 28, 28])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 10])],
        initializer=initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"Model exported to {path} ({_file_size_str(path)})")


def export_alexnet_onnx(c1, c2, c3, c4, c5, l1, l2, l3, path="alexnet.onnx"):
    """Export a trained AlexNet model to ONNX.

    Args:
        c1-c5: ConvLayer instances
        l1, l2, l3: Linear instances
        path: output .onnx file path
    """
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    # Sync all weights from GPU → CPU and transpose linears for ONNX Gemm
    def conv_params(layer, name):
        w = layer.filters.sync()
        b = layer.bias.sync()
        return [
            numpy_helper.from_array(w, f"{name}_w"),
            numpy_helper.from_array(b, f"{name}_b"),
        ]

    def linear_params(layer, name):
        w = layer.w.sync().T  # (in, out) → (out, in) for Gemm transB=1
        b = layer.b.sync()
        return [
            numpy_helper.from_array(w, f"{name}_w"),
            numpy_helper.from_array(b, f"{name}_b"),
        ]

    initializers = (
        conv_params(c1, "c1") + conv_params(c2, "c2") + conv_params(c3, "c3") +
        conv_params(c4, "c4") + conv_params(c5, "c5") +
        linear_params(l1, "l1") + linear_params(l2, "l2") + linear_params(l3, "l3")
    )

    # Build graph: matches train_alexnet.py forward pass
    nodes = [
        # Conv block 1: conv(11×11, stride=4, pad=2) → relu → maxpool(3×3, stride=2)
        helper.make_node("Conv", ["input", "c1_w", "c1_b"], ["conv1"],
                         kernel_shape=[11, 11], strides=[4, 4], pads=[2, 2, 2, 2]),
        helper.make_node("Relu", ["conv1"], ["relu1"]),
        helper.make_node("MaxPool", ["relu1"], ["pool1"],
                         kernel_shape=[3, 3], strides=[2, 2]),

        # Conv block 2: conv(5×5, pad=2) → relu → maxpool(3×3, stride=2)
        helper.make_node("Conv", ["pool1", "c2_w", "c2_b"], ["conv2"],
                         kernel_shape=[5, 5], pads=[2, 2, 2, 2]),
        helper.make_node("Relu", ["conv2"], ["relu2"]),
        helper.make_node("MaxPool", ["relu2"], ["pool2"],
                         kernel_shape=[3, 3], strides=[2, 2]),

        # Conv block 3: conv(3×3, pad=1) → relu
        helper.make_node("Conv", ["pool2", "c3_w", "c3_b"], ["conv3"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv3"], ["relu3"]),

        # Conv block 4: conv(3×3, pad=1) → relu
        helper.make_node("Conv", ["relu3", "c4_w", "c4_b"], ["conv4"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv4"], ["relu4"]),

        # Conv block 5: conv(3×3, pad=1) → relu → maxpool(3×3, stride=2)
        helper.make_node("Conv", ["relu4", "c5_w", "c5_b"], ["conv5"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv5"], ["relu5"]),
        helper.make_node("MaxPool", ["relu5"], ["pool5"],
                         kernel_shape=[3, 3], strides=[2, 2]),

        # Classifier
        helper.make_node("Flatten", ["pool5"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "l1_w", "l1_b"], ["fc1"], transB=1),
        helper.make_node("Relu", ["fc1"], ["relu6"]),
        helper.make_node("Gemm", ["relu6", "l2_w", "l2_b"], ["fc2"], transB=1),
        helper.make_node("Relu", ["fc2"], ["relu7"]),
        helper.make_node("Gemm", ["relu7", "l3_w", "l3_b"], ["output"], transB=1),
    ]

    graph = helper.make_graph(
        nodes,
        "AlexNet",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3, 224, 224])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])],
        initializer=initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"Model exported to {path} ({_file_size_str(path)})")


def _file_size_str(path):
    import os
    size = os.path.getsize(path)
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / (1024 * 1024):.1f} MB"
