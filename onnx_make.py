# import onnx
# import onnx.helper as helper
# from onnx import TensorProto
# import numpy as np

# def make_initializer_tensor(name,dims):
#     value = np.random.random(dims).astype(np.float32)
#     tensor = helper.make_tensor(
#         name = name,
#         data_type = helper.TensorProto.DataType.FLOAT,
#         dims = list(value.shape),
#         vals = value.tobytes(),
#         raw = True   
#     )
#     return tensor


# input = helper.make_tensor_value_info(
#     "conv1_input",TensorProto.FLOAT,[1,128,56,56]
# )
# w1 = make_initializer_tensor("conv1_w",[64,128,1,1])
# conv1_output = helper.make_tensor_value_info(
#     "conv1_output",TensorProto.FLOAT,[1,64,56,56]
# )
# conv1 = helper.make_node(
#     "conv",
#     inputs = ["conv1_input","conv1_w"],
#     outputs = ["conv1_output"],
# )
# graph = helper.make_graph(
#     nodes=[conv1],
#     name = "test",
#     inputs=[input],
#     outputs = [conv1_output],
#     initializer = [w1],
#     value_info = [conv1_output]
# )
# model = helper.make_model(graph)
# # onnx.checker.check_model(model)
# onnx.save(model,"demo.onnx")

import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto


def make_initializer_tensor(name, dims) -> TensorProto:
    value = np.random.random(dims).astype(np.float32)
    tensor = helper.make_tensor(
        name=name,
        data_type=TensorProto.DataType.FLOAT,
        dims=list(value.shape),
        vals=value.tobytes(),
        raw=True
    )
    return tensor


input = helper.make_tensor_value_info(
    'conv1_input', TensorProto.FLOAT, [1, 128, 56, 56])

# ----------------- Convolution 1x1 -----------------
w1 = make_initializer_tensor("conv1_w", [64, 128, 1, 1])
conv1_output = helper.make_tensor_value_info(
    'conv1_output', TensorProto.FLOAT, [1, 64, 56, 56])
conv1 = helper.make_node(
    op_type="Conv",
    inputs=["conv1_input", "conv1_w"],
    outputs=["conv1_output"],
    kernel_shape=[1, 1],
    strides=[1, 1],
    dilations=[1, 1],
    group=1,
    pads=[0, 0, 0, 0],
)

relu1_output = helper.make_tensor_value_info(
    'relu1_output', TensorProto.FLOAT, [1, 64, 56, 56])
relu1 = helper.make_node(
    "Relu", inputs=["conv1_output"], outputs=["relu1_output"])

# ----------------- Convolution 3x3 -----------------
w2 = make_initializer_tensor("conv2_w", [64, 64, 3, 3])
conv2_output = helper.make_tensor_value_info(
    'conv2_output', TensorProto.FLOAT, [1, 64, 56, 56])
conv2 = helper.make_node(
    "Conv",
    inputs=["relu1_output", "conv2_w"],
    outputs=["conv2_output"],
    kernel_shape=[3, 3],
    strides=[1, 1],
    dilations=[1, 1],
    group=1,
    pads=[1, 1, 1, 1],
)

relu2_output = helper.make_tensor_value_info(
    'relu2_output', TensorProto.FLOAT, [1, 64, 56, 56])
relu2 = helper.make_node(
    "Relu", inputs=["conv2_output"], outputs=["relu2_output"])

#  ----------------- Convolution 1x1 -----------------
w3 = make_initializer_tensor("conv3_w", [128, 64, 1, 1])
conv3_output = helper.make_tensor_value_info(
    'conv3_output', TensorProto.FLOAT, [1, 128, 56, 56])
conv3 = helper.make_node(
    "Conv",
    inputs=["relu2_output", "conv3_w"],
    outputs=["conv3_output"],
    kernel_shape=[1, 1],
    strides=[1, 1],
    dilations=[1, 1],
    group=1,
    pads=[0, 0, 0, 0],
)

add_output = helper.make_tensor_value_info(
    'add_output', TensorProto.FLOAT, [1, 128, 56, 56])
add = helper.make_node(
    "Add", inputs=["conv3_output", "conv1_input"], outputs=["add_output"])

# graph and model
graph = helper.make_graph(
    nodes=[conv1, relu1, conv2, relu2, conv3, add],
    name="residual_block",
    inputs=[input],
    outputs=[add_output],
    initializer=[w1, w2, w3],
    value_info=[conv1_output, relu1_output, conv2_output,
                relu2_output, conv3_output, add_output]
)
model = helper.make_model(graph)

# save model
onnx.checker.check_model(model)
onnx.save(model, "bottleneck_residual_block.onnx")