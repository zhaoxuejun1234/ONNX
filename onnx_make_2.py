import onnx
import onnx.helper as helper
from onnx import TensorProto
import numpy as np
def make_initializer_value_info(name,dims) -> TensorProto:
    value = np.random.random(dims).astype(np.float32)
    #Raw参数确定是用二进制方法存储参数
    tensor = helper.make_tensor(name =name,data_type = TensorProto.FLOAT,dims=list(value.shape),vals=value.tobytes(),raw=True)
    return tensor
# def make_initializer_value_info(name,dims):
#     value = np.random.random(dims).astype(np.float32)
#     tensor = helper.make_tensor(
#         name = name,
#         data_type = helper.TensorProto.DataType.FLOAT,
#         dims = list(value.shape),
#         vals = value.tobytes(),
#         raw = True   
#     )
#     return tensor
input = helper.make_tensor_value_info("Input",TensorProto.FLOAT,[1,128,56,56])

w = make_initializer_value_info("WW",[64,128,1,1])

output = helper.make_tensor_value_info("Output",TensorProto.FLOAT,[1,64,56,56])



Conv1 = helper.make_node("Conv",["Input","WW"],["Output"])



graph = helper.make_graph([Conv1],"Test",[input],[output],[w])
model  = helper.make_model(graph)
onnx.save(model,"my_onnx.onnx")


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


