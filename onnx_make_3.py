import onnx
import onnx.helper as helper
from onnx import TensorProto
import numpy as np
def make_initializer_info(name,dims):
    value = np.random.random(dims).astype(np.float32)
    tensor = helper.make_tensor(name,TensorProto.DataType.FLOAT,list(value.shape),value.tobytes(),True)
    return tensor
Conv1_Input = helper.make_tensor_value_info("Conv1_Input",TensorProto.DataType.FLOAT,[1, 128, 56, 56])
Conv1_output = helper.make_tensor_value_info("Conv1_Output",TensorProto.DataType.FLOAT,[1, 64, 56, 56])
w1 = make_initializer_info("w1",[64, 128, 1, 1])
conv1 = helper.make_node("Conv",["Conv1_Input","w1"],["Conv1_Output"],"conv1")


Relu_output = helper.make_tensor_value_info("Output",TensorProto.DataType.FLOAT,[1, 64, 56, 56])

relu1 = helper.make_node("Relu",["Conv1_Output"],["Output"],"relu_1")

graph = helper.make_graph(
    nodes=[conv1,relu1],
    name = "onnx_1",
    inputs=[Conv1_Input],
    outputs=[Relu_output],
    initializer=[w1]
)
model = helper.make_model(graph)
onnx.save(model,"my_onnx_2.onnx")