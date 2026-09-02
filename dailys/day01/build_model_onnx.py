from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


output_path = Path(__file__).resolve().parent / "onnx" / "tiny_mlp.onnx"
output_path.parent.mkdir(parents=True, exist_ok=True)



x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3])


W = numpy_helper.from_array(
        np.arange(12, dtype=np.float32).reshape(4, 3) / 12.0,
        name="W"
)

b = numpy_helper.from_array(
        np.zeros(3, dtype=np.float32),
        name="b"
)

nodes = [

        helper.make_node("MatMul",  ["x", "W"], ["xW"], name="matrix_multiply"),
        helper.make_node("Add", ["xW", "b"], ["pre_activation"], name="add_bias"),
        helper.make_node("Relu", ["pre_activation"], ["y"], name="relu")
]


graph = helper.make_graph(
        nodes=nodes,
        name="tiny_mlp",
        inputs=[x],
        outputs=[y],
        initializer=[W, b]
)


model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        producer_name="mobius_onnx_study"
)

onnx.checker.check_model(model)
model = shape_inference.infer_shapes(model)
onnx.checker.check_model(model)

onnx.save(model, output_path)
print(f"Saved onnx model {output_path}")
