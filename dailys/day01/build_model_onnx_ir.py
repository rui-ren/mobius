
from pathlib import Path

import numpy as np
import onnx_ir as ir


output_path = Path(__file__).resolve().parent / "onnx" / "tiny_mlp_ir.onnx"
output_path.parent.mkdir(parents=True, exist_ok=True)


OPSET_VERSION = {"": 24}

# Runtime input: the application supplies x for every inference request.
x = ir.Value(
    name="x",
    shape=ir.Shape([ir.SymbolicDim("batch"), 4]),
    type=ir.TensorType(ir.DataType.FLOAT),
)

xW = ir.Value(
    name="xW",
    shape=ir.Shape([ir.SymbolicDim("batch"), 3]),
    type=ir.TensorType(ir.DataType.FLOAT),
)

# Stored parameters: initializers are part of the model, not runtime inputs.
weight = ir.Value(
    name="W",
    shape=ir.Shape([4, 3]),
    type=ir.TensorType(ir.DataType.FLOAT),
    const_value=ir.tensor(np.arange(12, dtype=np.float32).reshape(4, 3) / 12.0),
)

bias = ir.Value(
    name="b",
    shape=ir.Shape([3]),
    type=ir.TensorType(ir.DataType.FLOAT),
    const_value=ir.tensor(np.zeros(3, dtype=np.float32)),
)

matmul = ir.Node(
    "",
    "MatMul",
    inputs=[x, weight],
    outputs=[xW],
    name="matrix_multiply"
)

pre_activation = ir.Value(
    name="pre_activation",
    shape=ir.Shape([ir.SymbolicDim("batch"), 3]),
    type=ir.TensorType(ir.DataType.FLOAT),
)

add = ir.Node(
    "",
    "Add",
    inputs=[xW, bias],
    outputs=[pre_activation],
    name="add_bias"
)

relu = ir.Node("", "Relu", inputs=[pre_activation], num_outputs=1, name="relu")

y = relu.outputs[0]
y.name = "y"
y.shape = ir.Shape([ir.SymbolicDim("batch"), 3])
y.dtype = ir.DataType.FLOAT

graph = ir.Graph(
    inputs=[x],
    outputs=[y],
    nodes=[matmul, add, relu],
    name="tiny_mlp",
    opset_imports=OPSET_VERSION,
)

graph.register_initializer(weight)
graph.register_initializer(bias)

model = ir.Model(graph=graph, ir_version=11, producer_name="mobius-onnx-study")
ir.save(model, output_path)
print(f"Model saved to {output_path}")
