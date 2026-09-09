from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir

OPSET_IMPORTS = {"": 24}
EPSILON = 1e-5
OUTPUT_PATH = Path(__file__).resolve().parent / "onnx" / "matmul_add_norm.onnx"

TENSOR_ROLES = {
    "x": "Runtime input",
    "weight": "Projection weight",
    "projection_bias": "Projection bias",
    "scale": "Normalization scale",
    "normalization_bias": "Normalization bias",
    "matmul_output": "Projected values",
    "biased_output": "Bias-adjusted values",
    "y": "Normalized output",
}


def _float_value(name: str, shape: list[int | ir.SymbolicDim]) -> ir.Value:
    return ir.Value(
        name=name,
        shape=ir.Shape(shape),
        type=ir.TensorType(ir.DataType.FLOAT),
    )


def _float_initializer(name: str, value: np.ndarray) -> ir.Value:
    return ir.Value(
        name=name,
        shape=ir.Shape(list(value.shape)),
        type=ir.TensorType(ir.DataType.FLOAT),
        const_value=ir.tensor(value),
    )


def build_model() -> ir.Model:
    """Build the MatMul -> Add -> LayerNormalization graph with static metadata."""
    batch = ir.SymbolicDim("batch")

    x = _float_value("x", [batch, 4])
    weight = _float_initializer(
        "weight",
        np.arange(12, dtype=np.float32).reshape(4, 3) / 12.0,
    )
    projection_bias = _float_initializer(
        "projection_bias",
        np.zeros(3, dtype=np.float32),
    )
    scale = _float_initializer("scale", np.ones(3, dtype=np.float32))
    normalization_bias = _float_initializer(
        "normalization_bias",
        np.zeros(3, dtype=np.float32),
    )

    # Project the hidden width: (batch, 4) @ (4, 3) -> (batch, 3).
    matmul_output = _float_value("matmul_output", [batch, 3])
    matmul = ir.Node(
        "",
        "MatMul",
        inputs=[x, weight],
        outputs=[matmul_output],
        name="project_hidden_width",
    )

    # Broadcast the rank-1 bias across the symbolic batch dimension.
    biased_output = _float_value("biased_output", [batch, 3])
    add = ir.Node(
        "",
        "Add",
        inputs=[matmul_output, projection_bias],
        outputs=[biased_output],
        name="add_projection_bias",
    )

    # Normalize the three-element final axis: (batch, 3) -> (batch, 3).
    y = _float_value("y", [batch, 3])
    layer_normalization = ir.Node(
        "",
        "LayerNormalization",
        inputs=[biased_output, scale, normalization_bias],
        outputs=[y],
        attributes=[
            ir.AttrInt64("axis", -1),
            ir.AttrFloat32("epsilon", EPSILON),
        ],
        name="normalize_projection",
    )

    graph = ir.Graph(
        inputs=[x],
        outputs=[y],
        nodes=[matmul, add, layer_normalization],
        initializers=[weight, projection_bias, scale, normalization_bias],
        name="matmul_add_norm",
        opset_imports=OPSET_IMPORTS,
    )
    return ir.Model(graph=graph, ir_version=11, producer_name="mobius-onnx-study")


def save_model(model: ir.Model, output_path: Path = OUTPUT_PATH) -> None:
    """Save the model and validate the serialized ONNX file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ir.save(model, output_path)
    onnx.checker.check_model(output_path.as_posix())
    print(f"Model saved to {output_path}")
    print("ONNX checker: PASS")


def _format_shape(value: ir.Value) -> str:
    return f"[{', '.join(str(dimension) for dimension in value.shape)}]"


def inspect_model(model_path: Path = OUTPUT_PATH) -> None:
    """Print the saved graph structure and a complete tensor table."""
    model = ir.load(model_path)
    graph = model.graph
    nodes = list(graph)

    print(f"IR version: {model.ir_version}")
    print(f"Opsets: {graph.opset_imports}")
    print(f"Graph: {graph.name}")
    print(f"Inputs: {[value.name for value in graph.inputs]}")
    print(f"Initializers: {list(graph.initializers)}")
    print(f"Outputs: {[value.name for value in graph.outputs]}")
    print("Nodes:")
    for index, node in enumerate(nodes, start=1):
        attributes = ", ".join(
            f"{name}={attribute.value}" for name, attribute in node.attributes.items()
        )
        suffix = f"; attributes: {attributes}" if attributes else ""
        print(
            f"  {index}. {node.op_type}: "
            f"{[value.name for value in node.inputs]} -> "
            f"{[value.name for value in node.outputs]}{suffix}"
        )

    values = {value.name: value for value in graph.inputs}
    values.update(graph.initializers)
    producers = {value.name: "Graph input" for value in graph.inputs}
    producers.update(dict.fromkeys(graph.initializers, "Initializer"))
    consumers: dict[str, list[str]] = {name: [] for name in values}

    for node in nodes:
        for value in node.inputs:
            consumers.setdefault(value.name, []).append(node.op_type)
        for value in node.outputs:
            values[value.name] = value
            producers[value.name] = node.op_type
            consumers.setdefault(value.name, [])

    for value in graph.outputs:
        consumers[value.name].append("Graph output")

    print("Tensor table:")
    print("| Tensor | Role | Shape | Dtype | Producer | Consumer |")
    print("|---|---|---|---|---|---|")
    for name, role in TENSOR_ROLES.items():
        value = values[name]
        print(
            f"| {name} | {role} | {_format_shape(value)} | {value.dtype.name} | "
            f"{producers[name]} | {', '.join(consumers[name])} |"
        )

    print(
        "Bias broadcasting: projection_bias has shape [3], so Add broadcasts it "
        "across every row of the [batch, 3] MatMul output."
    )
    print(
        "axis=-1 is valid because it selects the last axis of [batch, 3], "
        "whose width matches scale and normalization_bias."
    )


def main() -> None:
    model = build_model()
    save_model(model)
    inspect_model()


if __name__ == "__main__":
    main()
