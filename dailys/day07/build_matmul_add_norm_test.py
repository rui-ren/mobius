from __future__ import annotations

import onnx
import onnx_ir as ir
import pytest

from dailys.day07.build_matmul_add_norm import EPSILON, build_model, save_model


def _shape(value: ir.Value) -> list[str]:
    return [str(dimension) for dimension in value.shape]


def test_saved_graph_has_expected_static_contract(tmp_path):
    model_path = tmp_path / "matmul_add_norm.onnx"
    save_model(build_model(), model_path)

    onnx.checker.check_model(model_path.as_posix())
    graph = ir.load(model_path).graph
    nodes = list(graph)

    assert graph.opset_imports == {"": 24}
    assert [node.op_type for node in nodes] == ["MatMul", "Add", "LayerNormalization"]
    assert [value.name for value in graph.inputs] == ["x"]
    assert list(graph.initializers) == [
        "weight",
        "projection_bias",
        "scale",
        "normalization_bias",
    ]
    assert [value.name for value in graph.outputs] == ["y"]

    values = {value.name: value for value in graph.inputs}
    values.update(graph.initializers)
    values.update({value.name: value for node in nodes for value in node.outputs})

    expected_shapes = {
        "x": ["batch", "4"],
        "weight": ["4", "3"],
        "projection_bias": ["3"],
        "scale": ["3"],
        "normalization_bias": ["3"],
        "matmul_output": ["batch", "3"],
        "biased_output": ["batch", "3"],
        "y": ["batch", "3"],
    }
    assert {name: _shape(value) for name, value in values.items()} == expected_shapes
    assert all(value.dtype == ir.DataType.FLOAT for value in values.values())

    attributes = nodes[-1].attributes
    assert attributes["axis"].value == -1
    assert attributes["epsilon"].value == pytest.approx(EPSILON)
