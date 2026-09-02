import onnx_ir as ir

model = ir.load("dailys/day01/onnx/tiny_mlp_ir.onnx")
graph = model.graph

print("IR version:", model.ir_version)
print("Opsets:", graph.opset_imports)
print("Graph:", graph.name)
print("Inputs:", [(value.name, value.dtype, value.shape) for value in graph.inputs])
print("Initializers:", list(graph.initializers))
print("Outputs:", [(value.name, value.dtype, value.shape) for value in graph.outputs])

for node in graph:
    print(
        node.name,
        node.domain,
        node.op_type,
        [value.name for value in node.inputs],
        [value.name for value in node.outputs],
    )