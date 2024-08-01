from onnx_tool.llm import Builder, QWen_7B, Llama3_8B

if __name__ == '__main__':
    builder = Builder(**QWen_7B)
    builder.build_graph([1,128])
    builder.save_graph('QWen_7B.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    builder = Builder(**Llama3_8B)
    builder.build_graph([1,128])
    builder.save_graph('Llama3_8B.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()
