from onnx_tool.llm import *


def build_onnx_models():
    bs = 1
    seq_len = 128
    ids_shape = [bs, seq_len]
    builder = Builder(**gptj_6b)
    builder.build_graph(ids_shape)
    builder.save_graph('gptj_6b.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    builder = Builder(**QWen_7B)
    builder.build_graph(ids_shape)
    builder.save_graph('QWen_7B.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    builder = Builder(**Qwen2_72B_Instruct)
    builder.build_graph(ids_shape)
    builder.save_graph('Qwen2_72B_Instruct.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    builder = Builder(**Llama3_8B)
    builder.build_graph(ids_shape)
    builder.save_graph('Llama3_8B.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    builder = Builder(**llama_31_70B)
    builder.build_graph(ids_shape)
    builder.save_graph('llama_31_70B.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    builder = Builder(**phi3_mini)
    builder.build_graph(ids_shape)
    builder.save_graph('phi3_mini.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    builder = Builder(**Phi_3_medium_4k_instruct)
    builder.build_graph(ids_shape)
    builder.save_graph('Phi_3_medium_4k_instruct.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    builder = Builder(**Phi_3_small_8k_instruct)
    builder.build_graph(ids_shape)
    builder.save_graph('Phi_3_small_8k_instruct.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    builder = Builder(**phi2)
    builder.build_graph(ids_shape)
    builder.save_graph('phi2.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    builder = Builder(**yi_34B)
    builder.build_graph(ids_shape)
    builder.save_graph('yi_34B.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()


def profile_models():
    import tabulate
    bs = 1
    seq_len = 4096
    ids_shape = [bs, seq_len]
    models = [gptj_6b, yi_34B, phi2, phi3_mini, Phi_3_small_8k_instruct, Phi_3_medium_4k_instruct, Llama3_8B,
              llama_31_70B, QWen_7B, Qwen2_72B_Instruct]
    header = ['model_type', 'MACs', 'Parameters', 'KV Cache']
    rows = []
    for model in models:
        builder = Builder(**model)
        builder.build_graph(ids_shape)
        builder.graph.valid_shape = True
        builder.graph.profile()
        rows.append([builder.name, int(builder.graph.macs[0]), builder.graph.params, builder.kv_params])
    print(tabulate.tabulate(rows, headers=header))


if __name__ == '__main__':
    profile_models()
