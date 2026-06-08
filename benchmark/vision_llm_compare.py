"""
Qwen3.5-4B / Qwen3.5-35B-A3B 多模态模型 Vision vs LLM 计算量对比

分析不同输入图像分辨率下：
  1. Vision Encoder (ViT + Projector) 的 MACs 和参数量
  2. LLM 的 MACs（输入 seq_len = vision patches + text tokens）
  3. Vision / LLM 计算量比例
  4. Projector 输出的 token 数（patches）

关键：多模态模型中 LLM 的输入序列 = vision patches + text tokens，
     而非固定长度。因此 LLM MACs 随分辨率变化。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '.')

import onnx_tool.llm as _llm
import tabulate
from llm_test import Qwen3_5_4B, Qwen3_5_35B_A3B

import importlib
importlib.reload(_llm)
Builder = _llm.Builder


def profile_vision_vs_llm(model_config, model_label, resolutions, text_seq_len=1024):
    """对比 vision encoder 和 LLM 的计算量。

    LLM 的输入 seq_len = vision_patches + text_seq_len，
    模拟多模态推理时 vision tokens 和 text tokens 拼接后的实际输入。

    Args:
        model_config: 模型配置字典
        model_label: 模型名称标签
        resolutions: [(label, [C, H, W]), ...] 不同分辨率
        text_seq_len: 文本 token 数（不含 vision patches）

    Returns:
        rows: 表格行数据
    """
    print(f"\n{'='*70}")
    print(f"{model_label} Vision vs LLM MACs Analysis")
    print(f"{'='*70}")

    patch_size = model_config.get('vision_patch_size', 14)

    header = ['Resolution', 'Patches', 'Total Tokens',
              'Vision(G)', 'LLM(G)', 'Vis/LLM(%)', 'Total(G)']
    rows = []

    for label, image_shape in resolutions:
        h, w = image_shape[1], image_shape[2]
        num_patches = (h // patch_size) * (w // patch_size)
        total_tokens = num_patches + text_seq_len

        # ---- Vision Encoder ----
        builder = Builder(**model_config)
        builder.build_vision_graph(image_shape)
        builder.graph.valid_shape = True
        builder.graph.profile()
        vis_macs = int(builder.graph.macs[0] / 1e9)
        vis_params = builder.graph.params / 1e9

        # ---- LLM (seq_len = patches + text) ----
        bs = 1
        ids_shape = [bs, total_tokens]
        builder = Builder(**model_config)
        builder.build_graph(ids_shape)
        builder.graph.valid_shape = True
        builder.graph.profile()
        llm_macs = int(builder.graph.macs[0] / 1e9)
        llm_params = builder.graph.params / 1e9

        ratio = vis_macs / llm_macs * 100
        total_macs = vis_macs + llm_macs

        row = [label, num_patches, total_tokens,
               vis_macs, llm_macs, f'{ratio:.1f}', total_macs]
        rows.append(row)
        print(f"  {label}: patches={num_patches}, total_tokens={total_tokens}, "
              f"Vision={vis_macs}G, LLM={llm_macs}G, "
              f"Vis/LLM={ratio:.1f}%, Total={total_macs}G")

    print(f"\n{tabulate.tabulate(rows, headers=header)}")
    return rows


def profile_both_models():
    """对比 Qwen3.5-4B 和 Qwen3.5-35B-A3B 的 vision vs LLM 计算量"""

    resolutions = [
        ('224×224', [3, 224, 224]),
        ('336×336', [3, 336, 336]),
        ('448×448', [3, 448, 448]),
        ('672×672', [3, 672, 672]),
        ('896×896', [3, 896, 896]),
        ('1344×896', [3, 1344, 896]),
        ('1344×1344', [3, 1344, 1344]),
    ]

    # ---- Qwen3.5-4B ----
    rows_4b = profile_vision_vs_llm(Qwen3_5_4B, 'Qwen3.5-4B-Instruct', resolutions)

    # ---- Qwen3.5-35B-A3B ----
    rows_35b = profile_vision_vs_llm(Qwen3_5_35B_A3B, 'Qwen3.5-35B-A3B-Instruct', resolutions)

    # ---- 汇总对比表 ----
    print(f"\n{'='*70}")
    print("Cross-Model Comparison: Vision/LLM Ratio")
    print(f"{'='*70}")

    header = ['Resolution', 'Patches', 'Tokens',
              '4B Vis(G)', '4B LLM(G)', '4B Vis/LLM(%)',
              '35B Vis(G)', '35B LLM(G)', '35B Vis/LLM(%)']
    cross_rows = []
    for i, (label, _) in enumerate(resolutions):
        r4 = rows_4b[i]
        r35 = rows_35b[i]
        cross_rows.append([
            label, r4[1], r4[2],
            r4[3], r4[4], r4[5],
            r35[3], r35[4], r35[5],
        ])
    print(tabulate.tabulate(cross_rows, headers=header))

    # ---- 关键发现 ----
    print(f"\n--- Key Insights ---")
    r4_std = rows_4b[2]   # 448×448
    r35_std = rows_35b[2]
    print(f"  Standard (448×448, 1K text):")
    print(f"    Qwen3.5-4B:  Vision={r4_std[3]}G, LLM={r4_std[4]}G "
          f"(tokens={r4_std[2]}), Ratio={r4_std[5]}%")
    print(f"    Qwen3.5-35B: Vision={r35_std[3]}G, LLM={r35_std[4]}G "
          f"(tokens={r35_std[2]}), Ratio={r35_std[5]}%")

    r4_max = rows_4b[-1]
    r35_max = rows_35b[-1]
    print(f"  Max ({resolutions[-1][0]}, 1K text):")
    print(f"    Qwen3.5-4B:  Vision={r4_max[3]}G, LLM={r4_max[4]}G "
          f"(tokens={r4_max[2]}), Ratio={r4_max[5]}%")
    print(f"    Qwen3.5-35B: Vision={r35_max[3]}G, LLM={r35_max[4]}G "
          f"(tokens={r35_max[2]}), Ratio={r35_max[5]}%")


if __name__ == '__main__':
    profile_both_models()
