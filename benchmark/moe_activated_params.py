"""MoE Activated Parameters vs Sequence Length.

Demonstrates how sparse Mixture-of-Experts models only activate a fraction of
total parameters at small batch/sequence lengths. As seq_len grows, more tokens
activate different expert combinations until saturation.

Usage:
    python benchmark/moe_activated_params.py
"""

from onnx_tool.llm import Builder
from benchmark.llm_test import (
    Qwen3_5_35B_A3B,
    MiniMax_M2_7,
    DeepSeek_V4_Flash,
    DeepSeek_V4_Pro,
)


def get_activated_params(builder, seq_len):
    """Build graph and return (activated_params, total_params) for given seq_len."""
    builder.build_graph([1, seq_len])
    builder.graph.valid_shape = True
    builder.graph.profile()
    activated = sum(node.static_params for node in builder.graph.nodemap.values())
    total = builder.graph.params
    return activated, total


def main():
    models = [
        ("Qwen3.5-35B-A3B (MoE)", Qwen3_5_35B_A3B),
        ("MiniMax-M2.7 (MoE)", MiniMax_M2_7),
        ("DeepSeek-V4-Flash (MoE/MLA)", DeepSeek_V4_Flash),
        ("DeepSeek-V4-Pro (MoE/MLA)", DeepSeek_V4_Pro),
    ]

    seq_lens = [1, 2, 4, 8, 16, 32]

    # Header
    header = ["model", "Total(G)"] + [f"S={s}" for s in seq_lens]
    print(" | ".join(f"{h:>24}" if i == 0 else f"{h:>8}" for i, h in enumerate(header)))
    print("-" * (24 + 9 * (len(seq_lens) + 1)))

    for name, cfg in models:
        row = [name]
        total = None
        for seq_len in seq_lens:
            builder = Builder(**cfg)
            activated, total = get_activated_params(builder, seq_len)
            row.append(f"{activated / 1e9:.2f}")
        # Prepend total
        row.insert(1, f"{total / 1e9:.1f}")
        print(" | ".join(f"{c:>24}" if i == 0 else f"{c:>8}" for i, c in enumerate(row)))

    print()
    print("Notes:")
    print("  - Activated = sum of all nodes' static_params (weights actually accessed).")
    print("  - Total = graph.params (all static weights in the model).")
    print("  - S=32: Qwen3.5-35B-A3B & MiniMax-M2.7 experts fully activated (gap ≈ embedding).")
    print("  - S=32: DeepSeek-V4 not yet saturated (top-6 × 32 = 192 < num_experts).")


if __name__ == "__main__":
    main()
