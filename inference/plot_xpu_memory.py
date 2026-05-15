"""
XPU Memory Curve: PyTorch vs GraphInfer

Records XPU memory at each step during sequential inference:
  point 0: after init (before any inference)
  point 1: after first resolution
  point 2: after second resolution
  ...
Uses PIL for drawing.
"""

import torch
import torchvision
from infer import GraphInfer
from PIL import Image, ImageDraw, ImageFont


def main():
    device = "xpu" if torch.xpu.is_available() else "cpu"
    print(f"Using device: {device}")

    onnx_path = "../resnet18.onnx"

    test_images = [
        ("HD (720p)", (1, 3, 720, 1280)),
        ("Full HD (1080p)", (1, 3, 1080, 1920)),
        ("2K (1440p)", (1, 3, 1440, 2560)),
        ("4K (2160p)", (1, 3, 2160, 3840)),
        ("Square Large", (1, 3, 2048, 2048)),
        ("Mobile (360p)", (1, 3, 360, 640)),
        ("Small", (1, 3, 224, 224)),
    ]
    res_labels = [n for n, _ in test_images]
    n = len(test_images)

    # ===== PyTorch =====
    print("=== PyTorch ===")
    pt_model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )
    pt_model.eval().to(device)

    # point 0: after model init, before any inference
    torch.xpu.synchronize()
    torch.xpu.reset_peak_memory_stats()
    pt_points = [torch.xpu.memory_allocated() / 1024 / 1024]
    print(f"  [init]  XPU={pt_points[0]:.1f} MB")

    # Continuous sequential inference without resetting peak stats
    # Each point captures the cumulative peak up to that step
    for i, (name, shape) in enumerate(test_images):
        x = torch.randn(shape).to(device)
        with torch.no_grad():
            pt_model(x)
        torch.xpu.synchronize()
        peak = torch.xpu.max_memory_allocated() / 1024 / 1024
        pt_points.append(peak)
        print(f"  [{i+1}] {name:<22} XPU peak={peak:.1f} MB")

    # ===== GraphInfer =====
    print("\n=== GraphInfer ===")
    all_h = [s[2] for _, s in test_images]
    all_w = [s[3] for _, s in test_images]

    # point 0: after GraphInfer init
    infer = GraphInfer(
        onnx_path,
        {"input": ("batch", 3, "height", "width")},
        {"batch": (1, 1), "height": (min(all_h), max(all_h)), "width": (min(all_w), max(all_w))},
        dtype=torch.float32,
        device=device,
    )
    torch.xpu.synchronize()
    torch.xpu.reset_peak_memory_stats()
    gi_points = [torch.xpu.memory_allocated() / 1024 / 1024]
    print(f"  [init]  XPU={gi_points[0]:.1f} MB")

    # Continuous sequential inference without resetting peak stats
    for i, (name, shape) in enumerate(test_images):
        x = torch.randn(shape).to(device)
        infer.forward({"input": x}, debug=False)
        torch.xpu.synchronize()
        peak = torch.xpu.max_memory_allocated() / 1024 / 1024
        gi_points.append(peak)
        print(f"  [{i+1}] {name:<22} XPU peak={peak:.1f} MB")

    pool_size = infer.pool.compress_size / 1024 / 1024

    # ===== Draw curve with PIL =====
    W, H = 1400, 750
    margin_left, margin_right, margin_top, margin_bottom = 120, 60, 80, 120
    chart_w = W - margin_left - margin_right
    chart_h = H - margin_top - margin_bottom

    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arial.ttf", 20)
        font_label = ImageFont.truetype("arial.ttf", 14)
        font_tick = ImageFont.truetype("arial.ttf", 12)
        font_small = ImageFont.truetype("arial.ttf", 10)
    except Exception:
        font_title = ImageFont.load_default()
        font_label = font_title
        font_tick = font_title
        font_small = font_title

    # Title
    draw.text((W // 2, 20), "XPU Memory Curve During Sequential Inference",
              fill="black", font=font_title, anchor="mt")

    # Y range
    all_vals = pt_points + gi_points + [pool_size]
    y_max = max(all_vals) * 1.15
    y_min = 0
    y_range = y_max - y_min

    def y_to_px(v):
        return margin_top + chart_h - (v - y_min) / y_range * chart_h

    # Grid lines
    y_steps = 5
    for i in range(y_steps + 1):
        val = y_min + y_range * i / y_steps
        y = y_to_px(val)
        draw.line([(margin_left, y), (W - margin_right, y)], fill="#E0E0E0", width=1)
        draw.text((margin_left - 10, y), f"{val:.0f}", fill="gray", font=font_tick, anchor="rm")

    # Y axis label
    draw.text((15, margin_top + chart_h // 2), "XPU Memory (MB)", fill="black", font=font_label, anchor="mm")

    # X axis: steps 0..n
    x_step = chart_w / n
    pt_color = (231, 76, 60)
    gi_color = (52, 152, 219)
    pool_color = (46, 204, 113)

    # X tick labels
    step_labels = ["init"] + res_labels
    for i in range(n + 1):
        cx = margin_left + x_step * i
        draw.text((cx, margin_top + chart_h + 10), step_labels[i],
                  fill="black", font=font_tick, anchor="mt")

    # X axis label
    draw.text((W // 2, H - 15), "Step (init -> sequential inference)", fill="black", font=font_label, anchor="mb")

    # ---- PyTorch curve ----
    pt_pts_px = [(margin_left + x_step * i, y_to_px(pt_points[i])) for i in range(n + 1)]
    for i in range(len(pt_pts_px) - 1):
        draw.line([pt_pts_px[i], pt_pts_px[i + 1]], fill=pt_color, width=3)
    for i, (px, py) in enumerate(pt_pts_px):
        r = 6
        draw.ellipse([px - r, py - r, px + r, py + r], fill=pt_color, outline="white", width=2)
        draw.text((px, py - 18), f"{pt_points[i]:.0f}",
                  fill=pt_color, font=font_small, anchor="mb")

    # ---- GraphInfer curve ----
    gi_pts_px = [(margin_left + x_step * i, y_to_px(gi_points[i])) for i in range(n + 1)]
    for i in range(len(gi_pts_px) - 1):
        draw.line([gi_pts_px[i], gi_pts_px[i + 1]], fill=gi_color, width=3)
    for i, (px, py) in enumerate(gi_pts_px):
        r = 6
        draw.ellipse([px - r, py - r, px + r, py + r], fill=gi_color, outline="white", width=2)
        draw.text((px, py + 14), f"{gi_points[i]:.0f}",
                  fill=gi_color, font=font_small, anchor="mt")

    # Pool size line (horizontal dashed)
    pool_y = y_to_px(pool_size)
    for dx in range(margin_left, W - margin_right, 12):
        draw.line([(dx, pool_y), (min(dx + 6, W - margin_right), pool_y)], fill=pool_color, width=3)
    draw.text((W - margin_right - 10, pool_y - 18),
              f"GraphInfer Pool: {pool_size:.0f} MB", fill=pool_color, font=font_label, anchor="rb")

    # Legend
    lx, ly = margin_left + 20, margin_top + 20
    draw.line([(lx, ly + 7), (lx + 30, ly + 7)], fill=pt_color, width=3)
    draw.ellipse([lx + 12 - 4, ly + 7 - 4, lx + 12 + 4, ly + 7 + 4], fill=pt_color)
    draw.text((lx + 38, ly + 7), "PyTorch", fill="black", font=font_label, anchor="lm")

    draw.line([(lx + 120, ly + 7), (lx + 150, ly + 7)], fill=gi_color, width=3)
    draw.ellipse([lx + 132 - 4, ly + 7 - 4, lx + 132 + 4, ly + 7 + 4], fill=gi_color)
    draw.text((lx + 158, ly + 7), "GraphInfer", fill="black", font=font_label, anchor="lm")

    img.save("xpu_memory_curve.png")
    print(f"\nPlot saved to xpu_memory_curve.png")
    img.show()


if __name__ == "__main__":
    main()
