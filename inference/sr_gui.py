"""
超分模型图像处理工具 - 图形界面版

基于 GraphInfer 的超分推理 GUI 工具，支持：
  1. 通过文件对话框打开图像
  2. 显示原图 / 超分结果 / bicubic 对比
  3. 保存结果图像
  4. 批量处理文件夹
  5. Profile 分析

用法：
  # 启动 GUI
  python inference/sr_gui.py --onnx-path ./realesrgan-x4.onnx

  # 指定设备
  python inference/sr_gui.py --onnx-path ./realesrgan-x4.onnx --device cpu
"""

import argparse
import time
import os
import sys
import glob
import threading
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk

import torch
import numpy as np
from PIL import Image, ImageTk

# 添加父目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================================
# 图像 IO 工具
# ===========================================================================

def imread(path: str) -> np.ndarray:
    """读取图像文件，返回 RGB uint8 [H, W, 3]"""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def imwrite(path: str, img: np.ndarray):
    """写入图像文件，输入 RGB uint8 [H, W, 3]"""
    Image.fromarray(img).save(path)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """将模型输出的 CHW float32 tensor 转为 HWC uint8 图像

    RealESRGAN 的输入输出范围都是 [0, 1]，输出时乘 255 转 uint8。
    """
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    return arr


def image_to_tensor(img: np.ndarray, device: str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """将 HWC uint8 图像转为 NCHW float32 tensor

    RealESRGAN 的输入输出范围都是 [0, 1]，需要将 uint8 [0,255] 归一化到 [0,1]。
    """
    arr = img.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def make_comparison_grid(original: np.ndarray, sr: np.ndarray, bicubic: Optional[np.ndarray] = None) -> np.ndarray:
    """拼接对比图：原图 | 超分结果 (| bicubic)"""
    h_orig, w_orig = original.shape[:2]
    h_sr, w_sr = sr.shape[:2]

    scale = h_sr / h_orig
    orig_resized = np.array(Image.fromarray(original).resize(
        (int(w_orig * scale), h_sr), Image.BICUBIC))

    if bicubic is not None:
        h_bic, w_bic = bicubic.shape[:2]
        if h_bic != h_sr or w_bic != w_sr:
            bicubic = np.array(Image.fromarray(bicubic).resize((w_sr, h_sr), Image.BICUBIC))
        grid = np.concatenate([orig_resized, sr, bicubic], axis=1)
        grid = _add_label(grid, "Original (resized)", 0)
        grid = _add_label(grid, "GraphInfer SR", w_sr)
        grid = _add_label(grid, "Bicubic", w_sr * 2)
    else:
        grid = np.concatenate([orig_resized, sr], axis=1)
        grid = _add_label(grid, "Original (resized)", 0)
        grid = _add_label(grid, "GraphInfer SR", w_sr)

    return grid


def _add_label(img: np.ndarray, text: str, x_offset: int) -> np.ndarray:
    """在图像顶部添加文字标签"""
    from PIL import ImageDraw, ImageFont
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.rectangle([x_offset, 0, x_offset + tw + 8, th + 8], fill=(0, 0, 0))
    draw.text((x_offset + 4, 4), text, fill=(255, 255, 255), font=font)
    return np.array(pil)


# ===========================================================================
# 超分推理引擎
# ===========================================================================

class SREngine:
    """超分推理引擎，封装 GraphInfer 的初始化和推理"""

    def __init__(self, onnx_path: str, device: str = "xpu", dtype: torch.dtype = torch.float32):
        self.onnx_path = onnx_path
        self.device = device
        self.dtype = dtype
        self.scale = self._detect_scale(onnx_path)
        self.engine = None

    @staticmethod
    def _detect_scale(onnx_path: str) -> int:
        basename = os.path.basename(onnx_path).lower()
        if "x2" in basename or "2x" in basename:
            return 2
        elif "x3" in basename or "3x" in basename:
            return 3
        elif "x4" in basename or "4x" in basename:
            return 4
        return 4

    def initialize(self, max_height: int = 2160, max_width: int = 3840):
        """初始化 GraphInfer"""
        from inference.infer import GraphInfer

        # 从 ONNX 模型自动检测输入名称
        import onnx
        onnx_model = onnx.load(self.onnx_path)
        input_name = onnx_model.graph.input[0].name
        output_name = onnx_model.graph.output[0].name
        self.input_name = input_name
        self.output_name = output_name

        input_desc = {input_name: (1, 3, "height", "width")}
        input_range = {
            "height": (1, max_height),
            "width": (1, max_width),
        }

        t0 = time.perf_counter()
        self.engine = GraphInfer(
            self.onnx_path,
            input_desc,
            input_range,
            dtype=self.dtype,
            device=self.device,
        )
        elapsed = time.perf_counter() - t0
        return elapsed

    def warmup(self, height: int = 360, width: int = 640, iters: int = 2):
        """用指定分辨率预热"""
        if self.engine is None:
            raise RuntimeError("Engine not initialized.")
        dummy = torch.randn(1, 3, height, width, device=self.device, dtype=self.dtype)
        for i in range(iters):
            self.engine.forward({self.input_name: dummy}, debug=False)
            if self.device == "xpu":
                torch.xpu.synchronize()

    def process(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """处理单张图像，返回 (sr_img, elapsed_ms)"""
        if self.engine is None:
            raise RuntimeError("Engine not initialized.")

        h, w = img.shape[:2]
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        tensor = image_to_tensor(img, device=self.device, dtype=self.dtype)

        t0 = time.perf_counter()
        outputs = self.engine.forward({self.input_name: tensor}, debug=False)
        if self.device == "xpu":
            torch.xpu.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000

        sr_tensor = outputs[self.output_name]
        sr_img = tensor_to_image(sr_tensor)
        sr_img = sr_img[:h * self.scale, :w * self.scale, :]

        return sr_img, elapsed

    def process_bicubic(self, img: np.ndarray) -> np.ndarray:
        """用 bicubic 插值做对比 baseline"""
        h, w = img.shape[:2]
        pil_img = Image.fromarray(img)
        sr = pil_img.resize((w * self.scale, h * self.scale), Image.BICUBIC)
        return np.array(sr, dtype=np.uint8)


# ===========================================================================
# 图形界面
# ===========================================================================

class SRApp:
    """超分图像处理 GUI"""

    def __init__(self, engine: SREngine):
        self.engine = engine
        self.current_img: Optional[np.ndarray] = None
        self.current_sr: Optional[np.ndarray] = None
        self.current_bicubic: Optional[np.ndarray] = None
        self.current_path: Optional[str] = None

        self._build_ui()
        self._update_status("Ready. Open an image to start.")

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title(f"Super Resolution - {os.path.basename(self.engine.onnx_path)} ({self.engine.scale}x)")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)

        # ---- 顶部工具栏 ----
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="📂 Open Image", command=self._on_open).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📁 Batch Folder", command=self._on_batch).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="💾 Save Result", command=self._on_save).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📊 Profile", command=self._on_profile).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🖼 Show Comparison", command=self._on_compare).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self._model_label = ttk.Label(toolbar,
            text=f"Model: {os.path.basename(self.engine.onnx_path)} | {self.engine.scale}x | {self.engine.device}")
        self._model_label.pack(side=tk.LEFT, padx=4)

        # ---- 信息栏 ----
        info_frame = ttk.Frame(self.root, padding=4)
        info_frame.pack(side=tk.TOP, fill=tk.X)
        self._info_var = tk.StringVar(value="No image loaded.")
        ttk.Label(info_frame, textvariable=self._info_var, font=("Consolas", 10)).pack(side=tk.LEFT)

        # ---- 主区域：图像显示 ----
        view_frame = ttk.Frame(self.root, padding=4)
        view_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 左侧：原图
        left_frame = ttk.LabelFrame(view_frame, text="Original", padding=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self._orig_canvas = tk.Canvas(left_frame, bg="#1e1e1e", highlightthickness=0)
        self._orig_canvas.pack(fill=tk.BOTH, expand=True)
        self._orig_canvas.bind("<Configure>", lambda e: self._update_display())

        # 右侧：超分结果
        right_frame = ttk.LabelFrame(view_frame, text="Super Resolution", padding=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2)
        self._sr_canvas = tk.Canvas(right_frame, bg="#1e1e1e", highlightthickness=0)
        self._sr_canvas.pack(fill=tk.BOTH, expand=True)
        self._sr_canvas.bind("<Configure>", lambda e: self._update_display())

        # ---- 底部状态栏 ----
        status_frame = ttk.Frame(self.root, padding=2)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(status_frame, textvariable=self._status_var, font=("Consolas", 9)).pack(side=tk.LEFT)

        # 缓存 PhotoImage 防止被 GC
        self._orig_photo = None
        self._sr_photo = None

    # ---- 显示更新 ----

    def _update_display(self):
        """更新左右两个画布的图像显示"""
        # 原图画布
        if self.current_img is not None:
            cw = self._orig_canvas.winfo_width()
            ch = self._orig_canvas.winfo_height()
            if cw > 10 and ch > 10:
                pil_img = Image.fromarray(self.current_img)
                pil_img.thumbnail((cw, ch), Image.LANCZOS)
                self._orig_photo = ImageTk.PhotoImage(pil_img)
                self._orig_canvas.delete("all")
                x = (cw - pil_img.width) // 2
                y = (ch - pil_img.height) // 2
                self._orig_canvas.create_image(x, y, anchor=tk.NW, image=self._orig_photo)

        # 超分结果画布
        if self.current_sr is not None:
            cw = self._sr_canvas.winfo_width()
            ch = self._sr_canvas.winfo_height()
            if cw > 10 and ch > 10:
                pil_img = Image.fromarray(self.current_sr)
                pil_img.thumbnail((cw, ch), Image.LANCZOS)
                self._sr_photo = ImageTk.PhotoImage(pil_img)
                self._sr_canvas.delete("all")
                x = (cw - pil_img.width) // 2
                y = (ch - pil_img.height) // 2
                self._sr_canvas.create_image(x, y, anchor=tk.NW, image=self._sr_photo)

    def _update_status(self, text: str):
        self._status_var.set(text)
        self.root.update_idletasks()

    def _update_info(self):
        if self.current_img is None:
            self._info_var.set("No image loaded.")
            return
        h, w = self.current_img.shape[:2]
        parts = [f"Input: {w}x{h}"]
        if self.current_sr is not None:
            sh, sw = self.current_sr.shape[:2]
            parts.append(f"Output: {sw}x{sh} ({self.engine.scale}x)")
        if self.current_path:
            parts.append(f"File: {os.path.basename(self.current_path)}")
        self._info_var.set(" | ".join(parts))

    # ---- 事件处理 ----

    def _on_open(self):
        """打开单张图像"""
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.current_path = path
        self.current_img = imread(path)
        self._update_info()
        self._update_display()
        self._update_status(f"Loaded: {os.path.basename(path)} ({self.current_img.shape[1]}x{self.current_img.shape[0]})")

        # 后台线程处理
        self._process_in_thread()

    def _process_in_thread(self):
        """在后台线程中执行超分推理"""
        if self.current_img is None:
            return

        def task():
            try:
                self._update_status("Processing (bicubic)...")
                self.current_bicubic = self.engine.process_bicubic(self.current_img)

                self._update_status("Processing (GraphInfer SR)...")
                sr_img, elapsed = self.engine.process(self.current_img)
                self.current_sr = sr_img

                h, w = self.current_img.shape[:2]
                self.root.after(0, self._update_info)
                self.root.after(0, self._update_display)
                self.root.after(0, lambda: self._update_status(
                    f"Done: {w}x{h} -> {w * self.engine.scale}x{h * self.engine.scale}  ({elapsed:.1f} ms)"))
            except Exception as e:
                self.root.after(0, lambda: self._update_status(f"Error: {e}"))
                import traceback
                traceback.print_exc()

        threading.Thread(target=task, daemon=True).start()

    def _on_batch(self):
        """批量处理文件夹"""
        folder = filedialog.askdirectory(title="Select a folder with images")
        if not folder:
            return

        out_dir = os.path.join(folder, "sr_output")
        os.makedirs(out_dir, exist_ok=True)

        extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.webp")
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(folder, ext)))
            files.extend(glob.glob(os.path.join(folder, ext.upper())))
        files.sort()

        if not files:
            messagebox.showinfo("Info", f"No image files found in {folder}")
            return

        self._update_status(f"Batch processing {len(files)} images...")

        def task():
            try:
                total_time = 0.0
                for i, fpath in enumerate(files):
                    fname = os.path.basename(fpath)
                    name_no_ext = os.path.splitext(fname)[0]
                    self.root.after(0, lambda msg=f"  [{i+1}/{len(files)}] {fname}": self._update_status(msg))

                    img = imread(fpath)
                    sr_img, elapsed = self.engine.process(img)
                    total_time += elapsed

                    sr_path = os.path.join(out_dir, f"{name_no_ext}_sr.png")
                    imwrite(sr_path, sr_img)

                    bicubic = self.engine.process_bicubic(img)
                    grid = make_comparison_grid(img, sr_img, bicubic)
                    grid_path = os.path.join(out_dir, f"{name_no_ext}_compare.png")
                    imwrite(grid_path, grid)

                avg = total_time / len(files)
                msg = f"Batch done: {len(files)} images, total {total_time/1000:.2f}s, avg {avg:.1f}ms/image -> {out_dir}"
                self.root.after(0, lambda: self._update_status(msg))
                self.root.after(0, lambda: messagebox.showinfo("Batch Complete", msg))
            except Exception as e:
                self.root.after(0, lambda: self._update_status(f"Batch error: {e}"))
                import traceback
                traceback.print_exc()

        threading.Thread(target=task, daemon=True).start()

    def _on_save(self):
        """保存超分结果"""
        if self.current_sr is None:
            messagebox.showinfo("Info", "No result to save. Open an image first.")
            return

        default_name = "sr_result.png"
        if self.current_path:
            base = os.path.splitext(os.path.basename(self.current_path))[0]
            default_name = f"{base}_sr.png"

        path = filedialog.asksaveasfilename(
            title="Save super resolution result",
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All files", "*.*")],
        )
        if not path:
            return

        imwrite(path, self.current_sr)
        self._update_status(f"Saved: {path}")

        # 同时保存对比图
        if self.current_bicubic is not None:
            base = os.path.splitext(path)[0]
            grid = make_comparison_grid(self.current_img, self.current_sr, self.current_bicubic)
            grid_path = f"{base}_compare.png"
            imwrite(grid_path, grid)

    def _on_compare(self):
        """显示对比图（用系统图片查看器打开）"""
        if self.current_sr is None:
            messagebox.showinfo("Info", "No result to compare. Open an image first.")
            return
        if self.current_bicubic is None:
            messagebox.showinfo("Info", "Bicubic result not available.")
            return

        grid = make_comparison_grid(self.current_img, self.current_sr, self.current_bicubic)
        tmp_path = "_compare_temp.png"
        imwrite(tmp_path, grid)
        self._update_status(f"Comparison saved: {os.path.abspath(tmp_path)}")
        try:
            os.startfile(tmp_path)
        except Exception:
            pass

    def _on_profile(self):
        """对当前图像做 profile 分析"""
        if self.current_img is None:
            messagebox.showinfo("Info", "No image loaded. Open an image first.")
            return

        self._update_status("Profiling...")

        def task():
            try:
                tensor = image_to_tensor(self.current_img, device=self.engine.device, dtype=self.engine.dtype)
                result = self.engine.engine.forward({self.engine.input_name: tensor}, debug=False, profile=True)
                if self.engine.device == "xpu":
                    torch.xpu.synchronize()

                # 收集 profile 输出到字符串
                import io
                buf = io.StringIO()
                old_stdout = sys.stdout
                sys.stdout = buf
                try:
                    self.engine.engine.print_profile(result["__profile__"])
                finally:
                    sys.stdout = old_stdout
                profile_text = buf.getvalue()

                # 弹出窗口显示
                self.root.after(0, lambda: self._show_profile_window(profile_text))
                self.root.after(0, lambda: self._update_status("Profile done."))
            except Exception as e:
                self.root.after(0, lambda: self._update_status(f"Profile error: {e}"))

        threading.Thread(target=task, daemon=True).start()

    def _show_profile_window(self, text: str):
        """弹出 profile 结果窗口"""
        win = tk.Toplevel(self.root)
        win.title("Profile Results")
        win.geometry("800x500")
        text_widget = tk.Text(win, font=("Consolas", 10), bg="#1e1e1e", fg="#d4d4d4", wrap=tk.NONE)
        text_widget.insert(tk.END, text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)
        # 滚动条
        h_scroll = ttk.Scrollbar(win, orient=tk.HORIZONTAL, command=text_widget.xview)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll = ttk.Scrollbar(win, orient=tk.VERTICAL, command=text_widget.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

    def run(self):
        self.root.mainloop()


# ===========================================================================
# 命令行入口
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="超分图像处理工具 - GUI")
    parser.add_argument("--onnx-path", type=str, required=True,
                        help="ONNX 模型路径")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备（xpu/cpu，默认自动选择）")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float32", "float16"],
                        help="推理精度")
    parser.add_argument("--max-size", type=int, default=1080,
                        help="最大输入高度（默认 1080")
    parser.add_argument("--no-warmup", action="store_true",
                        help="跳过预热")
    args = parser.parse_args()

    # ---- 设备选择 ----
    if args.device:
        device = args.device
    else:
        device = "xpu" if torch.xpu.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    print(f"Device: {device}, dtype: {args.dtype}")

    # ---- 检查模型 ----
    if not os.path.isfile(args.onnx_path):
        print(f"Error: ONNX model not found: {args.onnx_path}")
        sys.exit(1)

    # ---- 初始化引擎 ----
    engine = SREngine(args.onnx_path, device=device, dtype=dtype)
    print(f"Initializing GraphInfer (max {args.max_size}, scale={engine.scale}x)...")
    init_time = engine.initialize(max_height=args.max_size, max_width=int(args.max_size * 16 / 9))
    engine.engine.print_summary()
    print(f"Init time: {init_time:.2f}s")

    if not args.no_warmup:
        print("Warming up...")
        engine.warmup(height=360, width=640, iters=2)
        print("Warmup done.")

    # ---- 启动 GUI ----
    app = SRApp(engine)
    app.run()


if __name__ == "__main__":
    main()
