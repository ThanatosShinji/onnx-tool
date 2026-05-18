"""
YOLO 目标检测模型 Precision-Recall / mAP 评估脚本

基于 onnx_tool + GraphInfer 推理引擎，支持：
  - yolo26l.onnx (YOLOv6-L) 及同类 YOLO 检测模型
  - COCO 格式标注数据集
  - mAP@0.5, mAP@0.5:0.95 计算
  - Precision-Recall 曲线数据导出
  - 单张/批量推理 + Profile 分析

用法：
  # 使用 yolo26l.onnx + COCO val2017
  python inference/yolo_eval.py --onnx yolo26l.onnx --coco-path /path/to/coco

  # 使用自定义图片目录 + 标注文件
  python inference/yolo_eval.py --onnx yolo26l.onnx --image-dir /path/to/images --anno-file /path/to/annotations.json

  # 仅推理少量图片做快速测试
  python inference/yolo_eval.py --onnx yolo26l.onnx --max-images 10

  # 启用 profile 分析
  python inference/yolo_eval.py --onnx yolo26l.onnx --profile

  # 导出 PR 曲线数据
  python inference/yolo_eval.py --onnx yolo26l.onnx --export-pr pr_data.json

  # 保存标注后的图片（预测框）
  python inference/yolo_eval.py --onnx yolo26l.onnx --image-dir /path/to/images --save-dir ./output

  # 保存标注图片（预测框 + GT 框对比）
  python inference/yolo_eval.py --onnx yolo26l.onnx --image-dir /path/to/images --anno-file /path/to/annotations.json --save-dir ./output --draw-gt

依赖：
  pip install pycocotools opencv-python
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# 添加父目录到 path，确保可以 import onnx_tool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer import GraphInfer
from kernels import KernelRegistry


# ===========================================================================
# YOLO 后处理：将模型输出解码为检测框
# ===========================================================================

def yolo_postprocess(
    output: np.ndarray,
    input_shape: Tuple[int, int],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> List[np.ndarray]:
    """
    YOLO 通用后处理：将 [1, N, 6] 输出解码为每张图的检测结果。

    输出格式 [batch_id, x1, y1, x2, y2, class_id, confidence] (xyxy 绝对坐标)。

    Args:
        output: 模型原始输出 [1, N, 6] 或 [N, 6]
        input_shape: (H, W) 输入图像尺寸
        conf_thres: 置信度阈值
        iou_thres: NMS IoU 阈值
        max_det: 每张图最大检测数

    Returns:
        list of np.ndarray, 每个元素为 [M, 7] (batch_id, x1, y1, x2, y2, cls, conf)
    """
    if output.ndim == 3:
        output = output[0]  # [N, 6]

    # Ultralytics YOLO end2end 输出格式: [x1, y1, x2, y2, confidence, class_id]
    # x1, y1, x2, y2 为 640x640 空间中的绝对像素坐标 (xyxy)
    if output.shape[1] < 6:
        return [np.zeros((0, 7), dtype=np.float32)]

    # 提取各分量
    x1 = output[:, 0]
    y1 = output[:, 1]
    x2 = output[:, 2]
    y2 = output[:, 3]
    conf = output[:, 4]
    cls_ids = output[:, 5]

    # 置信度过滤
    mask = conf > conf_thres
    x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
    conf = conf[mask]
    cls_ids = cls_ids[mask]

    if len(x1) == 0:
        return [np.zeros((0, 7), dtype=np.float32)]

    # 裁剪到图像范围内
    H, W = input_shape
    x1 = np.clip(x1, 0, W)
    y1 = np.clip(y1, 0, H)
    x2 = np.clip(x2, 0, W)
    y2 = np.clip(y2, 0, H)

    # 过滤无效框
    valid = (x2 > x1) & (y2 > y1)
    x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
    conf = conf[valid]
    cls_ids = cls_ids[valid]

    if len(x1) == 0:
        return [np.zeros((0, 7), dtype=np.float32)]

    # 构建检测结果
    dets = np.stack([
        np.zeros(len(x1)),  # batch_id = 0
        x1, y1, x2, y2,
        cls_ids,
        conf,
    ], axis=1)

    # NMS (per-class)
    keep = _nms_per_class(dets, iou_thres)
    dets = dets[keep]

    # 按置信度排序，取 top max_det
    if len(dets) > max_det:
        order = np.argsort(-dets[:, 6])
        dets = dets[order[:max_det]]

    return [dets]


def _nms_per_class(dets: np.ndarray, iou_thres: float) -> np.ndarray:
    """逐类 NMS，返回保留的索引"""
    if len(dets) == 0:
        return np.array([], dtype=np.int64)

    keep = []
    classes = np.unique(dets[:, 5])
    for cls_id in classes:
        cls_mask = dets[:, 5] == cls_id
        cls_dets = dets[cls_mask]
        cls_indices = np.where(cls_mask)[0]

        # 按置信度降序
        order = np.argsort(-cls_dets[:, 6])
        cls_dets = cls_dets[order]
        cls_indices = cls_indices[order]

        while len(cls_dets) > 0:
            keep.append(cls_indices[0])
            if len(cls_dets) == 1:
                break
            ious = _box_iou(cls_dets[0:1, 1:5], cls_dets[1:, 1:5])[0]
            mask = ious <= iou_thres
            cls_dets = cls_dets[1:][mask]
            cls_indices = cls_indices[1:][mask]

    return np.array(sorted(keep), dtype=np.int64)


def _box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """计算两组 xyxy 框的 IoU 矩阵 [N, M]"""
    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter

    return inter / np.maximum(union, 1e-7)


# ===========================================================================
# COCO 评估指标
# ===========================================================================

def compute_coco_metrics(
    all_dets: List[Dict],
    gt_annotations: Dict,
    num_classes: int = 80,
    iou_thresholds: Optional[List[float]] = None,
) -> Dict:
    """
    计算 COCO 风格 mAP 指标。

    Args:
        all_dets: 每张图的检测结果列表，每个元素为 dict:
                  {'image_id': int, 'boxes': [N,4] xyxy, 'scores': [N], 'labels': [N]}
        gt_annotations: COCO 格式标注 dict，包含 'images', 'annotations', 'categories'
        num_classes: 类别数
        iou_thresholds: IoU 阈值列表，默认 [0.5, 0.55, ..., 0.95]

    Returns:
        metrics dict
    """
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.5, 0.95, 10).tolist()

    # 构建 GT 索引: image_id -> [annotations]
    gt_by_image = defaultdict(list)
    for ann in gt_annotations['annotations']:
        gt_by_image[ann['image_id']].append(ann)

    # 按类别组织预测
    preds_by_class = defaultdict(list)  # class_id -> [(image_id, score, box, matched)]
    for det in all_dets:
        image_id = det['image_id']
        boxes = det['boxes']
        scores = det['scores']
        labels = det['labels']
        for i in range(len(boxes)):
            preds_by_class[int(labels[i])].append({
                'image_id': image_id,
                'score': float(scores[i]),
                'box': boxes[i].tolist(),
            })

    # 按置信度排序
    for cls_id in preds_by_class:
        preds_by_class[cls_id].sort(key=lambda x: -x['score'])

    # 计算每个 IoU 阈值下的 AP
    ap_per_iou = {}
    for iou_thr in iou_thresholds:
        aps = []
        for cls_id in range(num_classes):
            preds = preds_by_class.get(cls_id, [])
            if not preds and not any(
                ann['category_id'] == cls_id
                for anns in gt_by_image.values()
                for ann in anns
            ):
                continue  # 该类别无 GT 也无预测，跳过

            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))
            # 记录每个 GT 是否已被匹配
            gt_matched = defaultdict(set)  # image_id -> set of annotation indices

            for i, pred in enumerate(preds):
                image_id = pred['image_id']
                gt_anns = gt_by_image.get(image_id, [])
                gt_boxes = []
                gt_indices = []
                for j, ann in enumerate(gt_anns):
                    if ann['category_id'] == cls_id and j not in gt_matched[image_id]:
                        gt_boxes.append(ann['bbox'])  # COCO bbox: [x, y, w, h]
                        gt_indices.append(j)

                if not gt_boxes:
                    fp[i] = 1
                    continue

                # 转换 COCO xywh -> xyxy
                gt_boxes_xyxy = np.array([
                    [b[0], b[1], b[0] + b[2], b[1] + b[3]]
                    for b in gt_boxes
                ])
                pred_box = np.array(pred['box']).reshape(1, 4)
                ious = _box_iou(pred_box, gt_boxes_xyxy)[0]
                best_iou_idx = np.argmax(ious)
                best_iou = ious[best_iou_idx]

                if best_iou >= iou_thr:
                    tp[i] = 1
                    gt_matched[image_id].add(gt_indices[best_iou_idx])
                else:
                    fp[i] = 1

            # 累积 TP/FP
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            # 总 GT 数
            total_gt = sum(
                1 for anns in gt_by_image.values()
                for ann in anns if ann['category_id'] == cls_id
            )

            if total_gt == 0:
                continue

            recalls = tp_cumsum / total_gt
            precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-7)

            # 101-point interpolated AP
            ap = _compute_ap(recalls, precisions)
            aps.append(ap)

        ap_per_iou[f'mAP@{iou_thr:.2f}'] = np.mean(aps) if aps else 0.0

    # mAP@0.5
    mAP50 = ap_per_iou.get('mAP@0.50', 0.0)
    # mAP@0.5:0.95
    mAP = np.mean(list(ap_per_iou.values()))

    return {
        'mAP@0.5': mAP50,
        'mAP@0.5:0.95': mAP,
        'ap_per_iou': ap_per_iou,
    }


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """101-point interpolated average precision"""
    # 对 precision 做单调递减插值
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # 在 101 个 recall 点采样
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        idx = np.searchsorted(recalls, t, side='right')
        if idx < len(precisions):
            ap += precisions[idx]
    return ap / 101.0


# ===========================================================================
# 标注绘制
# ===========================================================================

# COCO 80 类名称
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]

# 20 种类别调色板（循环使用）
_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255), (255, 0, 128), (0, 255, 128),
    (128, 128, 0), (128, 0, 128), (0, 128, 128), (192, 192, 192),
    (128, 128, 128), (255, 128, 128), (128, 255, 128), (128, 128, 255),
]


def get_class_name(cls_id: int, categories: Optional[List[Dict]] = None) -> str:
    """根据类别 ID 获取类别名称"""
    if categories:
        for cat in categories:
            if cat['id'] == cls_id:
                return cat['name']
    if 0 <= cls_id < len(COCO_CLASSES):
        return COCO_CLASSES[cls_id]
    return f'cls_{cls_id}'


def draw_detections(
    image: np.ndarray,
    dets: np.ndarray,
    categories: Optional[List[Dict]] = None,
    conf_thres: float = 0.25,
    line_width: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    在图片上绘制检测框和标签。

    Args:
        image: BGR 图片 (H, W, 3) uint8
        dets: 检测结果 [M, 7] (batch_id, x1, y1, x2, y2, cls, conf)
        categories: COCO categories 列表，用于获取类别名称
        conf_thres: 低于此置信度的框不绘制
        line_width: 框线宽度
        font_scale: 文字大小

    Returns:
        绘制后的 BGR 图片
    """
    import cv2
    img = image.copy()
    H, W = img.shape[:2]

    for det in dets:
        if len(det) < 7:
            continue
        x1, y1, x2, y2 = int(det[1]), int(det[2]), int(det[3]), int(det[4])
        cls_id = int(det[5])
        conf = float(det[6])

        if conf < conf_thres:
            continue

        # 裁剪到图像范围
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        color = _COLORS[cls_id % len(_COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)

        # 标签文字
        cls_name = get_class_name(cls_id, categories)
        label = f'{cls_name} {conf:.2f}'
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        # 标签背景
        label_y1 = max(0, y1 - th - baseline - 4)
        label_y2 = y1
        cv2.rectangle(img, (x1, label_y1), (x1 + tw + 4, label_y2), color, -1)

        # 标签文字（白色）
        cv2.putText(img, label, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    return img


def draw_gt_boxes(
    image: np.ndarray,
    annotations: List[Dict],
    categories: Optional[List[Dict]] = None,
    line_width: int = 2,
) -> np.ndarray:
    """
    在图片上绘制 Ground Truth 标注框（虚线风格）。

    Args:
        image: BGR 图片 (H, W, 3) uint8
        annotations: COCO 格式的标注列表，每个包含 bbox (xywh), category_id
        categories: COCO categories 列表
        line_width: 框线宽度

    Returns:
        绘制后的 BGR 图片
    """
    import cv2
    img = image.copy()

    for ann in annotations:
        x, y, w, h = ann['bbox']
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cls_id = ann['category_id']
        color = _COLORS[cls_id % len(_COLORS)]

        # 虚线框
        dash_len = 8
        for dx in range(x1, x2, dash_len * 2):
            ex = min(dx + dash_len, x2)
            cv2.line(img, (dx, y1), (ex, y1), color, line_width)
            cv2.line(img, (dx, y2), (ex, y2), color, line_width)
        for dy in range(y1, y2, dash_len * 2):
            ey = min(dy + dash_len, y2)
            cv2.line(img, (x1, dy), (x1, ey), color, line_width)
            cv2.line(img, (x2, dy), (x2, ey), color, line_width)

        # GT 标签
        cls_name = get_class_name(cls_id, categories)
        label = f'GT: {cls_name}'
        font_scale = 0.4
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    return img


# ===========================================================================
# 数据加载
# ===========================================================================

def load_coco_annotations(coco_path: str) -> Dict:
    """加载 COCO 格式标注文件 (instances_val2017.json)"""
    with open(coco_path, 'r') as f:
        return json.load(f)


def load_image(image_path: str, target_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    加载并预处理图像（letterbox resize 保持宽高比 + /255.0 归一化）。

    Ultralytics YOLO 标准导出模型需要外部 /255.0 归一化。
    输入 BGR 图片，值范围 [0, 255]，输出 [0, 1]。

    Args:
        image_path: 图像路径
        target_size: (H, W) 目标尺寸

    Returns:
        (image, letterbox_info) 其中:
          - image: (1, 3, H, W) float32 numpy array, 值范围 [0, 1]
          - letterbox_info: (pad_left, pad_top, scale)
    """
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        H_orig, W_orig = img.shape[:2]
        H_dst, W_dst = target_size

        # Letterbox: 保持宽高比，缩放后填充到目标尺寸
        scale = min(H_dst / H_orig, W_dst / W_orig)
        new_h = int(H_orig * scale)
        new_w = int(W_orig * scale)

        img = cv2.resize(img, (new_w, new_h))

        # 填充到目标尺寸 (114 = 灰色填充，YOLO 标准)
        pad_h = H_dst - new_h
        pad_w = W_dst - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        # letterbox_info: (pad_left, pad_top, scale)
        letterbox_info = (pad_left, pad_top, scale)
        return img[np.newaxis, ...], letterbox_info
    except ImportError:
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        W_orig, H_orig = img.size

        H_dst, W_dst = target_size
        scale = min(H_dst / H_orig, W_dst / W_orig)
        new_h = int(H_orig * scale)
        new_w = int(W_orig * scale)

        img = img.resize((new_w, new_h))

        pad_h = H_dst - new_h
        pad_w = W_dst - new_w
        pad_top = pad_h // 2
        pad_left = pad_w // 2

        # PIL padding
        from PIL import ImageOps
        img = ImageOps.pad(img, (W_dst, H_dst), color=(114, 114, 114))

        img = np.array(img, dtype=np.float32) / 255.0
        img = img[:, :, ::-1]  # RGB -> BGR
        img = np.transpose(img, (2, 0, 1))

        letterbox_info = (pad_left, pad_top, scale)
        return img[np.newaxis, ...], letterbox_info


# ===========================================================================
# 主评估流程
# ===========================================================================

class YOLOEvaluator:
    """YOLO 检测模型评估器"""

    def __init__(
        self,
        onnx_path: str,
        input_size: Tuple[int, int] = (640, 640),
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.onnx_path = onnx_path
        self.input_size = input_size  # (H, W)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.dtype = dtype

        # 构建 GraphInfer
        input_desc = {
            'images': ('batch', 'channel', 'height', 'width'),
        }
        input_range = {
            'batch': (1, 1),
            'channel': (3, 3),
            'height': (input_size[0], input_size[0]),
            'width': (input_size[1], input_size[1]),
        }

        print(f"Loading ONNX model: {onnx_path}")
        t0 = time.time()
        self.infer = GraphInfer(
            onnx_path,
            input_desc=input_desc,
            input_range=input_range,
            dtype=dtype,
            device=device,
        )
        print(f"GraphInfer initialized in {time.time() - t0:.2f}s")
        self.infer.print_summary()

    def infer_single(self, image: np.ndarray) -> np.ndarray:
        """单张图片推理，返回原始输出 [1, N, 6]"""
        input_tensor = torch.from_numpy(image).to(device=self.device, dtype=self.dtype)
        outputs = self.infer.forward({'images': input_tensor})
        # 取第一个输出
        output_name = self.infer.compute_graph.output[0]
        return outputs[output_name].cpu().numpy()

    def evaluate_coco(
        self,
        coco_path: str,
        max_images: int = -1,
        profile: bool = False,
        save_dir: Optional[str] = None,
        draw_gt: bool = False,
    ) -> Dict:
        """
        在 COCO 验证集上评估。

        Args:
            coco_path: COCO 标注文件路径 (instances_val2017.json)
            max_images: 最大评估图片数 (-1 表示全部)
            profile: 是否启用 profile
            save_dir: 若指定，将标注后的图片保存到此目录
            draw_gt: 是否同时绘制 Ground Truth 标注

        Returns:
            metrics dict
        """
        gt = load_coco_annotations(coco_path)
        images = gt['images']

        if max_images > 0:
            images = images[:max_images]

        # 确定图片目录
        coco_dir = os.path.dirname(coco_path)
        image_dir = os.path.join(coco_dir, 'val2017')
        if not os.path.isdir(image_dir):
            # 尝试其他常见路径
            for candidate in ['images/val2017', 'val2017', '../val2017']:
                cand_path = os.path.join(coco_dir, candidate)
                if os.path.isdir(cand_path):
                    image_dir = cand_path
                    break

        all_dets = []
        total_time = 0.0
        num_classes = len(gt.get('categories', []))

        print(f"\nEvaluating {len(images)} images...")
        print(f"Image dir: {image_dir}")
        print(f"Classes: {num_classes}")

        for idx, img_info in enumerate(images):
            image_id = img_info['id']
            file_name = img_info['file_name']
            image_path = os.path.join(image_dir, file_name)

            if not os.path.isfile(image_path):
                print(f"  [{idx+1}/{len(images)}] SKIP: {file_name} not found")
                continue

            # 加载 + 推理
            img, lb_info = load_image(image_path, self.input_size)
            t0 = time.time()
            output = self.infer_single(img)
            elapsed = time.time() - t0
            total_time += elapsed

            # 后处理
            dets = yolo_postprocess(
                output, self.input_size,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
            )[0]

            all_dets.append({
                'image_id': image_id,
                'boxes': dets[:, 1:5] if len(dets) > 0 else np.zeros((0, 4)),
                'scores': dets[:, 6] if len(dets) > 0 else np.zeros(0),
                'labels': dets[:, 5] if len(dets) > 0 else np.zeros(0),
            })

            # 保存标注图片
            if save_dir:
                self._save_annotated_image(
                    image_path, dets, save_dir, file_name,
                    lb_info=lb_info,
                    gt_annotations=gt if draw_gt else None,
                    image_id=image_id,
                )

            if (idx + 1) % 10 == 0 or idx == len(images) - 1:
                avg_time = total_time / (idx + 1)
                print(f"  [{idx+1}/{len(images)}] avg {avg_time*1000:.1f}ms/img, "
                      f"dets: {len(dets)}")

        # 计算指标
        print(f"\nComputing COCO metrics...")
        metrics = compute_coco_metrics(all_dets, gt, num_classes=num_classes)

        metrics['num_images'] = len(images)
        metrics['total_time'] = total_time
        metrics['avg_time_ms'] = total_time / max(len(images), 1) * 1000
        metrics['fps'] = len(images) / total_time if total_time > 0 else 0

        return metrics

    def evaluate_image_dir(
        self,
        image_dir: str,
        anno_file: Optional[str] = None,
        max_images: int = -1,
        profile: bool = False,
        save_dir: Optional[str] = None,
        draw_gt: bool = False,
    ) -> Dict:
        """
        在图片目录上评估（可选标注文件）。

        Args:
            image_dir: 图片目录
            anno_file: 标注文件路径 (COCO JSON 格式)，可选
            max_images: 最大图片数
            profile: 是否启用 profile
            save_dir: 若指定，将标注后的图片保存到此目录
            draw_gt: 是否同时绘制 Ground Truth 标注

        Returns:
            metrics dict
        """
        # 收集图片
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = sorted([
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])
        if max_images > 0:
            image_files = image_files[:max_images]

        all_dets = []
        total_time = 0.0

        print(f"\nEvaluating {len(image_files)} images from {image_dir}")

        for idx, file_name in enumerate(image_files):
            image_path = os.path.join(image_dir, file_name)
            img, lb_info = load_image(image_path, self.input_size)
            t0 = time.time()
            output = self.infer_single(img)
            elapsed = time.time() - t0
            total_time += elapsed

            dets = yolo_postprocess(
                output, self.input_size,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
            )[0]

            all_dets.append({
                'image_id': idx,
                'file_name': file_name,
                'boxes': dets[:, 1:5] if len(dets) > 0 else np.zeros((0, 4)),
                'scores': dets[:, 6] if len(dets) > 0 else np.zeros(0),
                'labels': dets[:, 5] if len(dets) > 0 else np.zeros(0),
            })

            # 保存标注图片
            if save_dir:
                gt = None
                if draw_gt and anno_file and os.path.isfile(anno_file):
                    gt = load_coco_annotations(anno_file)
                self._save_annotated_image(
                    image_path, dets, save_dir, file_name,
                    lb_info=lb_info,
                    gt_annotations=gt,
                    image_id=idx,
                )

            if (idx + 1) % 10 == 0 or idx == len(image_files) - 1:
                avg_time = total_time / (idx + 1)
                print(f"  [{idx+1}/{len(image_files)}] avg {avg_time*1000:.1f}ms/img, "
                      f"dets: {len(dets)}")

        metrics = {
            'num_images': len(image_files),
            'total_time': total_time,
            'avg_time_ms': total_time / max(len(image_files), 1) * 1000,
            'fps': len(image_files) / total_time if total_time > 0 else 0,
            'detections': all_dets,
        }

        # 如果有标注文件，计算 mAP
        if anno_file and os.path.isfile(anno_file):
            gt = load_coco_annotations(anno_file)
            num_classes = len(gt.get('categories', []))
            coco_metrics = compute_coco_metrics(all_dets, gt, num_classes=num_classes)
            metrics.update(coco_metrics)

        return metrics

    def _save_annotated_image(
        self,
        image_path: str,
        dets: np.ndarray,
        save_dir: str,
        file_name: str,
        lb_info: Optional[Tuple[float, float, float]] = None,
        gt_annotations: Optional[Dict] = None,
        image_id: int = 0,
    ):
        """保存标注后的图片到指定目录

        Args:
            lb_info: letterbox 信息 (pad_left, pad_top, scale)，
                     用于将 640x640 上的坐标逆映射回原始分辨率
        """
        import cv2
        os.makedirs(save_dir, exist_ok=True)

        # 读取原始图片（BGR 格式，保持原始分辨率）
        img = cv2.imread(image_path)
        if img is None:
            return

        H_orig, W_orig = img.shape[:2]

        # 将检测框从 letterbox 640x640 坐标逆映射回原始分辨率
        scaled_dets = dets.copy()
        if len(scaled_dets) > 0:
            if lb_info is not None:
                pad_left, pad_top, scale = lb_info
                # 逆 letterbox: (coord - pad) / scale
                scaled_dets[:, 1] = (scaled_dets[:, 1] - pad_left) / scale
                scaled_dets[:, 2] = (scaled_dets[:, 2] - pad_top) / scale
                scaled_dets[:, 3] = (scaled_dets[:, 3] - pad_left) / scale
                scaled_dets[:, 4] = (scaled_dets[:, 4] - pad_top) / scale
            else:
                # Fallback: 简单 resize 缩放
                H_input, W_input = self.input_size
                scale_x = W_orig / W_input
                scale_y = H_orig / H_input
                scaled_dets[:, 1] *= scale_x
                scaled_dets[:, 2] *= scale_y
                scaled_dets[:, 3] *= scale_x
                scaled_dets[:, 4] *= scale_y

        # 绘制 GT 标注（虚线）
        if gt_annotations is not None:
            gt_anns = [a for a in gt_annotations.get('annotations', [])
                       if a['image_id'] == image_id]
            if gt_anns:
                categories = gt_annotations.get('categories')
                img = draw_gt_boxes(img, gt_anns, categories)

        # 绘制预测框
        categories = gt_annotations.get('categories') if gt_annotations else None
        img = draw_detections(img, scaled_dets, categories, conf_thres=self.conf_thres)

        out_path = os.path.join(save_dir, file_name)
        cv2.imwrite(out_path, img)


# ===========================================================================
# PR 曲线导出
# ===========================================================================

def export_pr_curve(
    evaluator: YOLOEvaluator,
    coco_path: str,
    output_path: str,
    max_images: int = 500,
    num_thresholds: int = 100,
):
    """
    导出 Precision-Recall 曲线数据到 JSON 文件。

    在不同置信度阈值下计算 Precision/Recall，生成 PR 曲线数据。
    """
    gt = load_coco_annotations(coco_path)
    images = gt['images'][:max_images]
    coco_dir = os.path.dirname(coco_path)
    image_dir = os.path.join(coco_dir, 'val2017')

    # 收集所有预测（不设置信度阈值）
    all_preds = []
    for idx, img_info in enumerate(images):
        file_name = img_info['file_name']
        image_path = os.path.join(image_dir, file_name)
        if not os.path.isfile(image_path):
            continue
        img, _ = load_image(image_path, evaluator.input_size)
        output = evaluator.infer_single(img)

        if output.ndim == 3:
            output = output[0]

        for det in output:
            all_preds.append({
                'image_id': img_info['id'],
                'box': det[:4].tolist(),
                'score': float(det[4]),
                'cls': int(det[5]),
            })

    # 按置信度排序
    all_preds.sort(key=lambda x: -x['score'])

    # 在不同阈值下计算 P/R
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    pr_points = []

    for thr in thresholds:
        # 过滤
        filtered = [p for p in all_preds if p['score'] >= thr]
        dets_by_image = defaultdict(list)
        for p in filtered:
            dets_by_image[p['image_id']].append(p)

        all_dets = []
        for img_info in images:
            preds = dets_by_image.get(img_info['id'], [])
            if preds:
                boxes = np.array([p['box'] for p in preds])
                scores = np.array([p['score'] for p in preds])
                labels = np.array([p['cls'] for p in preds])
            else:
                boxes = np.zeros((0, 4))
                scores = np.zeros(0)
                labels = np.zeros(0)
            all_dets.append({
                'image_id': img_info['id'],
                'boxes': boxes,
                'scores': scores,
                'labels': labels,
            })

        metrics = compute_coco_metrics(all_dets, gt, iou_thresholds=[0.5])
        pr_points.append({
            'threshold': float(thr),
            'mAP@0.5': metrics['mAP@0.5'],
            'num_detections': len(filtered),
        })

    with open(output_path, 'w') as f:
        json.dump({
            'model': evaluator.onnx_path,
            'pr_points': pr_points,
        }, f, indent=2)

    print(f"PR curve data exported to {output_path}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO Detection Model PR/mAP Evaluation"
    )
    parser.add_argument("--onnx", type=str, default="yolo26l.onnx",
                        help="ONNX 模型路径 (default: yolo26l.onnx)")
    parser.add_argument("--coco-path", type=str, default=None,
                        help="COCO 标注文件路径 (instances_val2017.json)")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="图片目录路径")
    parser.add_argument("--anno-file", type=str, default=None,
                        help="标注文件路径 (COCO JSON)")
    parser.add_argument("--input-size", type=int, nargs=2, default=[640, 640],
                        help="输入尺寸 H W (default: 640 640)")
    parser.add_argument("--conf-thres", type=float, default=0.25,
                        help="置信度阈值 (default: 0.25)")
    parser.add_argument("--iou-thres", type=float, default=0.45,
                        help="NMS IoU 阈值 (default: 0.45)")
    parser.add_argument("--max-images", type=int, default=-1,
                        help="最大评估图片数 (default: -1 全部)")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备 (cpu/xpu, default: auto)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16"],
                        help="推理精度 (default: float32)")
    parser.add_argument("--profile", action="store_true",
                        help="启用 op 级别 profile")
    parser.add_argument("--export-pr", type=str, default=None,
                        help="导出 PR 曲线数据到 JSON 文件")
    parser.add_argument("--output", type=str, default=None,
                        help="评估结果输出 JSON 文件")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="将标注后的图片保存到此目录")
    parser.add_argument("--draw-gt", action="store_true",
                        help="同时绘制 Ground Truth 标注框（需要 --anno-file 或 --coco-path）")
    return parser.parse_args()


def get_device(device_arg: str = None) -> str:
    if device_arg is not None:
        return device_arg
    if torch.xpu.is_available():
        return "xpu"
    return "cpu"


def main():
    args = parse_args()

    if not os.path.isfile(args.onnx):
        print(f"Error: ONNX model not found: {args.onnx}")
        sys.exit(1)

    device = get_device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    input_size = tuple(args.input_size)  # (H, W)

    print(f"{'='*60}")
    print(f"YOLO Detection Evaluation")
    print(f"{'='*60}")
    print(f"Model: {args.onnx}")
    print(f"Input size: {input_size[0]}x{input_size[1]}")
    print(f"Device: {device}, Dtype: {args.dtype}")
    print(f"Conf thres: {args.conf_thres}, IoU thres: {args.iou_thres}")

    evaluator = YOLOEvaluator(
        onnx_path=args.onnx,
        input_size=input_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=device,
        dtype=dtype,
    )

    # 评估
    if args.coco_path:
        metrics = evaluator.evaluate_coco(
            coco_path=args.coco_path,
            max_images=args.max_images,
            profile=args.profile,
            save_dir=args.save_dir,
            draw_gt=args.draw_gt,
        )
    elif args.image_dir:
        metrics = evaluator.evaluate_image_dir(
            image_dir=args.image_dir,
            anno_file=args.anno_file,
            max_images=args.max_images,
            profile=args.profile,
            save_dir=args.save_dir,
            draw_gt=args.draw_gt,
        )
    else:
        print("Error: Please specify --coco-path or --image-dir")
        sys.exit(1)

    # 打印结果
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Images: {metrics.get('num_images', 0)}")
    print(f"Total time: {metrics.get('total_time', 0):.2f}s")
    print(f"Avg time: {metrics.get('avg_time_ms', 0):.1f}ms/img")
    print(f"FPS: {metrics.get('fps', 0):.2f}")

    if 'mAP@0.5' in metrics:
        print(f"\nDetection Metrics:")
        print(f"  mAP@0.5:      {metrics['mAP@0.5']:.4f}")
        print(f"  mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
        if 'ap_per_iou' in metrics:
            print(f"\n  Per-IoU AP:")
            for k, v in metrics['ap_per_iou'].items():
                print(f"    {k}: {v:.4f}")

    # 导出结果
    if args.output:
        # 转换 numpy 类型为 Python 原生类型
        output_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (np.floating,)):
                output_metrics[k] = float(v)
            elif isinstance(v, (np.integer,)):
                output_metrics[k] = int(v)
            elif isinstance(v, dict):
                output_metrics[k] = {
                    kk: float(vv) if isinstance(vv, (np.floating,)) else vv
                    for kk, vv in v.items()
                }
            elif k == 'detections':
                continue  # 不导出原始检测结果
            else:
                output_metrics[k] = v
        with open(args.output, 'w') as f:
            json.dump(output_metrics, f, indent=2)
        print(f"\nResults exported to {args.output}")

    # 导出 PR 曲线
    if args.export_pr and args.coco_path:
        export_pr_curve(
            evaluator,
            coco_path=args.coco_path,
            output_path=args.export_pr,
            max_images=min(args.max_images if args.max_images > 0 else 500, 500),
        )

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
