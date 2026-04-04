"""
PyTorch checkpoint demo for DEIMv2 wholebody40 instance segmentation.
"""

import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageColor
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from engine.core import YAMLConfig, load_config
from engine.misc.mask_resize import resize_masks


AVERAGE_HEAD_WIDTH: float = 0.16 + 0.10
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
BODY_CLASS_ID = 0
DEFAULT_CONFIG = 'configs/deimv2/deimv2_dinov3_x_wholebody40_ins.yml'

BOX_COLORS = [
    ((216, 67, 21), 'Front'),
    ((255, 87, 34), 'Right-Front'),
    ((123, 31, 162), 'Right-Side'),
    ((255, 193, 7), 'Right-Back'),
    ((76, 175, 80), 'Back'),
    ((33, 150, 243), 'Left-Back'),
    ((156, 39, 176), 'Left-Side'),
    ((0, 188, 212), 'Left-Front'),
]

EDGES = [
    (21, 22), (21, 22),
    (21, 25),
    (22, 26), (22, 26),
    (26, 29), (26, 29),
    (29, 32), (29, 32),
    (22, 36), (22, 36),
    (25, 35),
    (35, 36), (35, 36),
    (36, 37), (36, 37),
    (37, 38), (37, 38),
    (38, 39), (38, 39),
]

OBJECT_CLASS_IDS = {0, 5, 6, 7, 16, 17, 18, 19, 20, 32, 33, 34, 39}
ATTRIBUTE_CLASS_IDS = {1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15}
KEYPOINT_CLASS_IDS = {21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 35, 36, 37, 38}
KEYPOINT_CLASS_ID_ORDER = (21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 35, 36, 37, 38)
KEYPOINT_DRAW_CLASS_IDS = KEYPOINT_CLASS_IDS
KEYPOINT_NMS_CLASS_IDS = KEYPOINT_CLASS_ID_ORDER
SKELETON_KEYPOINT_IDS = {21, 22, 25, 26, 29, 32, 35, 36, 37, 38, 39}

LEFT_SIDE_CLASS_IDS = {23, 27, 30, 33}
RIGHT_SIDE_CLASS_IDS = {24, 28, 31, 34}
SIDE_ATTR_CLASS_IDS = LEFT_SIDE_CLASS_IDS | RIGHT_SIDE_CLASS_IDS
SIDE_PARENT_TO_CHILDREN = {
    22: (23, 24),
    26: (27, 28),
    29: (30, 31),
    32: (33, 34),
}

LEFT_SIDE_COLOR = (0, 128, 0)
RIGHT_SIDE_COLOR = (255, 0, 255)


@dataclass(frozen=False)
class Box:
    classid: int
    score: float
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int
    source_idx: int = -1
    generation: int = -1
    gender: int = -1
    handedness: int = -1
    head_pose: int = -1
    is_used: bool = False
    person_id: int = -1
    track_id: int = -1


def make_instance_color(instance_idx: int) -> Tuple[int, int, int]:
    palette = [
        '#ff6b6b', '#4ecdc4', '#ffe66d', '#1a535c', '#ff9f1c',
        '#5f0f40', '#9a031e', '#fb8b24', '#0f4c5c', '#2ec4b6',
        '#3a86ff', '#8338ec', '#ff006e', '#8ac926', '#1982c4',
        '#6a4c93', '#e76f51', '#2a9d8f', '#e9c46a', '#264653',
    ]
    return ImageColor.getrgb(palette[instance_idx % len(palette)])


def list_image_paths(images_dir: Path) -> List[Path]:
    image_paths = [
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths, key=lambda path: path.name)


def load_checkpoint_state(resume_path: Path) -> Dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(resume_path, map_location='cpu', weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        checkpoint = torch.load(resume_path, map_location='cpu')
    if 'ema' in checkpoint and isinstance(checkpoint['ema'], dict) and 'module' in checkpoint['ema']:
        return checkpoint['ema']['module']
    if 'model' in checkpoint:
        return checkpoint['model']
    raise KeyError(f'Checkpoint {resume_path} does not contain `ema.module` or `model`.')


def tensor_state_only(state: Dict[str, object]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in state.items() if torch.is_tensor(v)}


def matched_tensor_state(current_state: Dict[str, object], loaded_state: Dict[str, object]):
    current_tensors = tensor_state_only(current_state)
    loaded_tensors = tensor_state_only(loaded_state)

    matched_state: Dict[str, torch.Tensor] = {}
    missing_keys: List[str] = []
    mismatched_keys: List[str] = []

    for key, value in current_tensors.items():
        if key not in loaded_tensors:
            missing_keys.append(key)
            continue
        if value.shape != loaded_tensors[key].shape:
            mismatched_keys.append(key)
            continue
        matched_state[key] = loaded_tensors[key]

    unexpected_keys = sorted(set(loaded_tensors.keys()) - set(current_tensors.keys()))
    return matched_state, missing_keys, mismatched_keys, unexpected_keys


def move_to_device(data, device: torch.device):
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    return data


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def is_onnx_model(model_path: Path) -> bool:
    return model_path.suffix.lower() == '.onnx'


def build_onnx_providers(device_arg: str | None, model_path: Path, inference_type: str):
    import onnxruntime as ort

    available_providers = set(ort.get_available_providers())
    requested = (device_arg or '').lower()
    inference_type = inference_type.lower()

    if requested.startswith('cuda'):
        if 'CUDAExecutionProvider' not in available_providers:
            raise RuntimeError('CUDAExecutionProvider is not available in this onnxruntime build.')
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']

    if requested == 'tensorrt':
        if 'TensorrtExecutionProvider' not in available_providers:
            raise RuntimeError('TensorrtExecutionProvider is not available in this onnxruntime build.')
        ep_type_params = {}
        if inference_type == 'fp16':
            ep_type_params = {
                'trt_fp16_enable': True,
            }
        elif inference_type == 'int8':
            ep_type_params = {
                'trt_fp16_enable': True,
                'trt_int8_enable': True,
                'trt_int8_calibration_table_name': 'calibration.flatbuffers',
            }
        else:
            raise ValueError(f'Unsupported inference type for TensorRT: {inference_type}')
        providers = [
            (
                'TensorrtExecutionProvider',
                {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': str(model_path.parent),
                    'trt_op_types_to_exclude': 'NonMaxSuppression,NonZero,RoiAlign',
                } | ep_type_params,
            )
        ]
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        return providers

    if requested and requested != 'cpu':
        raise ValueError(f'Unsupported ONNX device: {device_arg}. Use cpu, cuda, cuda:0, or tensorrt.')

    if device_arg is None and 'CUDAExecutionProvider' in available_providers:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


def build_transform(image_size: Sequence[int], normalize: bool) -> T.Compose:
    ops: List[object] = [
        T.Resize(tuple(image_size)),
        T.ToTensor(),
    ]
    if normalize:
        ops.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(ops)


def binary_mask_bbox(mask: np.ndarray) -> List[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def calculate_iou(base_obj: Box, target_obj: Box) -> float:
    inter_xmin = max(base_obj.x1, target_obj.x1)
    inter_ymin = max(base_obj.y1, target_obj.y1)
    inter_xmax = min(base_obj.x2, target_obj.x2)
    inter_ymax = min(base_obj.y2, target_obj.y2)
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    area1 = (base_obj.x2 - base_obj.x1) * (base_obj.y2 - base_obj.y1)
    area2 = (target_obj.x2 - target_obj.x1) * (target_obj.y2 - target_obj.y1)
    return inter_area / float(area1 + area2 - inter_area)


def find_most_relevant_obj(base_objs: List[Box], target_objs: List[Box]) -> None:
    for base_obj in base_objs:
        most_relevant_obj: Box | None = None
        best_score = 0.0
        best_iou = 0.0
        best_distance = float('inf')

        for target_obj in target_objs:
            distance = math.hypot(base_obj.cx - target_obj.cx, base_obj.cy - target_obj.cy)
            if not target_obj.is_used and distance <= 10.0:
                if target_obj.score >= best_score:
                    iou = calculate_iou(base_obj, target_obj)
                    if iou > best_iou:
                        most_relevant_obj = target_obj
                        best_iou = iou
                        best_distance = distance
                        best_score = target_obj.score
                    elif iou > 0.0 and iou == best_iou and distance < best_distance:
                        most_relevant_obj = target_obj
                        best_distance = distance
                        best_score = target_obj.score

        if most_relevant_obj is None:
            continue

        if most_relevant_obj.classid == 1:
            base_obj.generation = 0
        elif most_relevant_obj.classid == 2:
            base_obj.generation = 1
        elif most_relevant_obj.classid == 3:
            base_obj.gender = 0
        elif most_relevant_obj.classid == 4:
            base_obj.gender = 1
        elif most_relevant_obj.classid == 8:
            base_obj.head_pose = 0
        elif most_relevant_obj.classid == 9:
            base_obj.head_pose = 1
        elif most_relevant_obj.classid == 10:
            base_obj.head_pose = 2
        elif most_relevant_obj.classid == 11:
            base_obj.head_pose = 3
        elif most_relevant_obj.classid == 12:
            base_obj.head_pose = 4
        elif most_relevant_obj.classid == 13:
            base_obj.head_pose = 5
        elif most_relevant_obj.classid == 14:
            base_obj.head_pose = 6
        elif most_relevant_obj.classid == 15:
            base_obj.head_pose = 7
        elif most_relevant_obj.classid in LEFT_SIDE_CLASS_IDS:
            base_obj.handedness = 0
        elif most_relevant_obj.classid in RIGHT_SIDE_CLASS_IDS:
            base_obj.handedness = 1

        most_relevant_obj.is_used = True


def nms(target_objs: List[Box], iou_threshold: float) -> List[Box]:
    filtered_objs: List[Box] = []
    sorted_objs = sorted(target_objs, key=lambda box: box.score, reverse=True)

    while sorted_objs:
        current_box = sorted_objs.pop(0)
        if current_box.is_used:
            continue

        filtered_objs.append(current_box)
        current_box.is_used = True

        remaining_boxes = []
        for box in sorted_objs:
            if not box.is_used:
                iou_value = calculate_iou(current_box, box)
                if iou_value >= iou_threshold:
                    box.is_used = True
                else:
                    remaining_boxes.append(box)
        sorted_objs = remaining_boxes

    return filtered_objs


def build_result_boxes(
    result: Dict[str, torch.Tensor],
    image_width: int,
    image_height: int,
    object_score_threshold: float,
    attribute_score_threshold: float,
    keypoint_threshold: float,
    disable_generation_identification_mode: bool,
    disable_gender_identification_mode: bool,
    disable_left_and_right_hand_identification_mode: bool,
    disable_headpose_identification_mode: bool,
) -> List[Box]:
    labels = result['labels'].detach().cpu()
    scores = result['scores'].detach().cpu()
    boxes = result['boxes'].detach().cpu()

    result_boxes: List[Box] = []
    box_score_threshold = min(object_score_threshold, attribute_score_threshold, keypoint_threshold)

    for idx in range(len(labels)):
        score = float(scores[idx].item())
        if score <= box_score_threshold:
            continue

        classid = int(labels[idx].item())
        x1_f, y1_f, x2_f, y2_f = [float(v) for v in boxes[idx].tolist()]
        x1 = max(0, min(int(round(x1_f)), image_width - 1))
        y1 = max(0, min(int(round(y1_f)), image_height - 1))
        x2 = max(0, min(int(round(x2_f)), image_width - 1))
        y2 = max(0, min(int(round(y2_f)), image_height - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        result_boxes.append(
            Box(
                classid=classid,
                score=score,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                cx=(x1 + x2) // 2,
                cy=(y1 + y2) // 2,
                source_idx=idx,
            )
        )

    result_boxes = [
        box for box in result_boxes
        if (box.classid in OBJECT_CLASS_IDS and box.score >= object_score_threshold) or box.classid not in OBJECT_CLASS_IDS
    ]
    result_boxes = [
        box for box in result_boxes
        if (box.classid in ATTRIBUTE_CLASS_IDS and box.score >= attribute_score_threshold) or box.classid not in ATTRIBUTE_CLASS_IDS
    ]
    result_boxes = [
        box for box in result_boxes
        if (box.classid in KEYPOINT_CLASS_IDS and box.score >= keypoint_threshold) or box.classid not in KEYPOINT_CLASS_IDS
    ]

    if not disable_generation_identification_mode:
        body_boxes = [box for box in result_boxes if box.classid == 0]
        generation_boxes = [box for box in result_boxes if box.classid in [1, 2]]
        find_most_relevant_obj(body_boxes, generation_boxes)
    result_boxes = [box for box in result_boxes if box.classid not in [1, 2]]

    if not disable_gender_identification_mode:
        body_boxes = [box for box in result_boxes if box.classid == 0]
        gender_boxes = [box for box in result_boxes if box.classid in [3, 4]]
        find_most_relevant_obj(body_boxes, gender_boxes)
    result_boxes = [box for box in result_boxes if box.classid not in [3, 4]]

    if not disable_headpose_identification_mode:
        head_boxes = [box for box in result_boxes if box.classid == 7]
        headpose_boxes = [box for box in result_boxes if box.classid in [8, 9, 10, 11, 12, 13, 14, 15]]
        find_most_relevant_obj(head_boxes, headpose_boxes)
    result_boxes = [box for box in result_boxes if box.classid not in [8, 9, 10, 11, 12, 13, 14, 15]]

    if not disable_left_and_right_hand_identification_mode:
        for parent_classid, child_classids in SIDE_PARENT_TO_CHILDREN.items():
            parent_boxes = [box for box in result_boxes if box.classid == parent_classid]
            side_boxes = [box for box in result_boxes if box.classid in child_classids]
            find_most_relevant_obj(parent_boxes, side_boxes)
    result_boxes = [box for box in result_boxes if box.classid not in SIDE_ATTR_CLASS_IDS]

    for target_classid in KEYPOINT_NMS_CLASS_IDS:
        keypoint_boxes = [box for box in result_boxes if box.classid == target_classid]
        filtered_keypoint_boxes = nms(keypoint_boxes, iou_threshold=0.20)
        result_boxes = [box for box in result_boxes if box.classid != target_classid]
        result_boxes.extend(filtered_keypoint_boxes)

    return result_boxes


def prepare_prediction_payload(
    boxes: List[Box],
    result: Dict[str, torch.Tensor],
    mask_threshold: float,
    enable_masks: bool,
    enable_contours: bool,
) -> List[Dict[str, object]]:
    masks = result.get('masks') if enable_masks else None
    if masks is not None and torch.is_tensor(masks):
        masks = masks.detach().cpu()
    contours = result.get('contours') if enable_contours else None
    if contours is not None and torch.is_tensor(contours):
        contours = contours.detach().cpu()

    records: List[Dict[str, object]] = []
    for box in boxes:
        record: Dict[str, object] = {
            'label': box.classid,
            'score': box.score,
            'box_xyxy': [float(box.x1), float(box.y1), float(box.x2), float(box.y2)],
            'generation': box.generation,
            'gender': box.gender,
            'handedness': box.handedness,
            'head_pose': box.head_pose,
        }

        if masks is not None and box.classid == BODY_CLASS_ID and box.source_idx >= 0:
            mask_probs = lookup_mask_probs(masks, box.source_idx)
            if mask_probs is not None:
                binary_mask = mask_probs >= mask_threshold
                mask_bbox = binary_mask_bbox(binary_mask)
                if mask_bbox is not None:
                    record['mask_area'] = int(binary_mask.sum())
                    record['mask_bbox'] = mask_bbox

        if contours is not None and box.classid == BODY_CLASS_ID and box.source_idx >= 0:
            contour_probs = lookup_mask_probs(contours, box.source_idx)
            if contour_probs is not None:
                binary_contour = contour_probs >= mask_threshold
                contour_bbox = binary_mask_bbox(binary_contour)
                if contour_bbox is not None:
                    record['contour_area'] = int(binary_contour.sum())
                    record['contour_bbox'] = contour_bbox

        records.append(record)
    return records


def lookup_mask_probs(
    mask_store: torch.Tensor | Dict[int, torch.Tensor] | None,
    source_idx: int,
) -> np.ndarray | None:
    if mask_store is None or source_idx < 0:
        return None

    if isinstance(mask_store, dict):
        mask_value = mask_store.get(source_idx)
    else:
        if source_idx >= len(mask_store):
            return None
        mask_value = mask_store[source_idx]

    if mask_value is None:
        return None

    if torch.is_tensor(mask_value):
        mask_array = mask_value.detach().cpu().numpy()
    else:
        mask_array = np.asarray(mask_value)

    if mask_array.ndim == 3:
        return mask_array[0]
    return mask_array


def overlay_body_masks(
    image: Image.Image,
    result: Dict[str, torch.Tensor],
    boxes: List[Box],
    mask_threshold: float,
    mask_alpha: int,
    disable_render_classids: set[int],
) -> Image.Image:
    masks = result.get('masks')
    if masks is None or BODY_CLASS_ID in disable_render_classids:
        return image

    if torch.is_tensor(masks):
        masks = masks.detach().cpu()
    overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    body_instance_idx = 0

    for box in boxes:
        if box.classid != BODY_CLASS_ID or box.source_idx < 0:
            continue
        mask_probs = lookup_mask_probs(masks, box.source_idx)
        if mask_probs is None:
            continue
        binary_mask = mask_probs >= mask_threshold
        if not binary_mask.any():
            continue
        instance_color = make_instance_color(body_instance_idx)
        overlay[binary_mask] = np.array([instance_color[0], instance_color[1], instance_color[2], mask_alpha], dtype=np.uint8)
        body_instance_idx += 1

    if overlay[..., 3].max() == 0:
        return image

    base = image.convert('RGBA')
    mask_image = Image.fromarray(overlay)
    return Image.alpha_composite(base, mask_image).convert('RGB')


def overlay_body_contours(
    image: Image.Image,
    result: Dict[str, torch.Tensor],
    boxes: List[Box],
    contour_threshold: float,
    disable_render_classids: set[int],
) -> Image.Image:
    contours = result.get('contours')
    if contours is None or BODY_CLASS_ID in disable_render_classids:
        return image

    if torch.is_tensor(contours):
        contours = contours.detach().cpu()
    rendered = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    body_instance_idx = 0

    for box in boxes:
        if box.classid != BODY_CLASS_ID or box.source_idx < 0:
            continue
        contour_probs = lookup_mask_probs(contours, box.source_idx)
        if contour_probs is None:
            continue
        binary_contour = (contour_probs >= contour_threshold).astype(np.uint8)
        if not binary_contour.any():
            continue

        contour_segments, _ = cv2.findContours(binary_contour, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not contour_segments:
            continue

        instance_color = make_instance_color(body_instance_idx)
        cv2.drawContours(
            rendered,
            contour_segments,
            contourIdx=-1,
            color=(instance_color[2], instance_color[1], instance_color[0]),
            thickness=1,
        )
        body_instance_idx += 1

    return Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))


def draw_dashed_line(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
) -> None:
    dist = math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
    dashes = max(1, int(dist / dash_length))
    for idx in range(dashes):
        start = (
            int(pt1[0] + (pt2[0] - pt1[0]) * idx / dashes),
            int(pt1[1] + (pt2[1] - pt1[1]) * idx / dashes),
        )
        end = (
            int(pt1[0] + (pt2[0] - pt1[0]) * (idx + 0.5) / dashes),
            int(pt1[1] + (pt2[1] - pt1[1]) * (idx + 0.5) / dashes),
        )
        cv2.line(image, start, end, color, thickness)


def draw_dashed_rectangle(
    image: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
) -> None:
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])
    draw_dashed_line(image, top_left, top_right, color, thickness, dash_length)
    draw_dashed_line(image, top_right, bottom_right, color, thickness, dash_length)
    draw_dashed_line(image, bottom_right, bottom_left, color, thickness, dash_length)
    draw_dashed_line(image, bottom_left, top_left, color, thickness, dash_length)


def draw_skeleton(
    image: np.ndarray,
    boxes: List[Box],
    color: Tuple[int, int, int] = (0, 255, 255),
    max_dist_threshold: float = 500.0,
) -> None:
    person_boxes = [box for box in boxes if box.classid == 0]
    for person_id, person_box in enumerate(person_boxes):
        person_box.person_id = person_id

    for box in boxes:
        if box.classid in SKELETON_KEYPOINT_IDS:
            box.person_id = -1
            for person_box in person_boxes:
                if person_box.x1 <= box.cx <= person_box.x2 and person_box.y1 <= box.cy <= person_box.y2:
                    box.person_id = person_box.person_id
                    break

    classid_to_boxes: Dict[int, List[Box]] = {}
    for box in boxes:
        classid_to_boxes.setdefault(box.classid, []).append(box)

    edge_counts = Counter(EDGES)
    lines_to_draw: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    for (parent_id, child_id), repeat_count in edge_counts.items():
        parent_list = classid_to_boxes.get(parent_id, [])
        child_list = classid_to_boxes.get(child_id, [])
        if not parent_list or not child_list:
            continue

        parent_capacity = [repeat_count] * len(parent_list)
        child_used = [False] * len(child_list)
        pair_candidates: List[Tuple[float, int, int]] = []

        for parent_idx, parent_box in enumerate(parent_list):
            for child_idx, child_box in enumerate(child_list):
                if parent_box.person_id == child_box.person_id and parent_box.person_id is not None:
                    dist = math.hypot(parent_box.cx - child_box.cx, parent_box.cy - child_box.cy)
                    if dist <= max_dist_threshold:
                        pair_candidates.append((dist, parent_idx, child_idx))

        pair_candidates.sort(key=lambda item: item[0])

        for _, parent_idx, child_idx in pair_candidates:
            if parent_capacity[parent_idx] > 0 and not child_used[child_idx]:
                parent_box = parent_list[parent_idx]
                child_box = child_list[child_idx]
                lines_to_draw.append(((parent_box.cx, parent_box.cy), (child_box.cx, child_box.cy)))
                parent_capacity[parent_idx] -= 1
                child_used[child_idx] = True

    for point_a, point_b in lines_to_draw:
        cv2.line(image, point_a, point_b, color, thickness=2)


def draw_text_with_outline(
    image: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float = 0.7,
    outline_color: Tuple[int, int, int] = (255, 255, 255),
    outline_thickness: int = 2,
    text_thickness: int = 1,
) -> None:
    if not text:
        return
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness, cv2.LINE_AA)


def apply_face_mosaic(image: np.ndarray, box: Box) -> None:
    crop = image[box.y1:box.y2, box.x1:box.x2, :]
    if crop.size == 0:
        return
    width = max(1, abs(box.x2 - box.x1))
    height = max(1, abs(box.y2 - box.y1))
    small_box = cv2.resize(crop, (3, 3))
    normal_box = cv2.resize(small_box, (width, height))
    if normal_box.shape[0] != height or normal_box.shape[1] != width:
        normal_box = cv2.resize(small_box, (width, height))
    image[box.y1:box.y2, box.x1:box.x2, :] = normal_box


def get_render_color(
    box: Box,
    disable_gender_identification_mode: bool,
    disable_left_and_right_hand_identification_mode: bool,
    disable_headpose_identification_mode: bool,
) -> Tuple[int, int, int]:
    classid = box.classid
    color = (255, 255, 255)

    if classid == 0:
        if not disable_gender_identification_mode:
            if box.gender == 0:
                color = (255, 0, 0)
            elif box.gender == 1:
                color = (139, 116, 225)
            else:
                color = (0, 200, 255)
        else:
            color = (0, 200, 255)
    elif classid == 5:
        color = (0, 200, 255)
    elif classid == 6:
        color = (83, 36, 179)
    elif classid == 7:
        if not disable_headpose_identification_mode:
            color = BOX_COLORS[box.head_pose][0] if box.head_pose != -1 else (216, 67, 21)
        else:
            color = (0, 0, 255)
    elif classid == 16:
        color = (0, 200, 255)
    elif classid == 17:
        color = (255, 0, 0)
    elif classid == 18:
        color = (0, 255, 0)
    elif classid == 19:
        color = (0, 0, 255)
    elif classid == 20:
        color = (203, 192, 255)
    elif classid == 21:
        color = (0, 0, 255)
    elif classid == 22:
        color = (255, 0, 0)
        if not disable_left_and_right_hand_identification_mode:
            if box.handedness == 0:
                color = LEFT_SIDE_COLOR
            elif box.handedness == 1:
                color = RIGHT_SIDE_COLOR
    elif classid == 23:
        color = LEFT_SIDE_COLOR
    elif classid == 24:
        color = RIGHT_SIDE_COLOR
    elif classid == 25:
        color = (252, 189, 107)
    elif classid == 26:
        color = (0, 255, 0)
        if not disable_left_and_right_hand_identification_mode:
            if box.handedness == 0:
                color = LEFT_SIDE_COLOR
            elif box.handedness == 1:
                color = RIGHT_SIDE_COLOR
    elif classid == 27:
        color = LEFT_SIDE_COLOR
    elif classid == 28:
        color = RIGHT_SIDE_COLOR
    elif classid == 29:
        color = (0, 0, 255)
        if not disable_left_and_right_hand_identification_mode:
            if box.handedness == 0:
                color = LEFT_SIDE_COLOR
            elif box.handedness == 1:
                color = RIGHT_SIDE_COLOR
    elif classid == 30:
        color = LEFT_SIDE_COLOR
    elif classid == 31:
        color = RIGHT_SIDE_COLOR
    elif classid == 32:
        if not disable_left_and_right_hand_identification_mode:
            if box.handedness == 0:
                color = LEFT_SIDE_COLOR
            elif box.handedness == 1:
                color = RIGHT_SIDE_COLOR
            else:
                color = (0, 255, 0)
        else:
            color = (0, 255, 0)
    elif classid == 33:
        color = LEFT_SIDE_COLOR
    elif classid == 34:
        color = RIGHT_SIDE_COLOR
    elif classid == 35:
        color = (0, 0, 255)
    elif classid == 36:
        color = (255, 0, 0)
    elif classid == 37:
        color = (0, 0, 255)
    elif classid == 38:
        color = (255, 0, 0)
    elif classid == 39:
        color = (250, 0, 136)

    return color


def draw_detections(
    image: Image.Image,
    boxes: List[Box],
    disable_render_classids: set[int],
    keypoint_drawing_mode: str,
    enable_bone_drawing_mode: bool,
    enable_face_mosaic: bool,
    disable_generation_identification_mode: bool,
    disable_gender_identification_mode: bool,
    disable_left_and_right_hand_identification_mode: bool,
    disable_headpose_identification_mode: bool,
    bounding_box_line_width: int,
    enable_head_distance_measurement: bool,
    camera_horizontal_fov: int,
) -> Image.Image:
    debug_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    debug_image_h, debug_image_w = debug_image.shape[:2]
    white_line_width = bounding_box_line_width
    colored_line_width = white_line_width - 1

    for box in boxes:
        classid = box.classid
        if classid in disable_render_classids:
            continue

        color = get_render_color(
            box,
            disable_gender_identification_mode=disable_gender_identification_mode,
            disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
            disable_headpose_identification_mode=disable_headpose_identification_mode,
        )

        if (
            (classid == 0 and not disable_gender_identification_mode)
            or (classid == 7 and not disable_headpose_identification_mode)
            or (classid == 32 and not disable_left_and_right_hand_identification_mode)
            or classid == 16
            or classid in KEYPOINT_DRAW_CLASS_IDS
        ):
            if classid == 0:
                if box.gender == -1:
                    draw_dashed_rectangle(
                        image=debug_image,
                        top_left=(box.x1, box.y1),
                        bottom_right=(box.x2, box.y2),
                        color=color,
                        thickness=2,
                        dash_length=10,
                    )
                else:
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255, 255, 255), white_line_width)
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)
            elif classid == 7:
                if box.head_pose == -1:
                    draw_dashed_rectangle(
                        image=debug_image,
                        top_left=(box.x1, box.y1),
                        bottom_right=(box.x2, box.y2),
                        color=color,
                        thickness=2,
                        dash_length=10,
                    )
                else:
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255, 255, 255), white_line_width)
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)
            elif classid == 16:
                if enable_face_mosaic:
                    apply_face_mosaic(debug_image, box)
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255, 255, 255), white_line_width)
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)
            elif classid == 32:
                if box.handedness == -1:
                    draw_dashed_rectangle(
                        image=debug_image,
                        top_left=(box.x1, box.y1),
                        bottom_right=(box.x2, box.y2),
                        color=color,
                        thickness=2,
                        dash_length=10,
                    )
                else:
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255, 255, 255), white_line_width)
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)
            elif classid in KEYPOINT_DRAW_CLASS_IDS:
                if keypoint_drawing_mode in ['dot', 'both']:
                    cv2.circle(debug_image, (box.cx, box.cy), 4, (255, 255, 255), -1)
                    cv2.circle(debug_image, (box.cx, box.cy), 3, color, -1)
                if keypoint_drawing_mode in ['box', 'both']:
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255, 255, 255), 2)
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, 1)
        else:
            cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255, 255, 255), white_line_width)
            cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

        generation_txt = ''
        if box.generation == 0:
            generation_txt = 'Adult'
        elif box.generation == 1:
            generation_txt = 'Child'

        gender_txt = ''
        if box.gender == 0:
            gender_txt = 'M'
        elif box.gender == 1:
            gender_txt = 'F'

        attr_txt = f'{generation_txt}({gender_txt})' if gender_txt else generation_txt
        headpose_txt = BOX_COLORS[box.head_pose][1] if box.head_pose != -1 else ''
        attr_txt = f'{attr_txt} {headpose_txt}'.strip() if headpose_txt else attr_txt

        text_org = (
            box.x1 if box.x1 + 50 < debug_image_w else debug_image_w - 50,
            box.y1 - 10 if box.y1 - 25 > 0 else 20,
        )
        draw_text_with_outline(debug_image, attr_txt, text_org, color)

        handedness_txt = ''
        if classid in LEFT_SIDE_CLASS_IDS:
            handedness_txt = 'L'
        elif classid in RIGHT_SIDE_CLASS_IDS:
            handedness_txt = 'R'
        elif box.handedness == 0:
            handedness_txt = 'L'
        elif box.handedness == 1:
            handedness_txt = 'R'
        draw_text_with_outline(debug_image, handedness_txt, text_org, color)

        if enable_head_distance_measurement and classid == 7 and abs(box.x2 - box.x1) > 0:
            if camera_horizontal_fov > 90:
                focal_length = debug_image_w / (camera_horizontal_fov * (math.pi / 180))
            else:
                focal_length = debug_image_w / (2 * math.tan((camera_horizontal_fov / 2) * (math.pi / 180)))
            distance = (AVERAGE_HEAD_WIDTH * focal_length) / abs(box.x2 - box.x1)
            distance_org = (
                box.x1 + 5 if box.x1 < debug_image_w else debug_image_w - 50,
                box.y1 + 20 if box.y1 - 5 > 0 else 20,
            )
            cv2.putText(debug_image, f'{distance:.3f} m', distance_org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(debug_image, f'{distance:.3f} m', distance_org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 1, cv2.LINE_AA)

    if enable_bone_drawing_mode:
        draw_skeleton(image=debug_image, boxes=boxes, color=(0, 255, 255), max_dist_threshold=300)

    return Image.fromarray(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))


class InferenceModel(nn.Module):
    def __init__(
        self,
        cfg: YAMLConfig,
        state_dict: Dict[str, torch.Tensor],
        device: torch.device,
        mask_resize_origin: str = 'topleft',
    ):
        super().__init__()
        matched_state, missing_keys, mismatched_keys, unexpected_keys = matched_tensor_state(
            cfg.model.state_dict(),
            state_dict,
        )
        load_info = cfg.model.load_state_dict(matched_state, strict=False)
        if missing_keys or mismatched_keys or unexpected_keys:
            print(
                'Partially loaded checkpoint for inference: '
                f'matched={len(matched_state)}, '
                f'missing={len(missing_keys)}, '
                f'shape_mismatch={len(mismatched_keys)}, '
                f'unexpected={len(unexpected_keys)}'
            )
            if missing_keys:
                print(f'  Missing keys (first 10): {missing_keys[:10]}')
            if mismatched_keys:
                print(f'  Shape mismatches (first 10): {mismatched_keys[:10]}')
            if unexpected_keys:
                print(f'  Unexpected keys (first 10): {unexpected_keys[:10]}')
            if load_info.missing_keys:
                print(f'  load_state_dict missing keys (first 10): {load_info.missing_keys[:10]}')
            if load_info.unexpected_keys:
                print(f'  load_state_dict unexpected keys (first 10): {load_info.unexpected_keys[:10]}')
        self.model = cfg.model.eval().to(device)
        self.postprocessor = cfg.postprocessor.eval()
        self.postprocessor.mask_resize_origin = mask_resize_origin
        self.device = device

    @torch.inference_mode()
    def forward(
        self,
        image_tensor: torch.Tensor,
        orig_target_sizes: torch.Tensor,
        return_masks: bool=False,
        return_contours: bool=False,
    ):
        outputs = self.model(
            image_tensor,
            return_masks=return_masks,
            return_contours=return_contours,
        )
        outputs = move_to_device(outputs, torch.device('cpu'))
        orig_target_sizes = orig_target_sizes.to('cpu')
        return self.postprocessor(
            outputs,
            orig_target_sizes,
            return_masks=return_masks,
            return_contours=return_contours,
        )


class OnnxInferenceModel:
    def __init__(
        self,
        model_path: Path,
        device_arg: str | None,
        inference_type: str,
        mask_resize_origin: str = 'topleft',
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError('onnxruntime is required for ONNX inference.') from exc

        providers = build_onnx_providers(device_arg, model_path, inference_type)
        session_options = ort.SessionOptions()
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=providers,
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.mask_resize_origin = mask_resize_origin
        self.providers = self.session.get_providers()
        image_input = self.session.get_inputs()[0]
        image_shape = image_input.shape
        self.image_size = tuple(int(v) for v in image_shape[2:4]) if len(image_shape) >= 4 and all(isinstance(v, int) for v in image_shape[2:4]) else None

    def _decode_label_xyxy_score(
        self,
        label_xyxy_score: np.ndarray,
        orig_target_sizes: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        batch_results: List[Dict[str, torch.Tensor]] = []
        for batch_idx in range(label_xyxy_score.shape[0]):
            batch_pred = label_xyxy_score[batch_idx]
            boxes = torch.from_numpy(batch_pred[:, 1:5].astype(np.float32, copy=False))
            if 'orig_target_sizes' not in self.input_names:
                orig_w = float(orig_target_sizes[batch_idx, 0].item())
                orig_h = float(orig_target_sizes[batch_idx, 1].item())
                boxes[:, 0::2] *= orig_w
                boxes[:, 1::2] *= orig_h
            batch_results.append(
                {
                    'labels': torch.from_numpy(batch_pred[:, 0].astype(np.int64, copy=False)),
                    'boxes': boxes,
                    'scores': torch.from_numpy(batch_pred[:, 5].astype(np.float32, copy=False)),
                }
            )
        return batch_results

    def _resize_mask_batch(
        self,
        masks: np.ndarray,
        orig_target_sizes: torch.Tensor,
    ) -> List[torch.Tensor]:
        resized_batches: List[torch.Tensor] = []
        for batch_idx in range(masks.shape[0]):
            batch_masks = torch.from_numpy(masks[batch_idx]).to(dtype=torch.float32)
            batch_masks = resize_masks(
                batch_masks.unsqueeze(1),
                size=tuple(int(v) for v in orig_target_sizes[batch_idx, [1, 0]].tolist()),
                mode='bilinear',
                origin=self.mask_resize_origin,
            )
            resized_batches.append(batch_masks)
        return resized_batches

    def _resize_selected_mask_map(
        self,
        masks: np.ndarray,
        orig_target_size: torch.Tensor,
        selected_indices: Sequence[int],
    ) -> Dict[int, torch.Tensor]:
        unique_indices = sorted({int(idx) for idx in selected_indices if idx >= 0})
        if not unique_indices:
            return {}

        batch_masks = torch.from_numpy(masks[unique_indices]).to(dtype=torch.float32)
        resized_masks = resize_masks(
            batch_masks.unsqueeze(1),
            size=tuple(int(v) for v in orig_target_size[[1, 0]].tolist()),
            mode='bilinear',
            origin=self.mask_resize_origin,
        )
        return {
            source_idx: resized_masks[pos]
            for pos, source_idx in enumerate(unique_indices)
        }

    @torch.inference_mode()
    def __call__(
        self,
        image_tensor: torch.Tensor,
        orig_target_sizes: torch.Tensor,
        return_masks: bool = False,
        return_contours: bool = False,
    ):
        input_feed = {'images': image_tensor.detach().cpu().numpy().astype(np.float32, copy=False)}
        if 'orig_target_sizes' in self.input_names:
            input_feed['orig_target_sizes'] = orig_target_sizes.detach().cpu().numpy().astype(np.float32, copy=False)

        requested_outputs = ['label_xyxy_score']
        if return_masks:
            requested_outputs.append('masks')
        if return_contours:
            requested_outputs.append('contours')
        output_values = self.session.run(requested_outputs, input_feed)
        outputs = dict(zip(requested_outputs, output_values))

        if 'label_xyxy_score' not in outputs:
            raise KeyError('ONNX model output `label_xyxy_score` is required.')

        results = self._decode_label_xyxy_score(outputs['label_xyxy_score'], orig_target_sizes)

        if return_masks:
            if 'masks' not in outputs:
                raise RuntimeError('ONNX model does not provide `masks`, but --enable-masks was requested.')
            for batch_idx, result in enumerate(results):
                result['_onnx_masks'] = outputs['masks'][batch_idx]

        if return_contours:
            if 'contours' not in outputs:
                raise RuntimeError('ONNX model does not provide `contours`, but --enable-contours was requested.')
            for batch_idx, result in enumerate(results):
                result['_onnx_contours'] = outputs['contours'][batch_idx]

        return results


def save_predictions_json(output_dir: Path, image_path: Path, records: List[Dict[str, object]]) -> None:
    pred_dir = output_dir / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'image': image_path.name,
        'predictions': records,
    }
    with (pred_dir / f'{image_path.stem}.json').open('w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def process_images(args) -> None:
    config_path = Path(args.config)
    resume_path = Path(args.resume)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    use_onnx = is_onnx_model(resume_path)

    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    if not resume_path.exists():
        raise FileNotFoundError(f'Model file not found: {resume_path}')
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f'Image directory not found: {images_dir}')

    image_paths = list_image_paths(images_dir)
    if not image_paths:
        raise FileNotFoundError(f'No image files found in {images_dir}')

    output_dir.mkdir(parents=True, exist_ok=True)

    if use_onnx:
        yaml_cfg = load_config(str(config_path))
        model = OnnxInferenceModel(
            resume_path,
            args.device,
            args.inference_type,
            mask_resize_origin=args.mask_resize_origin,
        )
        image_size = model.image_size or tuple(yaml_cfg['eval_spatial_size'])
        normalize = bool(yaml_cfg.get('DINOv3STAs', False))
    else:
        if (args.device or '').lower() == 'tensorrt':
            raise ValueError('TensorRT inference is only supported when --resume points to an ONNX model.')
        cfg = YAMLConfig(str(config_path), resume=str(resume_path))
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        if 'DINOv3STAs' in cfg.yaml_cfg:
            cfg.yaml_cfg['DINOv3STAs']['weights_path'] = None

        state_dict = load_checkpoint_state(resume_path)
        device = resolve_device(args.device)
        model = InferenceModel(cfg, state_dict, device, mask_resize_origin=args.mask_resize_origin)
        image_size = cfg.yaml_cfg['eval_spatial_size']
        normalize = bool(cfg.yaml_cfg.get('DINOv3STAs', False))

    transform = build_transform(image_size, normalize)

    object_score_threshold = args.object_score_threshold if args.object_score_threshold is not None else args.score_threshold
    attribute_score_threshold = args.attribute_score_threshold if args.attribute_score_threshold is not None else args.score_threshold
    keypoint_threshold = args.keypoint_threshold if args.keypoint_threshold is not None else args.score_threshold
    disable_render_classids = set(args.disable_render_classids)

    print(f'Processing {len(image_paths)} images from {images_dir}')
    print(f'Using model: {resume_path}')
    if use_onnx:
        print(f'ONNX providers: {model.providers}')
    else:
        print(f'Device: {device}')
    print(f'Output directory: {output_dir}')
    print(f'Mask resize origin: {args.mask_resize_origin}')
    print(f'Enable masks: {args.enable_masks}')
    print(f'Enable contours: {args.enable_contours}')

    for image_path in tqdm(
        image_paths,
        desc='Processing images',
        dynamic_ncols=True,
        unit='image',
    ):
        image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = image.size
        orig_target_sizes = torch.tensor([[orig_w, orig_h]], dtype=torch.float32)
        image_tensor = transform(image).unsqueeze(0)
        if not use_onnx:
            image_tensor = image_tensor.to(device)

        results = model(
            image_tensor,
            orig_target_sizes,
            return_masks=args.enable_masks,
            return_contours=args.enable_contours,
        )
        result = results[0]

        boxes = build_result_boxes(
            result=result,
            image_width=orig_w,
            image_height=orig_h,
            object_score_threshold=object_score_threshold,
            attribute_score_threshold=attribute_score_threshold,
            keypoint_threshold=keypoint_threshold,
            disable_generation_identification_mode=args.disable_generation_identification_mode,
            disable_gender_identification_mode=args.disable_gender_identification_mode,
            disable_left_and_right_hand_identification_mode=args.disable_left_and_right_hand_identification_mode,
            disable_headpose_identification_mode=args.disable_headpose_identification_mode,
        )

        body_source_indices = [box.source_idx for box in boxes if box.classid == BODY_CLASS_ID and box.source_idx >= 0]
        if use_onnx and args.enable_masks:
            raw_masks = result.pop('_onnx_masks', None)
            if raw_masks is not None:
                result['masks'] = model._resize_selected_mask_map(
                    raw_masks,
                    orig_target_sizes[0],
                    body_source_indices,
                )
        if use_onnx and args.enable_contours:
            raw_contours = result.pop('_onnx_contours', None)
            if raw_contours is not None:
                result['contours'] = model._resize_selected_mask_map(
                    raw_contours,
                    orig_target_sizes[0],
                    body_source_indices,
                )

        rendered = overlay_body_masks(
            image=image.copy(),
            result=result,
            boxes=boxes,
            mask_threshold=args.mask_threshold,
            mask_alpha=args.mask_alpha,
            disable_render_classids=disable_render_classids,
        )
        rendered = overlay_body_contours(
            image=rendered,
            result=result,
            boxes=boxes,
            contour_threshold=args.mask_threshold,
            disable_render_classids=disable_render_classids,
        )
        rendered = draw_detections(
            image=rendered,
            boxes=boxes,
            disable_render_classids=disable_render_classids,
            keypoint_drawing_mode=args.keypoint_drawing_mode,
            enable_bone_drawing_mode=args.enable_bone_drawing_mode,
            enable_face_mosaic=args.enable_face_mosaic,
            disable_generation_identification_mode=args.disable_generation_identification_mode,
            disable_gender_identification_mode=args.disable_gender_identification_mode,
            disable_left_and_right_hand_identification_mode=args.disable_left_and_right_hand_identification_mode,
            disable_headpose_identification_mode=args.disable_headpose_identification_mode,
            bounding_box_line_width=args.bounding_box_line_width,
            enable_head_distance_measurement=not args.disable_head_distance_measurement,
            camera_horizontal_fov=args.camera_horizontal_fov,
        )
        rendered.save(output_dir / image_path.name)

        if args.save_raw_predictions:
            records = prepare_prediction_payload(
                boxes=boxes,
                result=result,
                mask_threshold=args.mask_threshold,
                enable_masks=args.enable_masks,
                enable_contours=args.enable_contours,
            )
            save_predictions_json(output_dir, image_path, records)


def parse_args():
    def check_positive(value: str) -> int:
        ivalue = int(value)
        if ivalue < 2:
            raise argparse.ArgumentTypeError(f'Invalid value: {ivalue}. Please specify an integer of 2 or greater.')
        return ivalue

    def check_alpha(value: str) -> int:
        ivalue = int(value)
        if not 0 <= ivalue <= 255:
            raise argparse.ArgumentTypeError(f'Invalid value: {ivalue}. Please specify an integer in the range 0-255.')
        return ivalue

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=DEFAULT_CONFIG)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--images_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default=None)
    parser.add_argument('--inference_type', type=str, choices=['fp16', 'int8'], default='fp16')
    parser.add_argument('--score_threshold', type=float, default=0.35)
    parser.add_argument('--object_score_threshold', '--object_socre_threshold', dest='object_score_threshold', type=float, default=None)
    parser.add_argument('--attribute_score_threshold', '--attribute_socre_threshold', dest='attribute_score_threshold', type=float, default=None)
    parser.add_argument('--keypoint_threshold', type=float, default=None)
    parser.add_argument('--mask_threshold', type=float, default=0.4)
    parser.add_argument('--mask_alpha', type=check_alpha, default=160)
    parser.add_argument('--mask_resize_origin', type=str, choices=['topleft', 'center'], default='topleft')
    parser.add_argument('--enable-masks', action='store_true')
    parser.add_argument('--enable-contours', action='store_true')
    parser.add_argument('--keypoint_drawing_mode', type=str, choices=['dot', 'box', 'both'], default='dot')
    parser.add_argument('--enable_bone_drawing_mode', action='store_true')
    parser.add_argument('--disable_generation_identification_mode', action='store_true')
    parser.add_argument('--disable_gender_identification_mode', action='store_true')
    parser.add_argument('--disable_left_and_right_hand_identification_mode', action='store_true')
    parser.add_argument('--disable_headpose_identification_mode', action='store_true')
    parser.add_argument('--disable_render_classids', type=int, nargs='*', default=[])
    parser.add_argument('--enable_face_mosaic', action='store_true')
    parser.add_argument('--disable_head_distance_measurement', action='store_true')
    parser.add_argument('--bounding_box_line_width', type=check_positive, default=2)
    parser.add_argument('--camera_horizontal_fov', type=int, default=90)
    parser.add_argument('--save_raw_predictions', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    process_images(parse_args())
