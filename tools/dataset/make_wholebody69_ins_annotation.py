import argparse
import copy
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

try:
    from pycocotools import mask as mask_utils
except Exception:  # pragma: no cover - optional dependency
    mask_utils = None


BODY_CATEGORY_ID = 0
DONOR_PERSON_CATEGORY_ID = 1
MASK_DONOR_BODY_CATEGORY_ID = 0
COCO_ID_RE = re.compile(r'(\d{12})')

HAND_KEYPOINTS = [
    (1, 'hand_thumb_cmc'),
    (2, 'hand_thumb_mcp'),
    (3, 'hand_thumb_ip'),
    (4, 'hand_thumb_tip'),
    (5, 'hand_index_mcp'),
    (6, 'hand_index_pip'),
    (7, 'hand_index_dip'),
    (8, 'hand_index_tip'),
    (9, 'hand_middle_mcp'),
    (10, 'hand_middle_pip'),
    (11, 'hand_middle_dip'),
    (12, 'hand_middle_tip'),
    (13, 'hand_ring_mcp'),
    (14, 'hand_ring_pip'),
    (15, 'hand_ring_dip'),
    (16, 'hand_ring_tip'),
    (17, 'hand_pinky_mcp'),
    (18, 'hand_pinky_pip'),
    (19, 'hand_pinky_dip'),
    (20, 'hand_pinky_tip'),
]

HAND_KEYPOINT_NAMES = [name for _, name in HAND_KEYPOINTS]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-json', type=Path, default=Path('wholebody49/annotations/train.json'))
    parser.add_argument('--val-json', type=Path, default=Path('wholebody49/annotations/val.json'))
    parser.add_argument(
        '--src-mask-donor-train-json',
        type=Path,
        default=Path('wholebody49/annotations/train_ins.json'),
    )
    parser.add_argument(
        '--src-mask-donor-val-json',
        type=Path,
        default=Path('wholebody49/annotations/val_ins.json'),
    )
    parser.add_argument(
        '--src-wholebody-train-json',
        type=Path,
        default=Path('raw/coco_wholebody_train_v1.0.json'),
    )
    parser.add_argument(
        '--src-wholebody-val-json',
        type=Path,
        default=Path('raw/coco_wholebody_val_v1.0.json'),
    )
    parser.add_argument('--src-classes-file', type=Path, default=Path('wholebody49/classes.txt'))
    parser.add_argument('--output-root', type=Path, default=Path('wholebody69'))
    parser.add_argument(
        '--report-json',
        type=Path,
        default=Path('wholebody69/annotations/merge_person_masks_hand_keypoints_report.json'),
    )
    parser.add_argument('--output-mask-format', choices=['rle', 'polygon'], default='rle')
    parser.add_argument('--match-iou-threshold', type=float, default=0.70)
    parser.add_argument('--keypoint-box-size', type=float, default=5.0)
    parser.add_argument('--label-mode', choices=['agnostic', 'sided', 'both'], default='agnostic')
    parser.add_argument('--asset-mode', choices=['none', 'copy'], default='copy')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_classes(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def save_classes(path: Path, class_names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(class_names) + '\n', encoding='utf-8')


def canonical_image_key(file_name: str) -> str:
    matched = COCO_ID_RE.search(Path(file_name).name)
    if matched:
        return f'{matched.group(1)}.jpg'
    return Path(file_name).name


def group_annotations_by_image(data: dict) -> Dict[int, List[dict]]:
    grouped = defaultdict(list)
    for annotation in data.get('annotations', []):
        grouped[annotation['image_id']].append(annotation)
    return grouped


def build_image_map_by_key(data: dict, report: Optional[dict] = None, label: str = 'images') -> Dict[str, dict]:
    images_by_key = {}
    update_image_map_by_key(images_by_key, data, report, label)
    return images_by_key


def update_image_map_by_key(
    images_by_key: Dict[str, dict],
    data: dict,
    report: Optional[dict] = None,
    label: str = 'images',
) -> None:
    for image in data.get('images', []):
        key = canonical_image_key(image['file_name'])
        if key in images_by_key and images_by_key[key]['id'] != image['id']:
            if report is not None:
                report.setdefault('duplicate_image_keys', []).append({
                    'source': label,
                    'key': key,
                    'first_image_id': images_by_key[key]['id'],
                    'duplicate_image_id': image['id'],
                })
            continue
        images_by_key[key] = image


def extend_annotations_by_image(grouped: Dict[int, List[dict]], data: dict) -> None:
    for annotation in data.get('annotations', []):
        grouped[annotation['image_id']].append(annotation)


def xywh_to_xyxy(box: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return x, y, x + w, y + h


def box_iou_xywh(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(box_a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(box_b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def scale_bbox(box: List[float], scale_x: float, scale_y: float) -> List[float]:
    return [
        float(box[0] * scale_x),
        float(box[1] * scale_y),
        float(box[2] * scale_x),
        float(box[3] * scale_y),
    ]


def scale_keypoint(x: float, y: float, scale_x: float, scale_y: float) -> Tuple[float, float]:
    return float(x * scale_x), float(y * scale_y)


def keypoint_bbox(
    x: float,
    y: float,
    size: float,
    image_width: int,
    image_height: int,
) -> Optional[List[float]]:
    half = size / 2.0
    x1 = max(0.0, x - half)
    y1 = max(0.0, y - half)
    x2 = min(float(image_width), x + half)
    y2 = min(float(image_height), y + half)
    w = x2 - x1
    h = y2 - y1
    if w <= 0.0 or h <= 0.0:
        return None
    return [float(x1), float(y1), float(w), float(h)]


def require_mask_utils(context: str) -> None:
    if mask_utils is None:
        raise RuntimeError(f'pycocotools is required for {context}.')


def segmentation_kind(segmentation) -> str:
    if isinstance(segmentation, list):
        return 'polygon'
    if isinstance(segmentation, dict):
        return 'rle'
    return 'invalid'


def decode_rle_mask(segmentation: dict) -> np.ndarray:
    require_mask_utils('RLE donor segmentations')
    decoded = mask_utils.decode(segmentation)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return (decoded > 0).astype(np.uint8)


def polygon_to_mask(segmentation: List[List[float]], width: int, height: int) -> Optional[np.ndarray]:
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    valid_polygon_count = 0
    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 != 0:
            return None
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 3:
            return None
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
        valid_polygon_count += 1
    return mask if valid_polygon_count > 0 else None


def segmentation_to_mask(segmentation, width: int, height: int) -> Optional[np.ndarray]:
    kind = segmentation_kind(segmentation)
    if kind == 'polygon':
        return polygon_to_mask(segmentation, width, height)
    if kind == 'rle':
        return decode_rle_mask(segmentation)
    return None


def resize_mask(mask: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    if mask.shape[1] == target_width and mask.shape[0] == target_height:
        return mask.astype(np.uint8)
    resized = cv2.resize(mask.astype(np.uint8), (int(target_width), int(target_height)), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.uint8)


def encode_compressed_rle(mask: np.ndarray) -> dict:
    require_mask_utils('RLE output')
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded['counts']
    if isinstance(counts, bytes):
        encoded['counts'] = counts.decode('utf-8')
    return encoded


def mask_to_polygons(mask: np.ndarray) -> List[List[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[List[float]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        polygon = contour[:, 0, :].astype(np.float32).reshape(-1).tolist()
        if len(polygon) >= 6:
            polygons.append([float(v) for v in polygon])
    return polygons


def serialize_mask(mask: np.ndarray, output_mask_format: str):
    if output_mask_format == 'rle':
        return encode_compressed_rle(mask)
    if output_mask_format == 'polygon':
        return mask_to_polygons(mask)
    raise ValueError(f'Unsupported output_mask_format: {output_mask_format}')


def mask_area(mask: np.ndarray) -> float:
    return float(mask.astype(np.uint8).sum())


def build_class_names(base_classes: List[str], label_mode: str) -> List[str]:
    class_names = list(base_classes)
    if label_mode in {'agnostic', 'both'}:
        class_names.extend(HAND_KEYPOINT_NAMES)
    if label_mode in {'sided', 'both'}:
        class_names.extend([f'left_{name}' for name in HAND_KEYPOINT_NAMES])
        class_names.extend([f'right_{name}' for name in HAND_KEYPOINT_NAMES])
    return class_names


def build_categories(base_categories: List[dict], class_names: List[str]) -> List[dict]:
    categories = [copy.deepcopy(cat) for cat in sorted(base_categories, key=lambda item: item['id'])]
    for category_id in range(len(categories), len(class_names)):
        categories.append({'id': category_id, 'name': class_names[category_id]})
    return categories


def category_ids_for_hand(label_mode: str, base_count: int, side: str, category_offset: int) -> List[Tuple[int, str]]:
    ids = []
    if label_mode in {'agnostic', 'both'}:
        ids.append((base_count + category_offset, 'agnostic'))
    if label_mode == 'sided':
        offset = 0 if side == 'left' else len(HAND_KEYPOINT_NAMES)
        ids.append((base_count + offset + category_offset, 'sided'))
    elif label_mode == 'both':
        side_base = base_count + len(HAND_KEYPOINT_NAMES)
        offset = 0 if side == 'left' else len(HAND_KEYPOINT_NAMES)
        ids.append((side_base + offset + category_offset, 'sided'))
    return ids


def init_split_report(split_name: str) -> dict:
    return {
        'split': split_name,
        'images_total': 0,
        'body_annotations_total': 0,
        'mask': {
            'images_with_donor': 0,
            'missing_donor_images': 0,
            'matched_body_annotations': 0,
            'unmatched_body_annotations': 0,
            'low_iou_rejections': 0,
            'invalid_donor_segmentations': 0,
            'area_recomputed': 0,
            'donor_segmentation_type_counts': {'polygon': 0, 'rle': 0, 'invalid': 0},
            'matched_iou_values': [],
            'missing_image_examples': [],
            'rejection_examples': [],
        },
        'hand': {
            'images_with_donor': 0,
            'missing_donor_images': 0,
            'matched_body_annotations': 0,
            'unmatched_body_annotations': 0,
            'low_iou_rejections': 0,
            'valid_left_hands': 0,
            'valid_right_hands': 0,
            'added_annotations': 0,
            'skipped_unlabeled_keypoints': 0,
            'skipped_invalid_bboxes': 0,
            'added_by_keypoint_index': {str(source_index): 0 for source_index, _ in HAND_KEYPOINTS},
            'added_by_side': {'left': 0, 'right': 0},
            'added_by_label_mode': {'agnostic': 0, 'sided': 0},
            'matched_iou_values': [],
            'missing_image_examples': [],
            'rejection_examples': [],
        },
    }


def append_example(report_section: dict, key: str, example: dict, limit: int = 20) -> None:
    if len(report_section[key]) < limit:
        report_section[key].append(example)


def match_body_annotations(
    body_indices: List[int],
    annotations: List[dict],
    target_image: dict,
    donor_image: Optional[dict],
    donor_annotations: Iterable[dict],
    threshold: float,
    donor_category_id: int,
) -> Tuple[Dict[int, dict], List[Tuple[int, dict, float]], int]:
    if donor_image is None:
        return {}, [], len(body_indices)

    scale_x = target_image['width'] / donor_image['width']
    scale_y = target_image['height'] / donor_image['height']
    prepared_donors = []
    for donor_ann in donor_annotations:
        if donor_ann.get('category_id') != donor_category_id:
            continue
        prepared_donors.append({
            'annotation': donor_ann,
            'bbox': scale_bbox(donor_ann['bbox'], scale_x, scale_y),
            'scale_x': scale_x,
            'scale_y': scale_y,
        })

    if not body_indices or not prepared_donors:
        return {}, [], len(body_indices)

    cost_matrix = np.ones((len(body_indices), len(prepared_donors)), dtype=np.float64)
    iou_matrix = np.zeros((len(body_indices), len(prepared_donors)), dtype=np.float64)
    for row, ann_index in enumerate(body_indices):
        target_bbox = annotations[ann_index]['bbox']
        for col, donor in enumerate(prepared_donors):
            iou_value = box_iou_xywh(target_bbox, donor['bbox'])
            iou_matrix[row, col] = iou_value
            cost_matrix[row, col] = 1.0 - iou_value

    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matched = {}
    rejected = []
    seen_rows = set()
    for row, col in zip(row_indices.tolist(), col_indices.tolist()):
        seen_rows.add(row)
        iou_value = float(iou_matrix[row, col])
        ann_index = body_indices[row]
        if iou_value < threshold:
            rejected.append((ann_index, prepared_donors[col]['annotation'], iou_value))
            continue
        matched[ann_index] = {**prepared_donors[col], 'iou': iou_value}

    unmatched_count = len(body_indices) - len(matched) - len(rejected)
    return matched, rejected, unmatched_count


def reset_instance_non_body_annotation(annotation: dict) -> None:
    annotation['segmentation'] = []
    annotation['area'] = 0


def add_hand_annotations_for_match(
    output_annotations: List[dict],
    target_image: dict,
    target_body_ann: dict,
    donor_match: dict,
    next_annotation_id: int,
    base_category_count: int,
    label_mode: str,
    keypoint_box_size: float,
    report: dict,
    update_report: bool = True,
) -> int:
    donor_ann = donor_match['annotation']
    for side, kpts_field, valid_field in (
        ('left', 'lefthand_kpts', 'lefthand_valid'),
        ('right', 'righthand_kpts', 'righthand_valid'),
    ):
        if not donor_ann.get(valid_field):
            continue
        if update_report:
            report[f'valid_{side}_hands'] += 1
        kpts = donor_ann.get(kpts_field, [])
        for category_offset, (source_keypoint_index, _keypoint_name) in enumerate(HAND_KEYPOINTS):
            kpt_offset = source_keypoint_index * 3
            if len(kpts) < kpt_offset + 3:
                if update_report:
                    report['skipped_unlabeled_keypoints'] += 1
                continue
            x, y, visibility = kpts[kpt_offset:kpt_offset + 3]
            if visibility <= 0:
                if update_report:
                    report['skipped_unlabeled_keypoints'] += 1
                continue
            scaled_x, scaled_y = scale_keypoint(x, y, donor_match['scale_x'], donor_match['scale_y'])
            bbox = keypoint_bbox(
                scaled_x,
                scaled_y,
                keypoint_box_size,
                int(target_image['width']),
                int(target_image['height']),
            )
            if bbox is None:
                if update_report:
                    report['skipped_invalid_bboxes'] += 1
                continue
            for category_id, annotation_label_mode in category_ids_for_hand(
                label_mode,
                base_category_count,
                side,
                category_offset,
            ):
                output_annotations.append({
                    'image_id': target_image['id'],
                    'id': next_annotation_id,
                    'category_id': category_id,
                    'bbox': bbox,
                    'area': float(bbox[2] * bbox[3]),
                    'segmentation': [],
                    'iscrowd': 0,
                    'person_annotation_id': target_body_ann['id'],
                    'wholebody_source_annotation_id': donor_ann['id'],
                    'hand_side': side,
                    'hand_keypoint_index': source_keypoint_index,
                    'keypoint_visibility': float(visibility),
                    'hand_label_mode': annotation_label_mode,
                })
                next_annotation_id += 1
                if update_report:
                    report['added_annotations'] += 1
                    report['added_by_keypoint_index'][str(source_keypoint_index)] += 1
                    report['added_by_side'][side] += 1
                    report['added_by_label_mode'][annotation_label_mode] += 1
    return next_annotation_id


def process_split(
    split_name: str,
    target_data: dict,
    categories: List[dict],
    base_category_count: int,
    mask_images_by_key: Dict[str, dict],
    mask_annotations_by_image: Dict[int, List[dict]],
    hand_images_by_key: Dict[str, dict],
    hand_annotations_by_image: Dict[int, List[dict]],
    match_iou_threshold: float,
    output_mask_format: str,
    label_mode: str,
    keypoint_box_size: float,
) -> Tuple[dict, dict, dict]:
    detection_output = copy.deepcopy(target_data)
    instance_output = copy.deepcopy(target_data)
    detection_output['categories'] = copy.deepcopy(categories)
    instance_output['categories'] = copy.deepcopy(categories)

    detection_output['annotations'] = [
        ann for ann in detection_output.get('annotations', []) if ann.get('category_id', -1) < base_category_count
    ]
    instance_output['annotations'] = [
        ann for ann in instance_output.get('annotations', []) if ann.get('category_id', -1) < base_category_count
    ]

    report = init_split_report(split_name)
    report['images_total'] = len(detection_output.get('images', []))

    det_ann_indices_by_image = defaultdict(list)
    for index, annotation in enumerate(detection_output.get('annotations', [])):
        det_ann_indices_by_image[annotation['image_id']].append(index)

    ins_ann_indices_by_image = defaultdict(list)
    for index, annotation in enumerate(instance_output.get('annotations', [])):
        ins_ann_indices_by_image[annotation['image_id']].append(index)
        if annotation.get('category_id') != BODY_CATEGORY_ID:
            reset_instance_non_body_annotation(annotation)

    next_det_ann_id = max((ann['id'] for ann in detection_output.get('annotations', [])), default=-1) + 1
    next_ins_ann_id = max((ann['id'] for ann in instance_output.get('annotations', [])), default=-1) + 1

    for image in tqdm(detection_output.get('images', []), desc=f'Processing {split_name}', dynamic_ncols=True):
        image_id = image['id']
        det_indices = det_ann_indices_by_image.get(image_id, [])
        body_indices = [
            ann_index for ann_index in det_indices
            if detection_output['annotations'][ann_index]['category_id'] == BODY_CATEGORY_ID
        ]
        if not body_indices:
            continue
        report['body_annotations_total'] += len(body_indices)

        image_key = canonical_image_key(image['file_name'])

        mask_image = mask_images_by_key.get(image_key)
        if mask_image is None:
            report['mask']['missing_donor_images'] += 1
            report['mask']['unmatched_body_annotations'] += len(body_indices)
            append_example(report['mask'], 'missing_image_examples', {
                'image_key': image_key,
                'target_file_name': image['file_name'],
            })
        else:
            report['mask']['images_with_donor'] += 1
            mask_matches, mask_rejections, mask_unmatched_count = match_body_annotations(
                body_indices,
                detection_output['annotations'],
                image,
                mask_image,
                mask_annotations_by_image.get(mask_image['id'], []),
                match_iou_threshold,
                MASK_DONOR_BODY_CATEGORY_ID,
            )
            report['mask']['unmatched_body_annotations'] += mask_unmatched_count + len(mask_rejections)
            report['mask']['low_iou_rejections'] += len(mask_rejections)
            for ann_index, donor_ann, iou_value in mask_rejections:
                append_example(report['mask'], 'rejection_examples', {
                    'image_key': image_key,
                    'target_annotation_id': detection_output['annotations'][ann_index]['id'],
                    'donor_annotation_id': donor_ann['id'],
                    'iou': round(iou_value, 6),
                })
            for det_ann_index, donor_match in mask_matches.items():
                ins_ann = instance_output['annotations'][det_ann_index]
                donor_ann = donor_match['annotation']
                segmentation = donor_ann.get('segmentation')
                kind = segmentation_kind(segmentation)
                report['mask']['donor_segmentation_type_counts'][kind] += 1
                donor_mask = segmentation_to_mask(segmentation, int(mask_image['width']), int(mask_image['height']))
                if donor_mask is None:
                    report['mask']['invalid_donor_segmentations'] += 1
                    reset_instance_non_body_annotation(ins_ann)
                    continue
                target_mask = resize_mask(donor_mask, int(image['width']), int(image['height']))
                if int(target_mask.sum()) <= 0:
                    report['mask']['invalid_donor_segmentations'] += 1
                    reset_instance_non_body_annotation(ins_ann)
                    continue
                ins_ann['segmentation'] = serialize_mask(target_mask, output_mask_format)
                ins_ann['area'] = mask_area(target_mask)
                report['mask']['matched_body_annotations'] += 1
                report['mask']['area_recomputed'] += 1
                report['mask']['matched_iou_values'].append(donor_match['iou'])

        hand_image = hand_images_by_key.get(image_key)
        if hand_image is None:
            report['hand']['missing_donor_images'] += 1
            report['hand']['unmatched_body_annotations'] += len(body_indices)
            append_example(report['hand'], 'missing_image_examples', {
                'image_key': image_key,
                'target_file_name': image['file_name'],
            })
            continue

        report['hand']['images_with_donor'] += 1
        hand_matches, hand_rejections, hand_unmatched_count = match_body_annotations(
            body_indices,
            detection_output['annotations'],
            image,
            hand_image,
            hand_annotations_by_image.get(hand_image['id'], []),
            match_iou_threshold,
            DONOR_PERSON_CATEGORY_ID,
        )
        report['hand']['unmatched_body_annotations'] += hand_unmatched_count + len(hand_rejections)
        report['hand']['low_iou_rejections'] += len(hand_rejections)
        for ann_index, donor_ann, iou_value in hand_rejections:
            append_example(report['hand'], 'rejection_examples', {
                'image_key': image_key,
                'target_annotation_id': detection_output['annotations'][ann_index]['id'],
                'donor_annotation_id': donor_ann['id'],
                'iou': round(iou_value, 6),
            })
        for det_ann_index, donor_match in hand_matches.items():
            target_body_ann = detection_output['annotations'][det_ann_index]
            next_det_ann_id = add_hand_annotations_for_match(
                detection_output['annotations'],
                image,
                target_body_ann,
                donor_match,
                next_det_ann_id,
                base_category_count,
                label_mode,
                keypoint_box_size,
                report['hand'],
                update_report=True,
            )
            ins_body_ann = instance_output['annotations'][det_ann_index]
            next_ins_ann_id = add_hand_annotations_for_match(
                instance_output['annotations'],
                image,
                ins_body_ann,
                donor_match,
                next_ins_ann_id,
                base_category_count,
                label_mode,
                keypoint_box_size,
                report['hand'],
                update_report=False,
            )
            report['hand']['matched_body_annotations'] += 1
            report['hand']['matched_iou_values'].append(donor_match['iou'])

    return detection_output, instance_output, finalize_split_report(report)


def summarize_ious(values: List[float]) -> dict:
    values = sorted(values)
    if not values:
        return {'count': 0, 'min': None, 'p50': None, 'p90': None, 'p95': None, 'max': None}
    return {
        'count': len(values),
        'min': float(values[0]),
        'p50': float(np.percentile(values, 50)),
        'p90': float(np.percentile(values, 90)),
        'p95': float(np.percentile(values, 95)),
        'max': float(values[-1]),
    }


def finalize_split_report(report: dict) -> dict:
    for section_name in ('mask', 'hand'):
        values = report[section_name].pop('matched_iou_values')
        report[section_name]['matched_iou_summary'] = summarize_ious(values)
    return report


def validate_dataset(data: dict, class_count: int, base_category_count: int, label_mode: str) -> dict:
    annotation_ids = [ann['id'] for ann in data.get('annotations', [])]
    duplicate_ids = [ann_id for ann_id, count in Counter(annotation_ids).items() if count > 1]
    invalid_category_count = 0
    invalid_bbox_count = 0
    invalid_hand_segmentation_count = 0
    for ann in data.get('annotations', []):
        category_id = ann.get('category_id')
        if not isinstance(category_id, int) or category_id < 0 or category_id >= class_count:
            invalid_category_count += 1
        bbox = ann.get('bbox', [])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            invalid_bbox_count += 1
        if category_id is not None and category_id >= base_category_count and ann.get('segmentation') not in ([], None):
            invalid_hand_segmentation_count += 1
    return {
        'categories_count': len(data.get('categories', [])),
        'annotations_count': len(data.get('annotations', [])),
        'duplicate_annotation_ids': duplicate_ids[:20],
        'duplicate_annotation_id_count': len(duplicate_ids),
        'invalid_category_count': invalid_category_count,
        'invalid_bbox_count': invalid_bbox_count,
        'invalid_hand_segmentation_count': invalid_hand_segmentation_count,
        'label_mode': label_mode,
    }


def copy_assets(src_root: Path, output_root: Path) -> None:
    for dirname in ('images', 'labels'):
        src = src_root / dirname
        dst = output_root / dirname
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)


def main():
    args = parse_args()
    output_annotations_dir = args.output_root / 'annotations'
    train_out = output_annotations_dir / 'train.json'
    val_out = output_annotations_dir / 'val.json'
    train_ins_out = output_annotations_dir / 'train_ins.json'
    val_ins_out = output_annotations_dir / 'val_ins.json'
    classes_out = args.output_root / 'classes.txt'

    train_data = load_json(args.train_json)
    val_data = load_json(args.val_json)
    mask_donor_train_data = load_json(args.src_mask_donor_train_json)
    mask_donor_val_data = load_json(args.src_mask_donor_val_json)
    wholebody_train_data = load_json(args.src_wholebody_train_json)
    wholebody_val_data = load_json(args.src_wholebody_val_json)

    base_classes = read_classes(args.src_classes_file)
    base_category_count = len(base_classes)
    class_names = build_class_names(base_classes, args.label_mode)
    categories = build_categories(train_data.get('categories', [])[:base_category_count], class_names)

    global_report = {
        'paths': {
            'train_json': str(args.train_json),
            'val_json': str(args.val_json),
            'src_mask_donor_train_json': str(args.src_mask_donor_train_json),
            'src_mask_donor_val_json': str(args.src_mask_donor_val_json),
            'src_wholebody_train_json': str(args.src_wholebody_train_json),
            'src_wholebody_val_json': str(args.src_wholebody_val_json),
            'src_classes_file': str(args.src_classes_file),
            'output_root': str(args.output_root),
            'classes_out': str(classes_out),
            'train_out': str(train_out),
            'val_out': str(val_out),
            'train_ins_out': str(train_ins_out),
            'val_ins_out': str(val_ins_out),
            'report_json': str(args.report_json),
        },
        'label_mode': args.label_mode,
        'asset_mode': args.asset_mode,
        'dry_run': args.dry_run,
        'base_category_count': base_category_count,
        'class_count': len(class_names),
        'keypoint_box_size': args.keypoint_box_size,
        'match_iou_threshold': args.match_iou_threshold,
        'output_mask_format': args.output_mask_format,
        'warnings': [],
        'duplicate_image_keys': [],
    }
    if args.label_mode != 'agnostic':
        global_report['warnings'].append(
            f'output_root is named {args.output_root.name}, but label_mode={args.label_mode} produces {len(class_names)} classes.'
        )

    mask_train_images_by_key = build_image_map_by_key(mask_donor_train_data, global_report, 'mask_train_donor')
    mask_train_annotations_by_image = group_annotations_by_image(mask_donor_train_data)
    mask_val_images_by_key = build_image_map_by_key(mask_donor_val_data, global_report, 'mask_val_donor')
    mask_val_annotations_by_image = group_annotations_by_image(mask_donor_val_data)
    hand_images_by_key = {}
    update_image_map_by_key(hand_images_by_key, wholebody_train_data, global_report, 'wholebody_train_donor')
    update_image_map_by_key(hand_images_by_key, wholebody_val_data, global_report, 'wholebody_val_donor')
    hand_annotations_by_image = defaultdict(list)
    extend_annotations_by_image(hand_annotations_by_image, wholebody_train_data)
    extend_annotations_by_image(hand_annotations_by_image, wholebody_val_data)

    train_det, train_ins, train_report = process_split(
        'train',
        train_data,
        categories,
        base_category_count,
        mask_train_images_by_key,
        mask_train_annotations_by_image,
        hand_images_by_key,
        hand_annotations_by_image,
        args.match_iou_threshold,
        args.output_mask_format,
        args.label_mode,
        args.keypoint_box_size,
    )
    val_det, val_ins, val_report = process_split(
        'val',
        val_data,
        categories,
        base_category_count,
        mask_val_images_by_key,
        mask_val_annotations_by_image,
        hand_images_by_key,
        hand_annotations_by_image,
        args.match_iou_threshold,
        args.output_mask_format,
        args.label_mode,
        args.keypoint_box_size,
    )

    global_report['train'] = train_report
    global_report['val'] = val_report
    global_report['validation'] = {
        'train': validate_dataset(train_det, len(class_names), base_category_count, args.label_mode),
        'val': validate_dataset(val_det, len(class_names), base_category_count, args.label_mode),
        'train_ins': validate_dataset(train_ins, len(class_names), base_category_count, args.label_mode),
        'val_ins': validate_dataset(val_ins, len(class_names), base_category_count, args.label_mode),
        'classes_txt_count': len(class_names),
    }

    if args.dry_run:
        print(json.dumps(global_report, ensure_ascii=False, indent=2))
        return

    save_classes(classes_out, class_names)
    save_json(train_out, train_det)
    save_json(val_out, val_det)
    save_json(train_ins_out, train_ins)
    save_json(val_ins_out, val_ins)
    if args.asset_mode == 'copy':
        copy_assets(args.train_json.parents[1], args.output_root)
    save_json(args.report_json, global_report)

    print('Saved classes to:', classes_out)
    print('Saved train detection output to:', train_out)
    print('Saved val detection output to:', val_out)
    print('Saved train instance output to:', train_ins_out)
    print('Saved val instance output to:', val_ins_out)
    print('Saved report to:', args.report_json)


if __name__ == '__main__':
    main()
