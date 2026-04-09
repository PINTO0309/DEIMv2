import argparse
import copy
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
FILENAME_RE = re.compile(r'^(\d{12})(?:_[^./]+)?(?:__rep\d+)?\.[A-Za-z0-9]+$')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-json',
        type=Path,
        default=Path('/media/b920405/ExtremeSSD/make_wholebody40/train.json'),
    )
    parser.add_argument(
        '--val-json',
        type=Path,
        default=Path('/media/b920405/ExtremeSSD/make_wholebody40/val.json'),
    )
    parser.add_argument(
        '--src-donor-json',
        type=Path,
        default=None,
        help='Single donor COCO JSON. When set, legacy src-train/src-val inputs are ignored.',
    )
    parser.add_argument(
        '--src-train-json',
        type=Path,
        default=Path('/media/b920405/ExtremeSSD/make_wholebody40/instances_train2017_person_only_no_crowd.json'),
    )
    parser.add_argument(
        '--src-val-json',
        type=Path,
        default=Path('/media/b920405/ExtremeSSD/make_wholebody40/instances_val2017_person_only_no_crowd.json'),
    )
    parser.add_argument(
        '--src-trainval-json',
        type=Path,
        default=Path('/media/b920405/ExtremeSSD/make_wholebody40/instances_trainval2017_person_only_no_crowd.json'),
    )
    parser.add_argument(
        '--train-out',
        type=Path,
        default=Path('/media/b920405/ExtremeSSD/make_wholebody40/train_ins.json'),
    )
    parser.add_argument(
        '--val-out',
        type=Path,
        default=Path('/media/b920405/ExtremeSSD/make_wholebody40/val_ins.json'),
    )
    parser.add_argument(
        '--output-mask-format',
        choices=['rle', 'polygon'],
        default='rle',
    )
    parser.add_argument(
        '--match-iou-threshold',
        type=float,
        default=0.70,
    )
    parser.add_argument(
        '--report-json',
        type=Path,
        default=Path('/media/b920405/ExtremeSSD/make_wholebody40/merge_person_masks_report.json'),
    )
    return parser.parse_args()


def require_mask_utils(context: str) -> None:
    if mask_utils is None:
        raise RuntimeError(f'pycocotools is required for {context}.')


def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def canonical_image_key(file_name: str) -> str:
    base_name = Path(file_name).name
    matched = FILENAME_RE.match(base_name)
    if matched:
        return f'{matched.group(1)}.jpg'
    return base_name


def assert_unique_ids(items: List[dict], field_name: str, file_label: str) -> None:
    seen = set()
    duplicates = set()
    for item in items:
        value = item[field_name]
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f'Duplicate {field_name} found in {file_label}: {sorted(list(duplicates))[:10]}')


def merge_unique_by_id(items_a: List[dict], items_b: List[dict], key: str) -> List[dict]:
    merged = {}
    for item in items_a + items_b:
        merged[item[key]] = copy.deepcopy(item)
    return [merged[idx] for idx in sorted(merged)]


def create_combined_donor(train_data: dict, val_data: dict) -> dict:
    assert_unique_ids(train_data.get('images', []), 'id', 'src-train images')
    assert_unique_ids(val_data.get('images', []), 'id', 'src-val images')
    assert_unique_ids(train_data.get('annotations', []), 'id', 'src-train annotations')
    assert_unique_ids(val_data.get('annotations', []), 'id', 'src-val annotations')

    combined = copy.deepcopy(train_data)
    combined['images'] = copy.deepcopy(train_data.get('images', [])) + copy.deepcopy(val_data.get('images', []))
    combined['annotations'] = copy.deepcopy(train_data.get('annotations', [])) + copy.deepcopy(val_data.get('annotations', []))
    combined['categories'] = merge_unique_by_id(train_data.get('categories', []), val_data.get('categories', []), 'id')
    combined['licenses'] = merge_unique_by_id(train_data.get('licenses', []), val_data.get('licenses', []), 'id')
    if 'info' not in combined and 'info' in val_data:
        combined['info'] = copy.deepcopy(val_data['info'])

    assert_unique_ids(combined.get('images', []), 'id', 'combined donor images')
    assert_unique_ids(combined.get('annotations', []), 'id', 'combined donor annotations')
    return combined


def build_image_maps(data: dict) -> Tuple[Dict[int, dict], Dict[str, dict]]:
    images_by_id = {}
    images_by_key = {}
    for image in data.get('images', []):
        images_by_id[image['id']] = image
        key = canonical_image_key(image['file_name'])
        if key in images_by_key and images_by_key[key]['id'] != image['id']:
            raise ValueError(f'Duplicate canonical image key found: {key}')
        images_by_key[key] = image
    return images_by_id, images_by_key


def group_annotations_by_image(data: dict) -> Dict[int, List[dict]]:
    grouped = defaultdict(list)
    for annotation in data.get('annotations', []):
        grouped[annotation['image_id']].append(annotation)
    return grouped


def xywh_to_xyxy(box: List[float]) -> List[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


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


def resize_mask(mask: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    if mask.shape[1] == target_width and mask.shape[0] == target_height:
        return mask.astype(np.uint8)
    resized = cv2.resize(mask.astype(np.uint8), (int(target_width), int(target_height)), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.uint8)


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
        points = contour[:, 0, :].astype(np.float32)
        polygon = points.reshape(-1).tolist()
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


def default_split_report(split_name: str) -> dict:
    return {
        'split': split_name,
        'images_total': 0,
        'body_annotations_total': 0,
        'non_body_annotations_total': 0,
        'matched_body_annotations': 0,
        'unmatched_body_annotations': 0,
        'non_body_initialized': 0,
        'missing_donor_images': 0,
        'images_with_donor': 0,
        'images_body_gt_more_than_donor': 0,
        'images_body_gt_less_than_donor': 0,
        'invalid_donor_segmentations': 0,
        'low_iou_rejections': 0,
        'area_recomputed': 0,
        'donor_segmentation_type_counts': {'polygon': 0, 'rle': 0, 'invalid': 0},
        'matched_iou_values': [],
        'rejection_examples': [],
        'missing_image_examples': [],
    }


def append_example(report: dict, key: str, example: dict, limit: int = 20) -> None:
    if len(report[key]) < limit:
        report[key].append(example)


def prepare_donor_annotations(
    donor_annotations: List[dict],
    donor_image: dict,
    target_image: dict,
    report: dict,
) -> List[dict]:
    prepared = []
    scale_x = target_image['width'] / donor_image['width']
    scale_y = target_image['height'] / donor_image['height']
    donor_width = int(donor_image['width'])
    donor_height = int(donor_image['height'])
    target_width = int(target_image['width'])
    target_height = int(target_image['height'])

    for donor_ann in donor_annotations:
        if donor_ann.get('category_id') != DONOR_PERSON_CATEGORY_ID:
            continue

        segmentation = donor_ann.get('segmentation')
        kind = segmentation_kind(segmentation)
        report['donor_segmentation_type_counts'][kind] += 1

        mask = segmentation_to_mask(segmentation, donor_width, donor_height)
        if mask is None:
            report['invalid_donor_segmentations'] += 1
            continue

        target_mask = resize_mask(mask, target_width, target_height)
        if int(target_mask.sum()) <= 0:
            report['invalid_donor_segmentations'] += 1
            continue

        prepared.append({
            'annotation': donor_ann,
            'bbox': scale_bbox(donor_ann['bbox'], scale_x, scale_y),
            'mask': target_mask,
        })
    return prepared


def reset_annotation_to_empty(annotation: dict) -> None:
    annotation['segmentation'] = []
    annotation['area'] = 0


def process_split(
    split_name: str,
    target_data: dict,
    donor_images_by_key: Dict[str, dict],
    donor_annotations_by_image: Dict[int, List[dict]],
    match_iou_threshold: float,
    output_mask_format: str,
) -> Tuple[dict, dict]:
    output = copy.deepcopy(target_data)
    report = default_split_report(split_name)
    report['images_total'] = len(output.get('images', []))

    target_ann_indices_by_image = defaultdict(list)
    for index, annotation in enumerate(output.get('annotations', [])):
        target_ann_indices_by_image[annotation['image_id']].append(index)

    for image in tqdm(
        output.get('images', []),
        desc=f'Processing {split_name}',
        dynamic_ncols=True,
    ):
        image_id = image['id']
        annotation_indices = target_ann_indices_by_image.get(image_id, [])
        if not annotation_indices:
            continue

        image_key = canonical_image_key(image['file_name'])
        body_indices = []
        for ann_index in annotation_indices:
            annotation = output['annotations'][ann_index]
            if annotation['category_id'] == BODY_CATEGORY_ID:
                report['body_annotations_total'] += 1
                body_indices.append(ann_index)
            else:
                report['non_body_annotations_total'] += 1
                report['non_body_initialized'] += 1
                reset_annotation_to_empty(annotation)

        if not body_indices:
            continue

        donor_image = donor_images_by_key.get(image_key)
        if donor_image is None:
            report['missing_donor_images'] += 1
            append_example(report, 'missing_image_examples', {'image_key': image_key, 'target_file_name': image['file_name']})
            for ann_index in body_indices:
                reset_annotation_to_empty(output['annotations'][ann_index])
                report['unmatched_body_annotations'] += 1
            continue

        report['images_with_donor'] += 1
        donor_annotations = donor_annotations_by_image.get(donor_image['id'], [])
        prepared_donors = prepare_donor_annotations(donor_annotations, donor_image, image, report)

        if len(body_indices) > len(prepared_donors):
            report['images_body_gt_more_than_donor'] += 1
        elif len(body_indices) < len(prepared_donors):
            report['images_body_gt_less_than_donor'] += 1

        if not prepared_donors:
            for ann_index in body_indices:
                reset_annotation_to_empty(output['annotations'][ann_index])
                report['unmatched_body_annotations'] += 1
            continue

        cost_matrix = np.ones((len(body_indices), len(prepared_donors)), dtype=np.float64)
        iou_matrix = np.zeros((len(body_indices), len(prepared_donors)), dtype=np.float64)
        for row, ann_index in enumerate(body_indices):
            target_bbox = output['annotations'][ann_index]['bbox']
            for col, donor in enumerate(prepared_donors):
                iou_value = box_iou_xywh(target_bbox, donor['bbox'])
                iou_matrix[row, col] = iou_value
                cost_matrix[row, col] = 1.0 - iou_value

        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        matched_body_rows = set()
        for row, col in zip(row_indices.tolist(), col_indices.tolist()):
            iou_value = float(iou_matrix[row, col])
            target_ann = output['annotations'][body_indices[row]]
            if iou_value < match_iou_threshold:
                report['low_iou_rejections'] += 1
                reset_annotation_to_empty(target_ann)
                report['unmatched_body_annotations'] += 1
                append_example(
                    report,
                    'rejection_examples',
                    {
                        'image_key': image_key,
                        'target_annotation_id': target_ann['id'],
                        'donor_annotation_id': prepared_donors[col]['annotation']['id'],
                        'iou': round(iou_value, 6),
                    }
                )
                matched_body_rows.add(row)
                continue

            donor_mask = prepared_donors[col]['mask']
            target_ann['segmentation'] = serialize_mask(donor_mask, output_mask_format)
            target_ann['area'] = mask_area(donor_mask)
            report['matched_body_annotations'] += 1
            report['area_recomputed'] += 1
            report['matched_iou_values'].append(iou_value)
            matched_body_rows.add(row)

        for row, ann_index in enumerate(body_indices):
            if row in matched_body_rows:
                continue
            reset_annotation_to_empty(output['annotations'][ann_index])
            report['unmatched_body_annotations'] += 1

    return output, report


def finalize_report(report: dict) -> dict:
    ious = sorted(report.pop('matched_iou_values'))
    if ious:
        report['matched_iou_summary'] = {
            'count': len(ious),
            'min': float(ious[0]),
            'p50': float(np.percentile(ious, 50)),
            'p90': float(np.percentile(ious, 90)),
            'p95': float(np.percentile(ious, 95)),
            'max': float(ious[-1]),
        }
    else:
        report['matched_iou_summary'] = {
            'count': 0,
            'min': None,
            'p50': None,
            'p90': None,
            'p95': None,
            'max': None,
        }
    return report


def resolve_donor_data(args) -> Tuple[dict, str, Optional[Path]]:
    if args.src_donor_json is not None:
        donor_data = load_json(args.src_donor_json)
        return donor_data, 'single_json', args.src_donor_json

    donor_train_data = load_json(args.src_train_json)
    donor_val_data = load_json(args.src_val_json)
    combined_donor = create_combined_donor(donor_train_data, donor_val_data)
    save_json(args.src_trainval_json, combined_donor)
    return combined_donor, 'legacy_merge', args.src_trainval_json


def main():
    args = parse_args()

    train_data = load_json(args.train_json)
    val_data = load_json(args.val_json)
    donor_data, donor_mode, donor_path = resolve_donor_data(args)

    donor_images_by_id, donor_images_by_key = build_image_maps(donor_data)
    donor_annotations_by_image = group_annotations_by_image(donor_data)
    _ = donor_images_by_id  # kept for symmetry with other maps

    train_output, train_report = process_split(
        split_name='train',
        target_data=train_data,
        donor_images_by_key=donor_images_by_key,
        donor_annotations_by_image=donor_annotations_by_image,
        match_iou_threshold=args.match_iou_threshold,
        output_mask_format=args.output_mask_format,
    )
    val_output, val_report = process_split(
        split_name='val',
        target_data=val_data,
        donor_images_by_key=donor_images_by_key,
        donor_annotations_by_image=donor_annotations_by_image,
        match_iou_threshold=args.match_iou_threshold,
        output_mask_format=args.output_mask_format,
    )

    save_json(args.train_out, train_output)
    save_json(args.val_out, val_output)

    report = {
        'paths': {
            'train_json': str(args.train_json),
            'val_json': str(args.val_json),
            'src_donor_json': str(args.src_donor_json) if args.src_donor_json is not None else None,
            'src_train_json': str(args.src_train_json),
            'src_val_json': str(args.src_val_json),
            'src_trainval_json': str(args.src_trainval_json),
            'resolved_donor_json': str(donor_path) if donor_path is not None else None,
            'train_out': str(args.train_out),
            'val_out': str(args.val_out),
        },
        'donor_mode': donor_mode,
        'output_mask_format': args.output_mask_format,
        'legacy_inputs_used': donor_mode == 'legacy_merge',
        'legacy_inputs_ignored': donor_mode == 'single_json',
        'match_iou_threshold': args.match_iou_threshold,
        'combined_donor': {
            'images_total': len(donor_data.get('images', [])),
            'annotations_total': len(donor_data.get('annotations', [])),
            'categories_total': len(donor_data.get('categories', [])),
        },
        'train': finalize_report(train_report),
        'val': finalize_report(val_report),
    }
    save_json(args.report_json, report)

    if donor_mode == 'legacy_merge':
        print('Saved combined donor to:', args.src_trainval_json)
    else:
        print('Using donor JSON:', donor_path)
    print('Saved train output to:', args.train_out)
    print('Saved val output to:', args.val_out)
    print('Saved report to:', args.report_json)


if __name__ == '__main__':
    main()
