#!/usr/bin/env python3
"""Merge SAMA tie RLE masks into existing wholebody body RLE masks."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


DEFAULT_ROOT = Path('tools/dataset/wholebody49_coco_ft')
COCO_KEY_RE = re.compile(r'(\d{12})')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Write train_ins_next.json by merging matching SAMA tie RLE masks '
            'into train_ins.json body RLE masks.'
        )
    )
    parser.add_argument(
        '--train-ins',
        type=Path,
        default=DEFAULT_ROOT / 'annotations' / 'train_ins.json',
        help='Input wholebody train instance annotation JSON.',
    )
    parser.add_argument(
        '--sama-rle',
        type=Path,
        default=DEFAULT_ROOT / 'annotations' / 'sama_coco_all_tie_only_no_crowd_rle.json',
        help='Input SAMA tie annotation JSON with compressed RLE segmentations.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_ROOT / 'annotations' / 'train_ins_next.json',
        help='Output annotation JSON.',
    )
    parser.add_argument(
        '--summary',
        type=Path,
        default=DEFAULT_ROOT / 'annotations' / 'train_ins_next_summary.json',
        help='Output summary JSON.',
    )
    parser.add_argument('--body-category-id', type=int, default=0)
    parser.add_argument('--tie-category-id', type=int, default=32)
    parser.add_argument(
        '--bbox-margin',
        type=float,
        default=0.0,
        help='Margin in output-image pixels when checking whether tie bbox is inside body bbox.',
    )
    parser.add_argument(
        '--assignment-method',
        choices=('smallest-containing-bbox', 'boundary-contact', 'mask-overlap'),
        default='mask-overlap',
        help=(
            'How to choose the target body RLE when multiple body RLE bboxes contain a tie. '
            'mask-overlap prioritizes the ratio of tie mask pixels inside the body mask. '
            'smallest-containing-bbox is the original behavior.'
        ),
    )
    parser.add_argument(
        '--boundary-contact-radius',
        type=int,
        default=2,
        help='Pixel radius used by --assignment-method boundary-contact.',
    )
    parser.add_argument(
        '--keep-area',
        action='store_true',
        help='Keep original body annotation area values instead of recalculating from merged RLE.',
    )
    parser.add_argument(
        '--indent',
        type=int,
        default=None,
        help='JSON indent for output files. Default writes compact JSON.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Compute summary without writing the updated train_ins_next.json.',
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def dump_json_atomic(path: Path, data: dict, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        if indent is not None:
            f.write('\n')
    os.replace(tmp_path, path)


def coco_key_from_file_name(file_name: str) -> str | None:
    match = COCO_KEY_RE.search(file_name)
    return match.group(1) if match else None


def bbox_inside(inside_bbox: list[float], outer_bbox: list[float], margin: float) -> bool:
    x, y, w, h = inside_bbox
    ox, oy, ow, oh = outer_bbox
    return (
        x >= ox - margin
        and y >= oy - margin
        and x + w <= ox + ow + margin
        and y + h <= oy + oh + margin
    )


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, float(bbox[2])) * max(0.0, float(bbox[3]))


def bbox_intersects(first_bbox: list[float], second_bbox: list[float]) -> bool:
    x1, y1, w1, h1 = first_bbox
    x2, y2, w2, h2 = second_bbox
    return x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2


def scale_sama_bbox(ann: dict, sama_images_by_id: dict[int, dict], width: int, height: int) -> list[float] | None:
    bbox = ann.get('bbox')
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    sama_image = sama_images_by_id.get(int(ann['image_id']))
    if sama_image is None:
        return None

    sx = width / float(sama_image['width'])
    sy = height / float(sama_image['height'])
    x, y, w, h = [float(value) for value in bbox]
    return [x * sx, y * sy, w * sx, h * sy]


def decode_rle_to_shape(segmentation: dict, width: int, height: int) -> np.ndarray:
    rle = dict(segmentation)
    if isinstance(rle.get('counts'), str):
        rle['counts'] = rle['counts'].encode('ascii')

    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    mask = mask.astype(bool)

    if mask.shape != (height, width):
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode='L')
        mask_image = mask_image.resize((width, height), Image.Resampling.NEAREST)
        mask = np.array(mask_image) > 0

    return mask


def encode_binary_mask(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle['counts']
    if isinstance(counts, bytes):
        counts = counts.decode('ascii')
    return {
        'size': [int(rle['size'][0]), int(rle['size'][1])],
        'counts': counts,
    }


def boundary_contact_score(body_mask: np.ndarray, tie_mask: np.ndarray, radius: int) -> int:
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    body = body_mask.astype(np.uint8)
    tie = tie_mask.astype(np.uint8)
    body_boundary = (cv2.dilate(body, kernel) > 0) & (cv2.erode(body, kernel) == 0)
    tie_boundary = (cv2.dilate(tie, kernel) > 0) & (cv2.erode(tie, kernel) == 0)
    tie_boundary_neighborhood = cv2.dilate(tie_boundary.astype(np.uint8), kernel).astype(bool)
    return int((body_boundary & tie_boundary_neighborhood).sum())


def tie_inside_body_near_boundary_score(body_mask: np.ndarray, tie_mask: np.ndarray, radius: int) -> int:
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tie_boundary = boundary_mask(tie_mask, radius)
    tie_boundary_neighborhood = cv2.dilate(tie_boundary.astype(np.uint8), kernel).astype(bool)
    return int((tie_mask & body_mask & tie_boundary_neighborhood).sum())


def boundary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_u8 = mask.astype(np.uint8)
    return (cv2.dilate(mask_u8, kernel) > 0) & (cv2.erode(mask_u8, kernel) == 0)


def build_sama_ties_by_key(sama: dict, tie_category_id: int) -> tuple[dict[int, dict], dict[str, list[dict]]]:
    images_by_id = {int(image['id']): image for image in sama.get('images', [])}
    ties_by_key: dict[str, list[dict]] = defaultdict(list)

    for ann in sama.get('annotations', []):
        if int(ann.get('category_id', -1)) != tie_category_id:
            continue
        if not isinstance(ann.get('segmentation'), dict):
            continue
        ties_by_key[f'{int(ann["image_id"]):012d}'].append(ann)

    return images_by_id, ties_by_key


def build_train_indexes(train: dict, body_category_id: int) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    body_rles_by_image_id: dict[int, list[dict]] = defaultdict(list)
    body_bboxes_by_image_id: dict[int, list[dict]] = defaultdict(list)

    for ann in train.get('annotations', []):
        if int(ann.get('category_id', -1)) != body_category_id:
            continue

        image_id = int(ann['image_id'])
        bbox = ann.get('bbox')
        if isinstance(bbox, list) and len(bbox) == 4:
            body_bboxes_by_image_id[image_id].append(ann)
        if isinstance(ann.get('segmentation'), dict):
            body_rles_by_image_id[image_id].append(ann)

    return body_rles_by_image_id, body_bboxes_by_image_id


def choose_target_body_rle(
    tie_bbox: list[float],
    tie_ann: dict,
    body_rle_anns: list[dict],
    margin: float,
    width: int,
    height: int,
    assignment_method: str,
    boundary_contact_radius: int,
) -> dict | None:
    containing = []
    intersecting = []
    for ann in body_rle_anns:
        bbox = ann.get('bbox')
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        body_bbox = [float(v) for v in bbox]
        if bbox_inside(tie_bbox, body_bbox, margin):
            containing.append(ann)
        if bbox_intersects(tie_bbox, body_bbox):
            intersecting.append(ann)

    if assignment_method == 'smallest-containing-bbox':
        if not containing:
            return None
        return min(containing, key=lambda ann: bbox_area([float(v) for v in ann['bbox']]))

    candidates = intersecting or containing
    if not candidates:
        return None

    tie_mask = decode_rle_to_shape(tie_ann['segmentation'], width, height)

    if assignment_method == 'boundary-contact':
        def boundary_score(ann: dict) -> tuple[int, int, float]:
            body_mask = decode_rle_to_shape(ann['segmentation'], width, height)
            contact = boundary_contact_score(body_mask, tie_mask, boundary_contact_radius)
            overlap = int((body_mask & tie_mask).sum())
            return contact, overlap, -bbox_area([float(v) for v in ann['bbox']])

        return max(candidates, key=boundary_score)

    tie_area = max(1, int(tie_mask.sum()))

    def overlap_score(ann: dict) -> tuple[float, int, int, int, float]:
        body_mask = decode_rle_to_shape(ann['segmentation'], width, height)
        overlap = int((body_mask & tie_mask).sum())
        inside_ratio = overlap / tie_area
        inside_near_boundary = tie_inside_body_near_boundary_score(body_mask, tie_mask, boundary_contact_radius)
        contact = boundary_contact_score(body_mask, tie_mask, boundary_contact_radius)
        return inside_ratio, overlap, inside_near_boundary, contact, -bbox_area([float(v) for v in ann['bbox']])

    return max(candidates, key=overlap_score)



def main() -> None:
    args = parse_args()

    train = load_json(args.train_ins)
    sama = load_json(args.sama_rle)

    sama_images_by_id, sama_ties_by_key = build_sama_ties_by_key(sama, args.tie_category_id)
    body_rles_by_image_id, body_bboxes_by_image_id = build_train_indexes(train, args.body_category_id)

    stats: Counter[str] = Counter()
    rows = []
    assignments_by_body_ann_id: dict[int, list[dict]] = defaultdict(list)

    for image in train.get('images', []):
        file_name = str(image.get('file_name', ''))
        coco_key = coco_key_from_file_name(file_name)
        if coco_key is None:
            stats['images_without_coco_key'] += 1
            continue
        if coco_key not in sama_ties_by_key:
            continue

        stats['matched_train_images'] += 1
        image_id = int(image['id'])
        width = int(image['width'])
        height = int(image['height'])
        body_rle_anns = body_rles_by_image_id.get(image_id, [])
        body_bbox_anns = body_bboxes_by_image_id.get(image_id, [])
        tie_anns = sama_ties_by_key[coco_key]

        row = {
            'file_name': file_name,
            'wholebody_image_id': image_id,
            'coco_key': coco_key,
            'body_bbox_annotations': len(body_bbox_anns),
            'body_rle_annotations': len(body_rle_anns),
            'tie_annotations_total': len(tie_anns),
            'tie_annotations_inside_body_bbox': 0,
            'tie_annotations_assigned_to_body_rle': 0,
            'tie_annotations_rejected_outside_body_bbox': 0,
            'tie_annotations_rejected_no_target_body_rle': 0,
            'skip_reason': None,
        }

        if not body_rle_anns:
            stats['skipped_no_body_rle_images'] += 1
            stats['skipped_no_body_rle_ties'] += len(tie_anns)
            row['skip_reason'] = 'no_body_rle'
            rows.append(row)
            continue

        body_bboxes = []
        for ann in body_bbox_anns:
            bbox = ann.get('bbox')
            if isinstance(bbox, list) and len(bbox) == 4:
                body_bboxes.append([float(value) for value in bbox])

        for tie_ann in tie_anns:
            tie_bbox = scale_sama_bbox(tie_ann, sama_images_by_id, width, height)
            if tie_bbox is None:
                stats['tie_rejected_missing_bbox_or_image'] += 1
                continue

            if not any(bbox_inside(tie_bbox, body_bbox, args.bbox_margin) for body_bbox in body_bboxes):
                stats['tie_rejected_outside_body_bbox'] += 1
                row['tie_annotations_rejected_outside_body_bbox'] += 1
                continue

            stats['tie_inside_body_bbox'] += 1
            row['tie_annotations_inside_body_bbox'] += 1

            target_body = choose_target_body_rle(
                tie_bbox=tie_bbox,
                tie_ann=tie_ann,
                body_rle_anns=body_rle_anns,
                margin=args.bbox_margin,
                width=width,
                height=height,
                assignment_method=args.assignment_method,
                boundary_contact_radius=args.boundary_contact_radius,
            )
            if target_body is None:
                stats['tie_rejected_no_target_body_rle'] += 1
                row['tie_annotations_rejected_no_target_body_rle'] += 1
                continue

            assignments_by_body_ann_id[int(target_body['id'])].append(tie_ann)
            stats['tie_assigned_to_body_rle'] += 1
            row['tie_annotations_assigned_to_body_rle'] += 1

        if row['tie_annotations_assigned_to_body_rle']:
            stats['images_with_assigned_tie'] += 1
        else:
            stats['images_without_assigned_tie'] += 1

        rows.append(row)

    updated_body_ann_ids = set()
    for image in train.get('images', []):
        image_id = int(image['id'])
        body_rle_anns = body_rles_by_image_id.get(image_id, [])
        if not body_rle_anns:
            continue

        width = int(image['width'])
        height = int(image['height'])

        for body_ann in body_rle_anns:
            tie_anns = assignments_by_body_ann_id.get(int(body_ann['id']))
            if not tie_anns:
                continue

            merged = decode_rle_to_shape(body_ann['segmentation'], width, height)
            original_area = float(mask_utils.area({
                'size': body_ann['segmentation']['size'],
                'counts': body_ann['segmentation']['counts'].encode('ascii')
                if isinstance(body_ann['segmentation'].get('counts'), str)
                else body_ann['segmentation']['counts'],
            }))

            for tie_ann in tie_anns:
                merged |= decode_rle_to_shape(tie_ann['segmentation'], width, height)

            merged_rle = encode_binary_mask(merged)
            old_area_value = float(body_ann.get('area', 0.0) or 0.0)
            new_area_value = float(mask_utils.area({
                'size': merged_rle['size'],
                'counts': merged_rle['counts'].encode('ascii'),
            }))

            changed = body_ann.get('segmentation') != merged_rle
            if not args.keep_area:
                changed = changed or old_area_value != new_area_value

            body_ann['segmentation'] = merged_rle
            if not args.keep_area:
                body_ann['area'] = new_area_value

            stats['body_rle_annotations_with_assigned_tie'] += 1
            stats['updated_body_rle_original_area'] += original_area
            stats['updated_body_rle_new_area'] += new_area_value
            if changed:
                stats['changed_body_rle_annotations'] += 1
                updated_body_ann_ids.add(int(body_ann['id']))
            else:
                stats['unchanged_body_rle_annotations_already_covered_tie'] += 1

    summary = {
        'train_ins': str(args.train_ins),
        'sama_rle': str(args.sama_rle),
        'output': str(args.output),
        'summary': str(args.summary),
        'body_category_id': args.body_category_id,
        'tie_category_id': args.tie_category_id,
        'bbox_margin': args.bbox_margin,
        'assignment_method': args.assignment_method,
        'boundary_contact_radius': args.boundary_contact_radius,
        'blank_fill_enabled': False,
        'skip_rule': 'skip image when train body RLE count is zero, even if matching tie RLE exists',
        'tie_filter': 'scaled tie bbox must be fully inside any train body bbox',
        'candidate_filter': (
            'smallest-containing-bbox uses body RLE bboxes that fully contain the tie bbox; '
            'mask-overlap and boundary-contact use body RLE bboxes that intersect the tie bbox'
        ),
        'update_scope': 'only existing body RLE annotation segmentation is replaced; bbox and annotation count are unchanged',
        'area_updated': not args.keep_area,
        'dry_run': args.dry_run,
        'matched_train_images': stats['matched_train_images'],
        'skipped_no_body_rle_images': stats['skipped_no_body_rle_images'],
        'skipped_no_body_rle_ties': stats['skipped_no_body_rle_ties'],
        'images_with_assigned_tie': stats['images_with_assigned_tie'],
        'images_without_assigned_tie': stats['images_without_assigned_tie'],
        'tie_inside_body_bbox': stats['tie_inside_body_bbox'],
        'tie_assigned_to_body_rle': stats['tie_assigned_to_body_rle'],
        'tie_rejected_outside_body_bbox': stats['tie_rejected_outside_body_bbox'],
        'tie_rejected_no_target_body_rle': stats['tie_rejected_no_target_body_rle'],
        'tie_rejected_missing_bbox_or_image': stats['tie_rejected_missing_bbox_or_image'],
        'body_rle_annotations_with_assigned_tie': stats['body_rle_annotations_with_assigned_tie'],
        'changed_body_rle_annotations': stats['changed_body_rle_annotations'],
        'unchanged_body_rle_annotations_already_covered_tie': stats[
            'unchanged_body_rle_annotations_already_covered_tie'
        ],
        'updated_body_rle_original_area': stats['updated_body_rle_original_area'],
        'updated_body_rle_new_area': stats['updated_body_rle_new_area'],
        'changed_body_annotation_ids': sorted(updated_body_ann_ids),
    }

    summary_data = {
        'summary': summary,
        'images': rows,
    }

    if not args.dry_run:
        dump_json_atomic(args.output, train, indent=args.indent)
    dump_json_atomic(args.summary, summary_data, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
