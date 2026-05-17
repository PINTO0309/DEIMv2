#!/usr/bin/env python3
"""Build wholebody49 annotations by adding generated bone boxes to wholebody48."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

try:
    from pycocotools import mask as mask_utils
except Exception:  # pragma: no cover - optional dependency
    mask_utils = None


KEYPOINT_NAMES = (
    'collarbone',
    'shoulder_left',
    'shoulder_right',
    'solar_plexus',
    'elbow_left',
    'elbow_right',
    'wrist_left',
    'wrist_right',
    'abdomen',
    'hip_joint_left',
    'hip_joint_right',
    'knee_left',
    'knee_right',
    'ankle_left',
    'ankle_right',
    'foot_left',
    'foot_right',
)

BONE_EDGES = (
    ('collarbone', 'shoulder_left'),
    ('collarbone', 'shoulder_right'),
    ('collarbone', 'solar_plexus'),
    ('solar_plexus', 'abdomen'),
    ('shoulder_left', 'elbow_left'),
    ('elbow_left', 'wrist_left'),
    ('shoulder_right', 'elbow_right'),
    ('elbow_right', 'wrist_right'),
    ('abdomen', 'hip_joint_left'),
    ('hip_joint_left', 'knee_left'),
    ('knee_left', 'ankle_left'),
    ('abdomen', 'hip_joint_right'),
    ('hip_joint_right', 'knee_right'),
    ('knee_right', 'ankle_right'),
)

BODY_CLASS_ID = 0
BONE_CLASS_ID = 48
BONE_CLASS_NAME = 'bone'
MASK_BYPASS_ASSIGNMENT_METHODS = {'single_body_fallback', 'isolated_bbox_fallback'}


@dataclass(frozen=True)
class Keypoint:
    ann_id: int
    category_id: int
    name: str
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]


@dataclass
class BodyRegion:
    index: int
    ann_id: int
    bbox: tuple[float, float, float, float]
    area: float
    mask: object | None = None
    isolated: bool = False


@dataclass
class Assignment:
    body_index: int
    method: str
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate tools/dataset/wholebody49 by adding class 48 bone boxes.',
    )
    parser.add_argument('--input-root', type=Path, default=Path('tools/dataset/wholebody48'))
    parser.add_argument('--output-root', type=Path, default=Path('tools/dataset/wholebody49'))
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=('coco', 'yolo'),
        default=['coco', 'yolo'],
        help='Annotation formats to write.',
    )
    parser.add_argument(
        '--write-ins-json',
        action='store_true',
        help='Also write train_ins.json/val_ins.json with generated bone annotations.',
    )
    parser.add_argument(
        '--image-link-mode',
        choices=('symlink', 'copy', 'none'),
        default='symlink',
        help='How to expose images in the output dataset root.',
    )
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--limit-images', type=int, default=None, help='Smoke-test limit per split.')
    parser.add_argument('--min-bone-thickness', type=float, default=3.0)
    parser.add_argument('--bone-thickness-ratio', type=float, default=0.12)
    parser.add_argument(
        '--endpoint-padding-ratio',
        type=float,
        default=0.45,
        help='Extra center-based padding relative to generated half thickness. Keypoint bbox size is not used.',
    )
    parser.add_argument(
        '--body-bbox-expand-ratio',
        type=float,
        default=0.08,
        help='Fallback expansion for assigning keypoints to body boxes when no mask covers them.',
    )
    parser.add_argument(
        '--max-edge-distance-ratio',
        type=float,
        default=0.75,
        help='Reject implausibly long fallback edges relative to the body bbox diagonal.',
    )
    parser.add_argument('--indent', type=int, default=2)
    parser.add_argument(
        '--no-json-progress',
        action='store_true',
        help='Disable progress bars while writing COCO JSON files.',
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: Path, data: dict, indent: int | None, progress: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        if not isinstance(data, dict):
            json.dump(data, f, ensure_ascii=False, indent=indent)
            return

        if indent is None:
            f.write('{')
            for key_index, (key, value) in enumerate(data.items()):
                if key_index:
                    f.write(', ')
                f.write(json.dumps(str(key), ensure_ascii=False))
                f.write(': ')
                write_json_value(f, value, indent, progress, f'{path.name}:{key}')
            f.write('}')
        else:
            write_pretty_json_dict(f, data, indent, progress, path.name)


def write_json_value(file_obj, value, indent: int | None, progress: bool, desc: str) -> None:
    if isinstance(value, list):
        file_obj.write('[')
        iterator = tqdm(value, desc=f'Writing {desc}', dynamic_ncols=True, disable=not progress)
        for index, item in enumerate(iterator):
            if index:
                file_obj.write(', ')
            file_obj.write(json.dumps(item, ensure_ascii=False, indent=indent))
        file_obj.write(']')
    else:
        file_obj.write(json.dumps(value, ensure_ascii=False, indent=indent))


def write_pretty_json_dict(file_obj, data: dict, indent: int, progress: bool, path_name: str) -> None:
    file_obj.write('{\n')
    items = list(data.items())
    for key_index, (key, value) in enumerate(items):
        file_obj.write(' ' * indent + json.dumps(str(key), ensure_ascii=False) + ': ')
        if isinstance(value, list):
            file_obj.write('[\n')
            iterator = tqdm(
                value,
                desc=f'Writing {path_name}:{key}',
                dynamic_ncols=True,
                disable=not progress,
            )
            for index, item in enumerate(iterator):
                if index:
                    file_obj.write(',\n')
                rendered = json.dumps(item, ensure_ascii=False, indent=indent)
                file_obj.write(indent_multiline(rendered, indent * 2))
            file_obj.write('\n' + ' ' * indent + ']')
        else:
            rendered = json.dumps(value, ensure_ascii=False, indent=indent)
            file_obj.write(indent_multiline(rendered, indent))
        if key_index + 1 < len(items):
            file_obj.write(',')
        file_obj.write('\n')
    file_obj.write('}')


def indent_multiline(text: str, spaces: int) -> str:
    prefix = ' ' * spaces
    return '\n'.join(prefix + line for line in text.splitlines())


def load_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def category_maps(categories: list[dict]) -> tuple[dict[str, int], dict[int, str]]:
    name_to_id = {str(cat['name']): int(cat['id']) for cat in categories}
    id_to_name = {int(cat['id']): str(cat['name']) for cat in categories}
    return name_to_id, id_to_name


def ensure_bone_category(categories: list[dict]) -> list[dict]:
    out = [dict(cat) for cat in categories if int(cat['id']) != BONE_CLASS_ID]
    out.append({'id': BONE_CLASS_ID, 'name': BONE_CLASS_NAME})
    return sorted(out, key=lambda cat: int(cat['id']))


def bbox_center(bbox: Iterable[float]) -> tuple[float, float]:
    x, y, w, h = [float(v) for v in bbox]
    return x + w / 2.0, y + h / 2.0


def bbox_area(bbox: Iterable[float]) -> float:
    _, _, w, h = [float(v) for v in bbox]
    return max(0.0, w) * max(0.0, h)


def bbox_intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    overlap_w = min(ax + aw, bx + bw) - max(ax, bx)
    overlap_h = min(ay + ah, by + bh) - max(ay, by)
    return max(0.0, overlap_w) * max(0.0, overlap_h)


def clamp_bbox(
    bbox: tuple[float, float, float, float],
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float] | None:
    x, y, w, h = bbox
    x1 = max(0.0, min(float(image_width), x))
    y1 = max(0.0, min(float(image_height), y))
    x2 = max(0.0, min(float(image_width), x + w))
    y2 = max(0.0, min(float(image_height), y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def expanded_bbox(bbox: tuple[float, float, float, float], ratio: float) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    pad_x = w * ratio
    pad_y = h * ratio
    return x - pad_x, y - pad_y, w + pad_x * 2.0, h + pad_y * 2.0


def point_in_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    x, y = point
    bx, by, bw, bh = bbox
    return bx <= x <= bx + bw and by <= y <= by + bh


def bbox_distance_to_point(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> float:
    x, y = point
    bx, by, bw, bh = bbox
    dx = max(bx - x, 0.0, x - (bx + bw))
    dy = max(by - y, 0.0, y - (by + bh))
    return math.hypot(dx, dy)


def segmentation_is_non_empty(segmentation) -> bool:
    if isinstance(segmentation, dict):
        return bool(segmentation.get('counts')) and bool(segmentation.get('size'))
    if isinstance(segmentation, list):
        return len(segmentation) > 0
    return False


def decode_mask(segmentation):
    if mask_utils is None or not isinstance(segmentation, dict):
        return None
    decoded = mask_utils.decode(segmentation)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return decoded > 0


def mask_contains(mask, point: tuple[float, float]) -> bool:
    if mask is None:
        return False
    x, y = point
    ix = int(x)
    iy = int(y)
    if iy < 0 or ix < 0 or iy >= mask.shape[0] or ix >= mask.shape[1]:
        return False
    return bool(mask[iy, ix])


def line_has_body_support(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    body: BodyRegion,
    samples: int = 5,
) -> bool:
    for idx in range(samples):
        t = idx / max(1, samples - 1)
        point = (
            point_a[0] * (1.0 - t) + point_b[0] * t,
            point_a[1] * (1.0 - t) + point_b[1] * t,
        )
        if body.mask is not None:
            if not mask_contains(body.mask, point):
                return False
        elif not point_in_bbox(point, body.bbox):
            return False
    return True


def build_annotations_by_image(annotations: list[dict]) -> dict[int, list[dict]]:
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in annotations:
        anns_by_image[int(ann['image_id'])].append(ann)
    return anns_by_image


def mark_isolated_body_regions(regions: list[BodyRegion]) -> None:
    for region in regions:
        region.isolated = True

    for index, first in enumerate(regions):
        for second in regions[index + 1:]:
            if bbox_intersection_area(first.bbox, second.bbox) > 0.0:
                first.isolated = False
                second.isolated = False


def build_body_regions(
    image_id: int,
    anns_by_image: dict[int, list[dict]],
    fallback_anns: list[dict],
) -> list[BodyRegion]:
    source_anns = anns_by_image.get(image_id) or fallback_anns
    regions: list[BodyRegion] = []
    for ann in source_anns:
        if int(ann.get('category_id', -1)) != BODY_CLASS_ID:
            continue
        bbox = tuple(float(v) for v in ann.get('bbox', [0, 0, 0, 0]))
        if bbox_area(bbox) <= 0:
            continue
        segmentation = ann.get('segmentation')
        mask = decode_mask(segmentation) if segmentation_is_non_empty(segmentation) else None
        regions.append(
            BodyRegion(
                index=len(regions),
                ann_id=int(ann.get('id', len(regions))),
                bbox=bbox,
                area=float(ann.get('area') or bbox_area(bbox)),
                mask=mask,
            )
        )
    mark_isolated_body_regions(regions)
    return regions


def collect_keypoints(
    image_annotations: list[dict],
    keypoint_ids: set[int],
    id_to_name: dict[int, str],
) -> list[Keypoint]:
    keypoints: list[Keypoint] = []
    for ann in image_annotations:
        category_id = int(ann.get('category_id', -1))
        if category_id not in keypoint_ids:
            continue
        bbox = tuple(float(v) for v in ann.get('bbox', [0, 0, 0, 0]))
        if bbox_area(bbox) <= 0:
            continue
        keypoints.append(
            Keypoint(
                ann_id=int(ann.get('id', len(keypoints))),
                category_id=category_id,
                name=id_to_name[category_id],
                bbox=bbox,
                center=bbox_center(bbox),
            )
        )
    return keypoints


def assign_keypoint(
    keypoint: Keypoint,
    body_regions: list[BodyRegion],
    bbox_expand_ratio: float,
) -> Assignment | None:
    mask_hits = [body for body in body_regions if mask_contains(body.mask, keypoint.center)]
    if mask_hits:
        body = min(mask_hits, key=lambda item: item.area)
        return Assignment(body.index, 'mask', 0.0)

    if len(body_regions) == 1:
        body = body_regions[0]
        return Assignment(body.index, 'single_body_fallback', bbox_distance_to_point(keypoint.center, body.bbox))

    isolated_bbox_hits = [
        body
        for body in body_regions
        if body.isolated and point_in_bbox(keypoint.center, expanded_bbox(body.bbox, bbox_expand_ratio))
    ]
    if isolated_bbox_hits:
        body = min(
            isolated_bbox_hits,
            key=lambda item: (
                bbox_distance_to_point(keypoint.center, item.bbox),
                item.area,
            ),
        )
        return Assignment(
            body.index,
            'isolated_bbox_fallback',
            bbox_distance_to_point(keypoint.center, body.bbox),
        )

    bbox_hits = [
        body
        for body in body_regions
        if point_in_bbox(keypoint.center, expanded_bbox(body.bbox, bbox_expand_ratio))
    ]
    if bbox_hits:
        body = min(
            bbox_hits,
            key=lambda item: (
                bbox_distance_to_point(keypoint.center, item.bbox),
                item.area,
            ),
        )
        return Assignment(body.index, 'bbox', bbox_distance_to_point(keypoint.center, body.bbox))
    return None


def build_bone_bbox(
    first: Keypoint,
    second: Keypoint,
    image_width: float,
    image_height: float,
    min_thickness: float,
    thickness_ratio: float,
    endpoint_padding_ratio: float,
) -> tuple[float, float, float, float] | None:
    x1, y1 = first.center
    x2, y2 = second.center
    distance = math.hypot(x2 - x1, y2 - y1)
    if distance <= 0:
        return None

    half_thickness = max(
        min_thickness / 2.0,
        distance * thickness_ratio / 2.0,
    )
    padding = half_thickness * (1.0 + max(0.0, endpoint_padding_ratio))
    left = min(x1, x2) - padding
    top = min(y1, y2) - padding
    right = max(x1, x2) + padding
    bottom = max(y1, y2) + padding
    return clamp_bbox((left, top, right - left, bottom - top), image_width, image_height)


def edge_candidate_is_valid(
    first: Keypoint,
    second: Keypoint,
    first_assignment: Assignment,
    second_assignment: Assignment,
    body_regions: list[BodyRegion],
    max_edge_distance_ratio: float,
) -> bool:
    if first_assignment.body_index != second_assignment.body_index:
        return False

    body = body_regions[first_assignment.body_index]
    distance = math.hypot(second.center[0] - first.center[0], second.center[1] - first.center[1])
    body_diag = math.hypot(body.bbox[2], body.bbox[3])
    if body_diag > 0 and distance > body_diag * max_edge_distance_ratio:
        return False

    if first_assignment.method == 'mask' and second_assignment.method == 'mask':
        return True

    if (
        first_assignment.method in MASK_BYPASS_ASSIGNMENT_METHODS
        or second_assignment.method in MASK_BYPASS_ASSIGNMENT_METHODS
    ):
        return True

    return line_has_body_support(first.center, second.center, body)


def generate_bones_for_image(
    image_info: dict,
    image_annotations: list[dict],
    ins_anns_by_image: dict[int, list[dict]],
    keypoint_ids: set[int],
    id_to_name: dict[int, str],
    args: argparse.Namespace,
) -> list[dict]:
    image_id = int(image_info['id'])
    image_width = float(image_info['width'])
    image_height = float(image_info['height'])
    body_regions = build_body_regions(image_id, ins_anns_by_image, image_annotations)
    if not body_regions:
        return []

    keypoints = collect_keypoints(image_annotations, keypoint_ids, id_to_name)
    if not keypoints:
        return []

    assignments: dict[int, Assignment] = {}
    by_body_and_name: dict[tuple[int, str], list[Keypoint]] = defaultdict(list)
    for keypoint in keypoints:
        assignment = assign_keypoint(keypoint, body_regions, args.body_bbox_expand_ratio)
        if assignment is None:
            continue
        assignments[keypoint.ann_id] = assignment
        by_body_and_name[(assignment.body_index, keypoint.name)].append(keypoint)

    generated: list[dict] = []
    seen_edges: set[tuple[int, str, str]] = set()
    for body in body_regions:
        for first_name, second_name in BONE_EDGES:
            first_candidates = by_body_and_name.get((body.index, first_name), [])
            second_candidates = by_body_and_name.get((body.index, second_name), [])
            if not first_candidates or not second_candidates:
                continue

            scored_pairs = []
            for first in first_candidates:
                for second in second_candidates:
                    first_assignment = assignments[first.ann_id]
                    second_assignment = assignments[second.ann_id]
                    if not edge_candidate_is_valid(
                        first,
                        second,
                        first_assignment,
                        second_assignment,
                        body_regions,
                        args.max_edge_distance_ratio,
                    ):
                        continue
                    method_bonus = 0.0
                    if first_assignment.method == 'mask' and second_assignment.method == 'mask':
                        method_bonus = -1_000_000.0
                    distance = math.hypot(second.center[0] - first.center[0], second.center[1] - first.center[1])
                    scored_pairs.append((method_bonus + distance, first, second))

            if not scored_pairs:
                continue

            _, first, second = min(scored_pairs, key=lambda item: item[0])
            edge_key = (body.index, first_name, second_name)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            bbox = build_bone_bbox(
                first,
                second,
                image_width,
                image_height,
                args.min_bone_thickness,
                args.bone_thickness_ratio,
                args.endpoint_padding_ratio,
            )
            if bbox is None:
                continue
            x, y, w, h = bbox
            generated.append(
                {
                    'image_id': image_id,
                    'category_id': BONE_CLASS_ID,
                    'bbox': [x, y, w, h],
                    'area': w * h,
                    'segmentation': [[x, y, x + w, y, x + w, y + h, x, y + h]],
                    'iscrowd': 0,
                    'bone_edge': [first_name, second_name],
                    'source_keypoint_ids': [first.ann_id, second.ann_id],
                    'body_annotation_id': body.ann_id,
                }
            )
    return generated


def selected_image_ids(dataset: dict, limit_images: int | None) -> set[int] | None:
    if limit_images is None:
        return None
    return {int(img['id']) for img in dataset.get('images', [])[:limit_images]}


def add_bones_to_coco_dataset(
    dataset: dict,
    ins_dataset: dict | None,
    args: argparse.Namespace,
) -> tuple[dict, dict[int, list[dict]]]:
    name_to_id, id_to_name = category_maps(dataset.get('categories', []))
    missing = [name for name in KEYPOINT_NAMES if name not in name_to_id]
    if missing:
        raise ValueError(f'Missing keypoint categories: {missing}')
    keypoint_ids = {name_to_id[name] for name in KEYPOINT_NAMES}

    limit_ids = selected_image_ids(dataset, args.limit_images)
    images = [
        img
        for img in dataset.get('images', [])
        if limit_ids is None or int(img['id']) in limit_ids
    ]
    image_ids = {int(img['id']) for img in images}
    base_annotations = [
        dict(ann)
        for ann in dataset.get('annotations', [])
        if int(ann['image_id']) in image_ids
    ]
    anns_by_image = build_annotations_by_image(base_annotations)
    ins_anns_by_image = (
        build_annotations_by_image(ins_dataset.get('annotations', []))
        if ins_dataset is not None
        else {}
    )

    bones_by_image: dict[int, list[dict]] = {}
    next_ann_id = max((int(ann.get('id', -1)) for ann in base_annotations), default=-1) + 1
    bone_count = 0
    for image_info in tqdm(images, desc='Generating bones', dynamic_ncols=True):
        image_id = int(image_info['id'])
        bones = generate_bones_for_image(
            image_info=image_info,
            image_annotations=anns_by_image.get(image_id, []),
            ins_anns_by_image=ins_anns_by_image,
            keypoint_ids=keypoint_ids,
            id_to_name=id_to_name,
            args=args,
        )
        for bone in bones:
            bone['id'] = next_ann_id
            next_ann_id += 1
        bones_by_image[image_id] = bones
        bone_count += len(bones)

    out_dataset = {
        key: value
        for key, value in dataset.items()
        if key not in {'images', 'annotations', 'categories'}
    }
    out_dataset['images'] = images
    out_dataset['annotations'] = base_annotations + [bone for bones in bones_by_image.values() for bone in bones]
    out_dataset['categories'] = ensure_bone_category(dataset.get('categories', []))
    print(f'generated bone annotations: {bone_count}')
    return out_dataset, bones_by_image


def write_yolo_labels(
    input_root: Path,
    output_root: Path,
    dataset: dict,
    bones_by_image: dict[int, list[dict]],
    overwrite: bool,
) -> None:
    labels_in = input_root / 'labels'
    labels_out = output_root / 'labels'
    labels_out.mkdir(parents=True, exist_ok=True)

    for image_info in tqdm(dataset.get('images', []), desc='Writing YOLO labels', dynamic_ncols=True):
        file_name = str(image_info['file_name'])
        stem = Path(file_name).stem
        src = labels_in / f'{stem}.txt'
        dst = labels_out / f'{stem}.txt'
        if dst.exists() and not overwrite:
            raise FileExistsError(f'Output label already exists: {dst}')

        original = src.read_text(encoding='utf-8').splitlines() if src.exists() else []
        lines = [line for line in original if line.strip()]
        image_width = float(image_info['width'])
        image_height = float(image_info['height'])
        for bone in bones_by_image.get(int(image_info['id']), []):
            x, y, w, h = [float(v) for v in bone['bbox']]
            xc = (x + w / 2.0) / image_width
            yc = (y + h / 2.0) / image_height
            nw = w / image_width
            nh = h / image_height
            lines.append(f'{BONE_CLASS_ID} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}')
        dst.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def prepare_output_root(input_root: Path, output_root: Path, args: argparse.Namespace) -> None:
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f'Output root already exists: {output_root}. Use --overwrite to replace files.')
    output_root.mkdir(parents=True, exist_ok=True)

    classes = load_classes(input_root / 'classes.txt')
    if len(classes) != BONE_CLASS_ID:
        raise ValueError(f'Expected {BONE_CLASS_ID} input classes, found {len(classes)}')
    if classes[-1] == BONE_CLASS_NAME:
        out_classes = classes
    else:
        out_classes = classes + [BONE_CLASS_NAME]
    (output_root / 'classes.txt').write_text('\n'.join(out_classes) + '\n', encoding='utf-8')

    for split_file in input_root.glob('*.txt'):
        if split_file.name == 'classes.txt':
            continue
        dst = output_root / split_file.name
        if dst.exists():
            dst.unlink()
        shutil.copy2(split_file, dst)

    images_in = input_root / 'images'
    images_out = output_root / 'images'
    if args.image_link_mode == 'none':
        return
    if images_out.exists() or images_out.is_symlink():
        if images_out.is_symlink() or images_out.is_file():
            images_out.unlink()
        elif args.overwrite:
            shutil.rmtree(images_out)
        else:
            raise FileExistsError(f'Output images path already exists: {images_out}')
    if args.image_link_mode == 'symlink':
        images_out.symlink_to(images_in.resolve(), target_is_directory=True)
    elif args.image_link_mode == 'copy':
        shutil.copytree(images_in, images_out)


def main() -> None:
    args = parse_args()
    input_root = args.input_root
    output_root = args.output_root
    annotations_in = input_root / 'annotations'

    if not input_root.is_dir():
        raise FileNotFoundError(f'Input dataset root not found: {input_root}')
    if mask_utils is None:
        print('warning: pycocotools is not available; RLE body masks will be ignored.')

    prepare_output_root(input_root, output_root, args)
    (output_root / 'annotations').mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        coco_path = annotations_in / f'{split}.json'
        ins_path = annotations_in / f'{split}_ins.json'
        if not coco_path.is_file():
            raise FileNotFoundError(f'COCO annotation not found: {coco_path}')
        print(f'processing split: {split}')
        dataset = load_json(coco_path)
        ins_dataset = load_json(ins_path) if ins_path.is_file() else None
        out_dataset, bones_by_image = add_bones_to_coco_dataset(dataset, ins_dataset, args)

        if 'coco' in args.formats:
            out_path = output_root / 'annotations' / f'{split}.json'
            if out_path.exists() and not args.overwrite:
                raise FileExistsError(f'Output annotation already exists: {out_path}')
            write_json(out_path, out_dataset, args.indent, progress=not args.no_json_progress)
            if args.write_ins_json and ins_dataset is not None:
                ins_out_dataset = dict(out_dataset)
                ins_base = load_json(ins_path)
                ins_base_annotations = [
                    dict(ann)
                    for ann in ins_base.get('annotations', [])
                    if int(ann['image_id']) in {int(img['id']) for img in out_dataset['images']}
                ]
                next_id = max((int(ann.get('id', -1)) for ann in ins_base_annotations), default=-1) + 1
                ins_bones = []
                for bones in bones_by_image.values():
                    for bone in bones:
                        ins_bone = dict(bone)
                        ins_bone['id'] = next_id
                        next_id += 1
                        ins_bones.append(ins_bone)
                ins_out_dataset['annotations'] = ins_base_annotations + ins_bones
                write_json(
                    output_root / 'annotations' / f'{split}_ins.json',
                    ins_out_dataset,
                    args.indent,
                    progress=not args.no_json_progress,
                )

        if 'yolo' in args.formats:
            write_yolo_labels(input_root, output_root, out_dataset, bones_by_image, args.overwrite)


if __name__ == '__main__':
    main()
