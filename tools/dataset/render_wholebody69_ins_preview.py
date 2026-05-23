"""
Render visual previews from wholebody69 annotations.

The wholebody69 hand keypoints are stored as small bbox annotations with empty
segmentations, so this renderer draws both segmentation masks and hand point
boxes/centers.
"""

import argparse
import colorsys
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

try:
    from pycocotools import mask as mask_utils
except Exception:  # pragma: no cover - optional dependency
    mask_utils = None


DEFAULT_ANN_JSON = 'wholebody69/annotations/train_ins.json'
DEFAULT_IMAGES_DIR = 'wholebody49/images'
DEFAULT_OUTPUT_DIR = 'wholebody69/val_ins_preview'
DEFAULT_LIMIT = 100
DEFAULT_HAND_CATEGORY_START = 49


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render wholebody69 masks and hand keypoint bbox annotations onto source images.',
    )
    parser.add_argument('--ann-json', type=Path, default=Path(DEFAULT_ANN_JSON))
    parser.add_argument('--images-dir', type=Path, default=Path(DEFAULT_IMAGES_DIR))
    parser.add_argument('--output-dir', type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--category-ids', type=int, nargs='*', default=None)
    parser.add_argument('--image-ids', type=int, nargs='*', default=None)
    parser.add_argument('--file-names', type=str, nargs='*', default=None)
    parser.add_argument('--hand-category-start', type=int, default=DEFAULT_HAND_CATEGORY_START)
    parser.add_argument('--include-segmentations', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--include-hand-keypoints', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        '--include-empty-boxes',
        action='store_true',
        help='Also draw non-hand annotations whose segmentation is empty.',
    )
    parser.add_argument('--draw-labels', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--draw-hand-labels', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--mask-alpha', type=int, default=96)
    parser.add_argument('--hand-radius', type=int, default=3)
    return parser.parse_args()


def require_mask_utils(context: str) -> None:
    if mask_utils is None:
        raise RuntimeError(f'pycocotools is required for {context}.')


def load_annotation(annotation_path: Path) -> dict:
    with annotation_path.open('r', encoding='utf-8') as f:
        return json.load(f)


def make_category_lookup(categories: List[dict]) -> Dict[int, str]:
    return {int(cat['id']): str(cat.get('name', cat['id'])) for cat in categories}


def make_instance_color(annotation_id: int) -> Tuple[int, int, int]:
    hue = ((int(annotation_id) * 137) % 360) / 360.0
    sat = 0.72
    val = 0.95
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return int(r * 255), int(g * 255), int(b * 255)


def make_hand_color(keypoint_index: int, side: Optional[str]) -> Tuple[int, int, int]:
    hue = ((int(keypoint_index) * 17) % 21) / 21.0
    sat = 0.78
    val = 0.98
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    if side == 'left':
        return int(r * 255), int(g * 220), int(b * 255)
    if side == 'right':
        return int(r * 220), int(g * 255), int(b * 255)
    return int(r * 255), int(g * 255), int(b * 255)


def segmentation_is_non_empty(segmentation) -> bool:
    if isinstance(segmentation, list):
        return len(segmentation) > 0
    if isinstance(segmentation, dict):
        return bool(segmentation.get('counts')) and bool(segmentation.get('size'))
    return False


def is_hand_annotation(annotation: dict, hand_category_start: int) -> bool:
    return (
        int(annotation.get('category_id', -1)) >= hand_category_start
        or 'hand_keypoint_index' in annotation
    )


def annotation_matches_filter(
    annotation: dict,
    allowed_categories: Optional[set],
    hand_category_start: int,
    include_segmentations: bool,
    include_hand_keypoints: bool,
    include_empty_boxes: bool,
) -> bool:
    if allowed_categories is not None and int(annotation.get('category_id', -1)) not in allowed_categories:
        return False

    is_hand = is_hand_annotation(annotation, hand_category_start)
    if include_hand_keypoints and is_hand:
        return True
    if include_segmentations and segmentation_is_non_empty(annotation.get('segmentation')):
        return True
    if include_empty_boxes:
        return True
    return False


def build_selection(
    dataset: dict,
    limit: int,
    shuffle: bool,
    seed: int,
    category_ids: Optional[List[int]],
    image_ids: Optional[List[int]],
    file_names: Optional[List[str]],
    hand_category_start: int,
    include_segmentations: bool,
    include_hand_keypoints: bool,
    include_empty_boxes: bool,
) -> Tuple[List[dict], Dict[int, List[dict]]]:
    allowed_image_ids = None if image_ids is None else set(int(image_id) for image_id in image_ids)
    allowed_file_names = None if file_names is None else set(file_names)
    allowed_categories = None if category_ids is None else set(int(category_id) for category_id in category_ids)

    annotations_by_image_id: Dict[int, List[dict]] = defaultdict(list)
    for ann in dataset.get('annotations', []):
        if not annotation_matches_filter(
            annotation=ann,
            allowed_categories=allowed_categories,
            hand_category_start=hand_category_start,
            include_segmentations=include_segmentations,
            include_hand_keypoints=include_hand_keypoints,
            include_empty_boxes=include_empty_boxes,
        ):
            continue
        annotations_by_image_id[int(ann['image_id'])].append(ann)

    selected_images = []
    for image in dataset.get('images', []):
        image_id = int(image['id'])
        file_name = str(image.get('file_name', ''))
        if image_id not in annotations_by_image_id:
            continue
        if allowed_image_ids is not None and image_id not in allowed_image_ids:
            continue
        if allowed_file_names is not None and file_name not in allowed_file_names:
            continue
        selected_images.append(image)

    selected_images.sort(key=lambda img: str(img.get('file_name', '')))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected_images)

    if limit > 0:
        selected_images = selected_images[:limit]

    return selected_images, annotations_by_image_id


def decode_segmentation_mask(segmentation: dict) -> np.ndarray:
    require_mask_utils('RLE preview rendering')
    decoded = mask_utils.decode(segmentation)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return (decoded > 0).astype(np.uint8)


def apply_rle_overlay(
    overlay: Image.Image,
    segmentation: dict,
    color: Tuple[int, int, int],
    alpha: int,
) -> Image.Image:
    mask = decode_segmentation_mask(segmentation)
    layer = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    layer[mask > 0] = (*color, alpha)
    return Image.alpha_composite(overlay, Image.fromarray(layer, mode='RGBA'))


def draw_segmentation(
    overlay: Image.Image,
    annotation: dict,
    color: Tuple[int, int, int],
    mask_alpha: int,
) -> Image.Image:
    draw_overlay = ImageDraw.Draw(overlay)
    segmentation = annotation.get('segmentation', [])
    if isinstance(segmentation, list):
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 != 0:
                continue
            points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
            draw_overlay.polygon(points, fill=(*color, mask_alpha), outline=(*color, 220))
    elif isinstance(segmentation, dict):
        overlay = apply_rle_overlay(overlay, segmentation, color, mask_alpha)
    return overlay


def bbox_xyxy(annotation: dict) -> Tuple[int, int, int, int]:
    x, y, w, h = annotation.get('bbox', [0, 0, 0, 0])
    return (
        int(round(float(x))),
        int(round(float(y))),
        int(round(float(x) + float(w))),
        int(round(float(y) + float(h))),
    )


def safe_label_position(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    image_size: Tuple[int, int],
    x1: int,
    y1: int,
    label_text: str,
) -> Tuple[int, int, int, int, int, int]:
    text_bbox = draw.textbbox((x1, y1), label_text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    tx = min(max(0, x1), max(0, image_size[0] - text_w - 6))
    ty = max(0, y1 - text_h - 4)
    if ty == 0:
        ty = min(max(0, y1 + 2), max(0, image_size[1] - text_h - 4))
    return tx, ty, tx + text_w + 6, ty + text_h + 4, text_w, text_h


def draw_box_label(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    image_size: Tuple[int, int],
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    label_text: str,
) -> None:
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    tx1, ty1, tx2, ty2, _, _ = safe_label_position(draw, font, image_size, x1, y1, label_text)
    draw.rectangle([tx1, ty1, tx2, ty2], fill=color)
    draw.text((tx1 + 3, ty1 + 2), label_text, fill='black', font=font)


def draw_hand_keypoint(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    image_size: Tuple[int, int],
    annotation: dict,
    category_names: Dict[int, str],
    hand_category_start: int,
    radius: int,
    draw_label: bool,
) -> None:
    category_id = int(annotation['category_id'])
    keypoint_index = int(annotation.get('hand_keypoint_index', category_id - hand_category_start))
    side = annotation.get('hand_side')
    color = make_hand_color(keypoint_index, side)
    x1, y1, x2, y2 = bbox_xyxy(annotation)
    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))

    draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color, outline=(0, 0, 0), width=1)

    if draw_label:
        prefix = 'L' if side == 'left' else 'R' if side == 'right' else '?'
        label_text = f'{prefix}{keypoint_index}:{category_names.get(category_id, category_id)}'
        tx1, ty1, tx2, ty2, _, _ = safe_label_position(draw, font, image_size, x2 + 2, y1, label_text)
        draw.rectangle([tx1, ty1, tx2, ty2], fill=color)
        draw.text((tx1 + 3, ty1 + 2), label_text, fill='black', font=font)


def draw_summary(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    image_size: Tuple[int, int],
    image_info: dict,
    annotations: List[dict],
    hand_category_start: int,
) -> None:
    total = len(annotations)
    hand_count = sum(1 for ann in annotations if is_hand_annotation(ann, hand_category_start))
    segm_count = sum(1 for ann in annotations if segmentation_is_non_empty(ann.get('segmentation')))
    side_counts = Counter(ann.get('hand_side') for ann in annotations if is_hand_annotation(ann, hand_category_start))
    summary = (
        f"id={image_info.get('id')}  ann={total}  segm={segm_count}  "
        f"hand={hand_count}  L={side_counts.get('left', 0)} R={side_counts.get('right', 0)}"
    )
    text_bbox = draw.textbbox((8, 8), summary, font=font)
    x2 = min(image_size[0] - 1, text_bbox[2] + 10)
    draw.rectangle([6, 6, x2, text_bbox[3] + 10], fill=(0, 0, 0))
    draw.text((8, 8), summary, fill='white', font=font)


def draw_preview(
    base_image: Image.Image,
    image_info: dict,
    annotations: List[dict],
    category_names: Dict[int, str],
    hand_category_start: int,
    draw_labels: bool,
    draw_hand_labels: bool,
    mask_alpha: int,
    hand_radius: int,
) -> Image.Image:
    image = base_image.convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))

    for ann in annotations:
        if not segmentation_is_non_empty(ann.get('segmentation')):
            continue
        color = make_instance_color(int(ann['id']))
        overlay = draw_segmentation(overlay, ann, color, mask_alpha)

    rendered = Image.alpha_composite(image, overlay).convert('RGB')
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()

    for ann in annotations:
        if is_hand_annotation(ann, hand_category_start):
            continue
        if not segmentation_is_non_empty(ann.get('segmentation')):
            continue
        category_id = int(ann['category_id'])
        label_text = f"{category_names.get(category_id, str(category_id))} id={ann.get('id')}"
        color = make_instance_color(int(ann['id']))
        if draw_labels:
            draw_box_label(draw, font, rendered.size, bbox_xyxy(ann), color, label_text)
        else:
            draw.rectangle(bbox_xyxy(ann), outline=color, width=2)

    for ann in annotations:
        if not is_hand_annotation(ann, hand_category_start):
            continue
        draw_hand_keypoint(
            draw=draw,
            font=font,
            image_size=rendered.size,
            annotation=ann,
            category_names=category_names,
            hand_category_start=hand_category_start,
            radius=hand_radius,
            draw_label=draw_hand_labels,
        )

    draw_summary(draw, font, rendered.size, image_info, annotations, hand_category_start)
    return rendered


def image_manifest_record(image_info: dict, annotations: List[dict], hand_category_start: int) -> dict:
    hand_annotations = [ann for ann in annotations if is_hand_annotation(ann, hand_category_start)]
    segm_annotations = [ann for ann in annotations if segmentation_is_non_empty(ann.get('segmentation'))]
    return {
        'image_id': int(image_info['id']),
        'file_name': str(image_info['file_name']),
        'num_annotations': len(annotations),
        'num_segmentations': len(segm_annotations),
        'num_hand_keypoints': len(hand_annotations),
        'num_left_hand_keypoints': sum(1 for ann in hand_annotations if ann.get('hand_side') == 'left'),
        'num_right_hand_keypoints': sum(1 for ann in hand_annotations if ann.get('hand_side') == 'right'),
    }


def render_preview(
    ann_json: Path,
    images_dir: Path,
    output_dir: Path,
    limit: int,
    shuffle: bool,
    seed: int,
    category_ids: Optional[List[int]],
    image_ids: Optional[List[int]],
    file_names: Optional[List[str]],
    hand_category_start: int,
    include_segmentations: bool,
    include_hand_keypoints: bool,
    include_empty_boxes: bool,
    draw_labels: bool,
    draw_hand_labels: bool,
    mask_alpha: int,
    hand_radius: int,
) -> None:
    if not ann_json.exists():
        raise FileNotFoundError(f'Annotation file not found: {ann_json}')
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f'Image directory not found: {images_dir}')

    dataset = load_annotation(ann_json)
    category_names = make_category_lookup(dataset.get('categories', []))
    selected_images, annotations_by_image_id = build_selection(
        dataset=dataset,
        limit=limit,
        shuffle=shuffle,
        seed=seed,
        category_ids=category_ids,
        image_ids=image_ids,
        file_names=file_names,
        hand_category_start=hand_category_start,
        include_segmentations=include_segmentations,
        include_hand_keypoints=include_hand_keypoints,
        include_empty_boxes=include_empty_boxes,
    )

    if not selected_images:
        raise ValueError('No images matched the requested preview filters.')

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[dict] = []
    missing_images: List[str] = []

    for image_info in tqdm(selected_images, desc='Rendering wholebody69 preview', dynamic_ncols=True):
        image_id = int(image_info['id'])
        file_name = str(image_info['file_name'])
        image_path = images_dir / file_name
        if not image_path.exists():
            missing_images.append(file_name)
            continue

        annotations = annotations_by_image_id[image_id]
        with Image.open(image_path) as image:
            rendered = draw_preview(
                base_image=image,
                image_info=image_info,
                annotations=annotations,
                category_names=category_names,
                hand_category_start=hand_category_start,
                draw_labels=draw_labels,
                draw_hand_labels=draw_hand_labels,
                mask_alpha=mask_alpha,
                hand_radius=hand_radius,
            )
            rendered.save(output_dir / file_name)

        manifest.append(image_manifest_record(image_info, annotations, hand_category_start))

    manifest_path = output_dir / 'render_manifest.json'
    with manifest_path.open('w', encoding='utf-8') as f:
        json.dump(
            {
                'annotation_file': str(ann_json),
                'images_dir': str(images_dir),
                'output_dir': str(output_dir),
                'requested_limit': limit,
                'rendered_images': len(manifest),
                'missing_images': missing_images,
                'category_ids': category_ids,
                'image_ids': image_ids,
                'file_names': file_names,
                'hand_category_start': hand_category_start,
                'include_segmentations': include_segmentations,
                'include_hand_keypoints': include_hand_keypoints,
                'include_empty_boxes': include_empty_boxes,
                'draw_labels': draw_labels,
                'draw_hand_labels': draw_hand_labels,
                'images': manifest,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f'Rendered {len(manifest)} images to {output_dir}')
    if missing_images:
        print(f'Skipped {len(missing_images)} missing images. See {manifest_path} for details.')


def main() -> None:
    args = parse_args()
    render_preview(
        ann_json=args.ann_json,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
        category_ids=args.category_ids,
        image_ids=args.image_ids,
        file_names=args.file_names,
        hand_category_start=args.hand_category_start,
        include_segmentations=args.include_segmentations,
        include_hand_keypoints=args.include_hand_keypoints,
        include_empty_boxes=args.include_empty_boxes,
        draw_labels=args.draw_labels,
        draw_hand_labels=args.draw_hand_labels,
        mask_alpha=args.mask_alpha,
        hand_radius=args.hand_radius,
    )


if __name__ == '__main__':
    main()
