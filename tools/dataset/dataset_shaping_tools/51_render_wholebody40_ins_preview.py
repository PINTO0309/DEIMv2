"""
Render a small visual preview set from wholebody40 instance segmentation annotations.
"""

import argparse
import colorsys
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

try:
    from pycocotools import mask as mask_utils
except Exception:  # pragma: no cover - optional dependency
    mask_utils = None


DEFAULT_ANN_JSON = '/media/b920405/ExtremeSSD/make_wholebody40/wholebody40/annotations/val_ins.json'
DEFAULT_IMAGES_DIR = '/media/b920405/ExtremeSSD/make_wholebody40/wholebody40/images'
DEFAULT_OUTPUT_DIR = '/media/b920405/ExtremeSSD/make_wholebody40/wholebody40/val_ins_preview'
DEFAULT_LIMIT = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render wholebody40 instance segmentation annotations onto source images.',
    )
    parser.add_argument('--ann-json', type=str, default=DEFAULT_ANN_JSON)
    parser.add_argument('--images-dir', type=str, default=DEFAULT_IMAGES_DIR)
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument(
        '--category-ids',
        type=int,
        nargs='*',
        default=None,
        help='Optional category filter. If omitted, all non-empty segmentations are rendered.',
    )
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
    # Spread instance colors across the hue wheel while keeping saturation/value readable.
    hue = ((int(annotation_id) * 137) % 360) / 360.0
    sat = 0.72
    val = 0.95
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(r * 255), int(g * 255), int(b * 255))


def segmentation_is_non_empty(segmentation) -> bool:
    if isinstance(segmentation, list):
        return len(segmentation) > 0
    if isinstance(segmentation, dict):
        return bool(segmentation.get('counts')) and bool(segmentation.get('size'))
    return False


def build_selection(
    dataset: dict,
    limit: int,
    shuffle: bool,
    seed: int,
    category_ids: List[int] | None,
) -> Tuple[List[dict], Dict[int, List[dict]]]:
    images = dataset.get('images', [])
    annotations = dataset.get('annotations', [])

    allowed_categories = None if category_ids is None else set(category_ids)
    annotations_by_image_id: Dict[int, List[dict]] = defaultdict(list)
    for ann in annotations:
        segmentation = ann.get('segmentation')
        if not segmentation_is_non_empty(segmentation):
            continue
        if allowed_categories is not None and ann.get('category_id') not in allowed_categories:
            continue
        annotations_by_image_id[int(ann['image_id'])].append(ann)

    selected_images = [img for img in images if int(img['id']) in annotations_by_image_id]
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
    alpha: int = 96,
) -> Image.Image:
    mask = decode_segmentation_mask(segmentation)
    layer = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    layer[mask > 0] = (*color, alpha)
    return Image.alpha_composite(overlay, Image.fromarray(layer, mode='RGBA'))


def draw_segmentation_overlay(
    base_image: Image.Image,
    annotations: List[dict],
    category_names: Dict[int, str],
) -> Image.Image:
    image = base_image.convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    for ann in annotations:
        category_id = int(ann['category_id'])
        color = make_instance_color(int(ann['id']))
        segmentation = ann.get('segmentation', [])
        if isinstance(segmentation, list):
            for polygon in segmentation:
                if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 != 0:
                    continue
                points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
                draw_overlay.polygon(points, fill=(*color, 96), outline=(*color, 220))
        elif isinstance(segmentation, dict):
            overlay = apply_rle_overlay(overlay, segmentation, color)

    rendered = Image.alpha_composite(image, overlay).convert('RGB')
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()

    for ann in annotations:
        category_id = int(ann['category_id'])
        color = make_instance_color(int(ann['id']))
        x, y, w, h = ann.get('bbox', [0, 0, 0, 0])
        x1 = int(round(float(x)))
        y1 = int(round(float(y)))
        x2 = int(round(float(x + w)))
        y2 = int(round(float(y + h)))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        label_text = f"{category_names.get(category_id, str(category_id))} id={ann.get('id')}"
        text_bbox = draw.textbbox((x1, y1), label_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        bg_y1 = max(0, y1 - text_h - 4)
        bg_y2 = bg_y1 + text_h + 4
        draw.rectangle([x1, bg_y1, x1 + text_w + 6, bg_y2], fill=color)
        draw.text((x1 + 3, bg_y1 + 2), label_text, fill='black', font=font)

    summary_text = f'segm instances: {len(annotations)}'
    text_bbox = draw.textbbox((8, 8), summary_text, font=font)
    draw.rectangle(
        [6, 6, text_bbox[2] + 10, text_bbox[3] + 10],
        fill=(0, 0, 0),
    )
    draw.text((8, 8), summary_text, fill='white', font=font)
    return rendered


def render_preview(
    ann_json: Path,
    images_dir: Path,
    output_dir: Path,
    limit: int,
    shuffle: bool,
    seed: int,
    category_ids: List[int] | None,
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
    )

    if not selected_images:
        raise ValueError('No images with non-empty segmentation were found for the requested filter.')

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[dict] = []
    missing_images: List[str] = []

    for image_info in tqdm(selected_images, desc='Rendering preview', dynamic_ncols=True):
        image_id = int(image_info['id'])
        file_name = str(image_info['file_name'])
        image_path = images_dir / file_name
        if not image_path.exists():
            missing_images.append(file_name)
            continue

        with Image.open(image_path) as image:
            rendered = draw_segmentation_overlay(
                base_image=image,
                annotations=annotations_by_image_id[image_id],
                category_names=category_names,
            )
            rendered.save(output_dir / file_name)

        manifest.append(
            {
                'image_id': image_id,
                'file_name': file_name,
                'num_segmentations': len(annotations_by_image_id[image_id]),
            }
        )

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
        ann_json=Path(args.ann_json),
        images_dir=Path(args.images_dir),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
        category_ids=args.category_ids,
    )


if __name__ == '__main__':
    main()
