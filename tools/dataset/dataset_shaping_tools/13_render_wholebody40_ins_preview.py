"""
Render a small visual preview set from wholebody40 instance segmentation annotations.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
from tqdm import tqdm


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


def load_annotation(annotation_path: Path) -> dict:
    with annotation_path.open('r', encoding='utf-8') as f:
        return json.load(f)


def make_category_lookup(categories: List[dict]) -> Dict[int, str]:
    return {int(cat['id']): str(cat.get('name', cat['id'])) for cat in categories}


def make_category_colors(category_ids: List[int]) -> Dict[int, Tuple[int, int, int]]:
    palette = [
        '#ff6b6b', '#4ecdc4', '#ffe66d', '#1a535c', '#ff9f1c',
        '#5f0f40', '#9a031e', '#fb8b24', '#0f4c5c', '#2ec4b6',
        '#3a86ff', '#8338ec', '#ff006e', '#8ac926', '#1982c4',
        '#6a4c93', '#e76f51', '#2a9d8f', '#e9c46a', '#264653',
    ]
    colors: Dict[int, Tuple[int, int, int]] = {}
    for idx, category_id in enumerate(sorted(set(category_ids))):
        colors[category_id] = ImageColor.getrgb(palette[idx % len(palette)])
    return colors


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
        if not isinstance(segmentation, list) or len(segmentation) == 0:
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


def draw_polygon_overlay(
    base_image: Image.Image,
    annotations: List[dict],
    category_names: Dict[int, str],
    category_colors: Dict[int, Tuple[int, int, int]],
) -> Image.Image:
    image = base_image.convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    for ann in annotations:
        category_id = int(ann['category_id'])
        color = category_colors.get(category_id, (255, 0, 0))
        segmentation = ann.get('segmentation', [])
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 != 0:
                continue
            points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
            draw_overlay.polygon(points, fill=(*color, 96), outline=(*color, 220))

    rendered = Image.alpha_composite(image, overlay).convert('RGB')
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()

    for ann in annotations:
        category_id = int(ann['category_id'])
        color = category_colors.get(category_id, (255, 0, 0))
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

    used_category_ids = [
        int(ann['category_id'])
        for image in selected_images
        for ann in annotations_by_image_id[int(image['id'])]
    ]
    category_colors = make_category_colors(used_category_ids)

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
            rendered = draw_polygon_overlay(
                base_image=image,
                annotations=annotations_by_image_id[image_id],
                category_names=category_names,
                category_colors=category_colors,
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
