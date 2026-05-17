#!/usr/bin/env python3
"""Render wholebody keypoint boxes and generated bone boxes for visual checks."""

from __future__ import annotations

import argparse
import colorsys
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


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
)
BONE_CLASS_ID = 48
BONE_CLASS_NAME = 'bone'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render only human keypoint annotations and generated bone annotations.',
    )
    parser.add_argument('--dataset-root', type=Path, default=Path('tools/dataset/wholebody49'))
    parser.add_argument('--split', default='val')
    parser.add_argument(
        '--format',
        choices=('coco', 'yolo'),
        default='coco',
        help='Annotation source to render.',
    )
    parser.add_argument('--ann-json', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--draw-labels', action='store_true')
    parser.add_argument(
        '--image-list',
        type=Path,
        default=None,
        help='Optional txt file with image filenames. Defaults to {dataset-root}/{split}.txt.',
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def class_color(category_id: int) -> tuple[int, int, int]:
    if category_id == BONE_CLASS_ID:
        return 255, 140, 0
    hue = ((category_id * 137) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.78, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def category_maps(categories: list[dict]) -> tuple[dict[str, int], dict[int, str]]:
    name_to_id = {str(cat['name']): int(cat['id']) for cat in categories}
    id_to_name = {int(cat['id']): str(cat['name']) for cat in categories}
    return name_to_id, id_to_name


def target_category_ids(name_to_id: dict[str, int]) -> set[int]:
    missing = [name for name in KEYPOINT_NAMES if name not in name_to_id]
    if missing:
        raise ValueError(f'Missing keypoint classes: {missing}')
    return {name_to_id[name] for name in KEYPOINT_NAMES} | {BONE_CLASS_ID}


def build_coco_records(dataset: dict) -> tuple[list[dict], dict[int, list[dict]], dict[int, str]]:
    name_to_id, id_to_name = category_maps(dataset.get('categories', []))
    targets = target_category_ids(name_to_id)
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in dataset.get('annotations', []):
        category_id = int(ann.get('category_id', -1))
        if category_id in targets:
            annotations_by_image[int(ann['image_id'])].append(ann)
    return dataset.get('images', []), annotations_by_image, id_to_name


def read_image_list(path: Path | None) -> list[str] | None:
    if path is None or not path.is_file():
        return None
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def yolo_bbox_to_coco(parts: list[str], image_width: int, image_height: int) -> dict:
    category_id = int(parts[0])
    xc, yc, w, h = [float(v) for v in parts[1:5]]
    box_w = w * image_width
    box_h = h * image_height
    x = xc * image_width - box_w / 2.0
    y = yc * image_height - box_h / 2.0
    return {
        'category_id': category_id,
        'bbox': [x, y, box_w, box_h],
        'id': -1,
    }


def build_yolo_records(
    dataset_root: Path,
    split: str,
    image_list_file: Path | None,
) -> tuple[list[dict], dict[int, list[dict]], dict[int, str]]:
    classes = load_classes(dataset_root / 'classes.txt')
    name_to_id = {name: idx for idx, name in enumerate(classes)}
    id_to_name = {idx: name for idx, name in enumerate(classes)}
    targets = target_category_ids(name_to_id)
    images_dir = dataset_root / 'images'
    labels_dir = dataset_root / 'labels'
    image_names = read_image_list(image_list_file or (dataset_root / f'{split}.txt'))
    if image_names is None:
        image_names = sorted(path.name for path in images_dir.iterdir() if path.suffix.lower() in {'.jpg', '.jpeg', '.png'})

    images: list[dict] = []
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for image_id, file_name in enumerate(image_names):
        image_path = images_dir / file_name
        if not image_path.is_file():
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        images.append({'id': image_id, 'file_name': file_name, 'width': width, 'height': height})
        label_path = labels_dir / f'{Path(file_name).stem}.txt'
        if not label_path.is_file():
            continue
        for row in label_path.read_text(encoding='utf-8').splitlines():
            parts = row.strip().split()
            if len(parts) < 5:
                continue
            category_id = int(parts[0])
            if category_id not in targets:
                continue
            annotations_by_image[image_id].append(yolo_bbox_to_coco(parts, width, height))
    return images, annotations_by_image, id_to_name


def select_images(
    images: list[dict],
    annotations_by_image: dict[int, list[dict]],
    limit: int,
    shuffle: bool,
    seed: int,
) -> list[dict]:
    selected = [img for img in images if annotations_by_image.get(int(img['id']))]
    selected.sort(key=lambda img: str(img.get('file_name', '')))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected)
    if limit > 0:
        selected = selected[:limit]
    return selected


def draw_annotation(
    image: Image.Image,
    annotation: dict,
    id_to_name: dict[int, str],
    font: ImageFont.ImageFont,
    draw_labels: bool,
) -> None:
    draw = ImageDraw.Draw(image, 'RGBA')
    category_id = int(annotation['category_id'])
    color = class_color(category_id)
    x, y, w, h = [float(v) for v in annotation['bbox']]
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))

    if category_id == BONE_CLASS_ID:
        draw.rectangle([x1, y1, x2, y2], fill=(*color, 56), outline=(*color, 235), width=3)
    else:
        draw.rectangle([x1, y1, x2, y2], outline=(*color, 245), width=2)
        cx = int(round(x + w / 2.0))
        cy = int(round(y + h / 2.0))
        r = 3
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*color, 255), outline=(0, 0, 0, 220))

    if not draw_labels:
        return
    text = id_to_name.get(category_id, str(category_id))
    text_box = draw.textbbox((x1, y1), text, font=font)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    label_y = max(0, y1 - text_h - 4)
    draw.rectangle([x1, label_y, x1 + text_w + 6, label_y + text_h + 4], fill=(*color, 220))
    draw.text((x1 + 3, label_y + 2), text, fill=(0, 0, 0, 255), font=font)


def render_preview(
    dataset_root: Path,
    images: list[dict],
    annotations_by_image: dict[int, list[dict]],
    id_to_name: dict[int, str],
    output_dir: Path,
    limit: int,
    shuffle: bool,
    seed: int,
    draw_labels: bool,
) -> None:
    images_dir = dataset_root / 'images'
    selected_images = select_images(images, annotations_by_image, limit, shuffle, seed)
    if not selected_images:
        raise ValueError('No images with keypoint or bone annotations were found.')

    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    manifest = []
    for image_info in tqdm(selected_images, desc='Rendering bone preview', dynamic_ncols=True):
        image_id = int(image_info['id'])
        file_name = str(image_info['file_name'])
        image_path = images_dir / file_name
        if not image_path.is_file():
            continue
        with Image.open(image_path) as source:
            rendered = source.convert('RGB')
        annotations = sorted(
            annotations_by_image.get(image_id, []),
            key=lambda ann: 0 if int(ann.get('category_id', -1)) == BONE_CLASS_ID else 1,
        )
        for annotation in annotations:
            draw_annotation(rendered, annotation, id_to_name, font, draw_labels)

        summary = f"kp+bone: {len(annotations)}  bones: {sum(1 for ann in annotations if int(ann['category_id']) == BONE_CLASS_ID)}"
        draw = ImageDraw.Draw(rendered)
        text_box = draw.textbbox((8, 8), summary, font=font)
        draw.rectangle([6, 6, text_box[2] + 10, text_box[3] + 10], fill=(0, 0, 0))
        draw.text((8, 8), summary, fill=(255, 255, 255), font=font)

        out_name = f'{Path(file_name).stem}_bones_preview.jpg'
        rendered.save(output_dir / out_name, quality=92)
        manifest.append({'image': file_name, 'output': out_name, 'annotations': len(annotations)})

    with (output_dir / 'manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    output_dir = args.output_dir or (dataset_root / f'{args.split}_bones_preview')

    if args.format == 'coco':
        ann_json = args.ann_json or (dataset_root / 'annotations' / f'{args.split}.json')
        dataset = load_json(ann_json)
        images, annotations_by_image, id_to_name = build_coco_records(dataset)
    else:
        images, annotations_by_image, id_to_name = build_yolo_records(
            dataset_root,
            args.split,
            args.image_list,
        )

    render_preview(
        dataset_root=dataset_root,
        images=images,
        annotations_by_image=annotations_by_image,
        id_to_name=id_to_name,
        output_dir=output_dir,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
        draw_labels=args.draw_labels,
    )


if __name__ == '__main__':
    main()
