#!/usr/bin/env python3
"""Create a fine-tuning split for wholebody49_coco.

The default split keeps the first 15 images from the existing validation set
as validation and moves the remaining validation images back into training.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path('tools/dataset/wholebody49_coco')
DEFAULT_OUTPUT_ROOT = Path('tools/dataset/wholebody49_coco_ft')
DEFAULT_VAL_COUNT = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate tools/dataset/wholebody49_coco_ft from wholebody49_coco.'
    )
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--val-count', type=int, default=DEFAULT_VAL_COUNT)
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Remove an existing output root before generating the dataset.',
    )
    return parser.parse_args()


def read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_split(path: Path, names: list[str]) -> None:
    path.write_text(''.join(f'{name}\n' for name in names), encoding='utf-8')


def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def dump_json(path: Path, dataset: dict) -> None:
    with path.open('w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False)


def validate_source(source_root: Path) -> None:
    required = [
        source_root / 'train.txt',
        source_root / 'val.txt',
        source_root / 'classes.txt',
        source_root / 'images',
        source_root / 'labels',
        source_root / 'annotations' / 'train.json',
        source_root / 'annotations' / 'val.json',
        source_root / 'annotations' / 'train_ins.json',
        source_root / 'annotations' / 'val_ins.json',
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing required source paths:\n' + '\n'.join(str(path) for path in missing))


def prepare_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f'Output root already exists: {output_root}. Use --overwrite to replace it.')
        shutil.rmtree(output_root)

    (output_root / 'images').mkdir(parents=True)
    (output_root / 'labels').mkdir(parents=True)
    (output_root / 'annotations').mkdir(parents=True)


def build_splits(source_root: Path, val_count: int) -> tuple[list[str], list[str]]:
    train_names = read_split(source_root / 'train.txt')
    val_names = read_split(source_root / 'val.txt')

    if val_count <= 0:
        raise ValueError('--val-count must be positive')
    if val_count >= len(val_names):
        raise ValueError(f'--val-count must be smaller than existing val size ({len(val_names)})')

    new_val_names = val_names[:val_count]
    new_train_names = train_names + val_names[val_count:]

    train_set = set(new_train_names)
    val_set = set(new_val_names)
    if len(train_set) != len(new_train_names):
        raise ValueError('New train split contains duplicate file names')
    if len(val_set) != len(new_val_names):
        raise ValueError('New val split contains duplicate file names')
    overlap = train_set & val_set
    if overlap:
        raise ValueError(f'New train/val split overlaps: {sorted(overlap)[:5]}')

    return new_train_names, new_val_names


def group_annotations_by_image(dataset: dict) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for ann in dataset.get('annotations', []):
        grouped[int(ann['image_id'])].append(ann)
    return grouped


def make_subset_dataset(
    template: dict,
    source_datasets: list[dict],
    split_names: list[str],
) -> dict:
    images_by_name = {}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)

    for dataset in source_datasets:
        for image in dataset.get('images', []):
            images_by_name[image['file_name']] = image
        grouped = group_annotations_by_image(dataset)
        for image_id, annotations in grouped.items():
            annotations_by_image[image_id].extend(annotations)

    selected_images = []
    selected_annotations = []
    next_ann_id = 0

    for file_name in split_names:
        if file_name not in images_by_name:
            raise KeyError(f'Image {file_name!r} not found in source annotations')

        image = dict(images_by_name[file_name])
        selected_images.append(image)
        for ann in annotations_by_image.get(int(image['id']), []):
            new_ann = dict(ann)
            new_ann['id'] = next_ann_id
            next_ann_id += 1
            selected_annotations.append(new_ann)

    subset = {key: value for key, value in template.items() if key not in {'images', 'annotations'}}
    subset['images'] = selected_images
    subset['annotations'] = selected_annotations
    return subset


def write_annotation_pair(source_root: Path, output_root: Path, stem: str, train_names: list[str], val_names: list[str]) -> None:
    annotations_dir = source_root / 'annotations'
    output_annotations_dir = output_root / 'annotations'

    train_dataset = load_json(annotations_dir / f'train{stem}.json')
    val_dataset = load_json(annotations_dir / f'val{stem}.json')

    train_subset = make_subset_dataset(train_dataset, [train_dataset, val_dataset], train_names)
    val_subset = make_subset_dataset(val_dataset, [val_dataset], val_names)

    dump_json(output_annotations_dir / f'train{stem}.json', train_subset)
    dump_json(output_annotations_dir / f'val{stem}.json', val_subset)

    print(
        f'wrote train{stem}.json: images={len(train_subset["images"])} '
        f'annotations={len(train_subset["annotations"])}'
    )
    print(
        f'wrote val{stem}.json: images={len(val_subset["images"])} '
        f'annotations={len(val_subset["annotations"])}'
    )


def copy_dataset_files(source_root: Path, output_root: Path, all_names: list[str]) -> None:
    shutil.copy2(source_root / 'classes.txt', output_root / 'classes.txt')

    for index, file_name in enumerate(all_names, start=1):
        src_image = source_root / 'images' / file_name
        src_label = source_root / 'labels' / f'{Path(file_name).stem}.txt'
        if not src_image.is_file():
            raise FileNotFoundError(f'Missing image file: {src_image}')
        if not src_label.is_file():
            raise FileNotFoundError(f'Missing label file: {src_label}')

        shutil.copy2(src_image, output_root / 'images' / file_name)
        shutil.copy2(src_label, output_root / 'labels' / src_label.name)

        if index % 1000 == 0 or index == len(all_names):
            print(f'copied {index}/{len(all_names)} image/label pairs')


def verify_output(output_root: Path, train_names: list[str], val_names: list[str]) -> None:
    output_train_names = read_split(output_root / 'train.txt')
    output_val_names = read_split(output_root / 'val.txt')
    if output_train_names != train_names:
        raise AssertionError('train.txt content mismatch')
    if output_val_names != val_names:
        raise AssertionError('val.txt content mismatch')

    train_set = set(output_train_names)
    val_set = set(output_val_names)
    if len(output_train_names) != 15013:
        raise AssertionError(f'Expected 15013 train images, got {len(output_train_names)}')
    if len(output_val_names) != 15:
        raise AssertionError(f'Expected 15 val images, got {len(output_val_names)}')
    if len(train_set | val_set) != 15028:
        raise AssertionError(f'Expected 15028 unique images, got {len(train_set | val_set)}')
    if train_set & val_set:
        raise AssertionError('train/val splits overlap')

    for split in ('train', 'val'):
        expected_names = set(output_train_names if split == 'train' else output_val_names)
        expected_image_count = len(expected_names)
        for stem in ('', '_ins'):
            dataset = load_json(output_root / 'annotations' / f'{split}{stem}.json')
            images = dataset.get('images', [])
            annotations = dataset.get('annotations', [])
            image_ids = [image['id'] for image in images]
            ann_ids = [ann['id'] for ann in annotations]
            image_id_set = set(image_ids)

            if len(images) != expected_image_count:
                raise AssertionError(f'{split}{stem}.json image count mismatch: {len(images)}')
            if {image['file_name'] for image in images} != expected_names:
                raise AssertionError(f'{split}{stem}.json file_name set mismatch')
            if len(image_ids) != len(image_id_set):
                raise AssertionError(f'{split}{stem}.json duplicate image ids')
            if len(ann_ids) != len(set(ann_ids)):
                raise AssertionError(f'{split}{stem}.json duplicate annotation ids')
            if any(int(ann['image_id']) not in image_id_set for ann in annotations):
                raise AssertionError(f'{split}{stem}.json contains annotation with unknown image_id')

    for file_name in output_train_names + output_val_names:
        image_path = output_root / 'images' / file_name
        label_path = output_root / 'labels' / f'{Path(file_name).stem}.txt'
        if not image_path.is_file():
            raise AssertionError(f'Missing copied image: {image_path}')
        if not label_path.is_file():
            raise AssertionError(f'Missing copied label: {label_path}')

    print('verification passed')


def main() -> None:
    args = parse_args()
    source_root = args.source_root
    output_root = args.output_root

    validate_source(source_root)
    train_names, val_names = build_splits(source_root, args.val_count)

    prepare_output(output_root, args.overwrite)
    write_split(output_root / 'train.txt', train_names)
    write_split(output_root / 'val.txt', val_names)

    write_annotation_pair(source_root, output_root, '', train_names, val_names)
    write_annotation_pair(source_root, output_root, '_ins', train_names, val_names)
    copy_dataset_files(source_root, output_root, train_names + val_names)
    verify_output(output_root, train_names, val_names)

    print(f'created {output_root}')


if __name__ == '__main__':
    main()
