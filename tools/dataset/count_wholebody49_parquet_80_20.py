#!/usr/bin/env python3
"""Print wholebody49 Parquet label counts after an 80:20 image split."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
RESET = '\033[0m'

CATEGORIES_METADATA_KEY = b'coco_categories_json'
DEFAULT_ANNOTATIONS_DIR = Path('tools/dataset/wholebody49/annotations')


def require_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            'pyarrow is required to read Parquet files. '
            'Install it with `pip install pyarrow` or sync the project dependencies.'
        ) from exc
    return pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Count wholebody49 labels from train/val Parquet files after an 80:20 split.',
    )
    parser.add_argument(
        '--annotations-dir',
        type=Path,
        default=DEFAULT_ANNOTATIONS_DIR,
        help='Directory containing train.parquet and val.parquet.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=309,
        help='Random seed used before splitting image rows.',
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Train split ratio. Validation receives the remaining rows.',
    )
    return parser.parse_args()


def load_categories(parquet_file: Any, path: Path) -> dict[int, dict[str, Any]]:
    metadata = parquet_file.schema_arrow.metadata or {}
    categories_json = metadata.get(CATEGORIES_METADATA_KEY)
    if categories_json is None:
        raise ValueError(f'{path} is missing {CATEGORIES_METADATA_KEY!r} metadata.')

    categories = json.loads(categories_json.decode('utf-8'))
    return {int(category['id']): category for category in categories}


def read_annotation_rows(path: Path, pq: Any) -> tuple[list[str], dict[int, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f'Parquet file not found: {path}')

    parquet_file = pq.ParquetFile(path)
    categories = load_categories(parquet_file, path)
    rows: list[str] = []

    for row_group in range(parquet_file.metadata.num_row_groups):
        table = parquet_file.read_row_group(row_group, columns=['annotations_json'])
        rows.extend(table.column('annotations_json').to_pylist())

    return rows, categories


def count_class_ids(annotation_rows: list[str]) -> defaultdict[int, int]:
    class_id_count: defaultdict[int, int] = defaultdict(int)
    for annotations_json in annotation_rows:
        for annotation in json.loads(annotations_json):
            class_id = annotation.get('category_id')
            if class_id is not None:
                class_id_count[int(class_id)] += 1
    return class_id_count


def print_summary(
    train_images: int,
    val_images: int,
    train_class_id_count: defaultdict[int, int],
    val_class_id_count: defaultdict[int, int],
    categories: dict[int, dict[str, Any]],
) -> None:
    total_images = train_images + val_images
    total_annotations = sum(train_class_id_count.values()) + sum(val_class_id_count.values())
    summary_width = len(f'{total_annotations:,}')
    print('')
    print(f'{YELLOW}Train images     :{RESET}{train_images:>{summary_width},}')
    print(f'{YELLOW}Validation images:{RESET}{val_images:>{summary_width},}')
    print(f'{GREEN}Total images     :{RESET}{total_images:>{summary_width},}')
    print(f'{GREEN}Total annotations:{RESET}{total_annotations:>{summary_width},}')
    print('===================================================')

    print(f'{RED}Train Set Class ID Count{RESET}')
    total_train_count = 0
    for class_id, count in sorted(train_class_id_count.items()):
        name = categories.get(int(class_id), {}).get('name', 'unknown')
        print(f'{BLUE}class_id:{RESET}{int(class_id):>2} {BLUE}name:{RESET}{name:>20} {BLUE}count:{RESET}{count:>7,}')
        total_train_count += count
    print('---------------------------------------------------')
    print(f'{GREEN}Total count for train set:{RESET}{total_train_count:>7,}')
    print('===================================================')

    print(f'{RED}Validation Set Class ID Count{RESET}')
    total_val_count = 0
    for class_id, count in sorted(val_class_id_count.items()):
        name = categories.get(int(class_id), {}).get('name', 'unknown')
        print(f'{BLUE}class_id:{RESET}{int(class_id):>2} {BLUE}name:{RESET}{name:>20} {BLUE}count:{RESET}{count:>7,}')
        total_val_count += count
    print('---------------------------------------------------')
    print(f'{GREEN}Total count for validation set:{RESET}{total_val_count:>7,}')
    print('===================================================')
    print('')


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError('--train-ratio must be greater than 0.0 and less than 1.0')

    pq = require_pyarrow_parquet()
    train_rows, train_categories = read_annotation_rows(args.annotations_dir / 'train.parquet', pq)
    val_rows, val_categories = read_annotation_rows(args.annotations_dir / 'val.parquet', pq)
    if train_categories != val_categories:
        raise ValueError('train.parquet and val.parquet category metadata differ.')

    dataset = train_rows + val_rows
    random.Random(args.seed).shuffle(dataset)
    split_point = int(len(dataset) * args.train_ratio)
    train_set = dataset[:split_point]
    val_set = dataset[split_point:]

    print_summary(
        train_images=len(train_set),
        val_images=len(val_set),
        train_class_id_count=count_class_ids(train_set),
        val_class_id_count=count_class_ids(val_set),
        categories=train_categories,
    )


if __name__ == '__main__':
    main()
