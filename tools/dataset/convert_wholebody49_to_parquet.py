#!/usr/bin/env python3
"""Convert wholebody49 COCO JSON annotations to image-row Parquet files."""

import argparse
import importlib.util
import json
from pathlib import Path


DEFAULT_STEMS = ('train', 'train_ins', 'val', 'val_ins')
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_coco_parquet_helpers():
    module_path = REPO_ROOT / 'engine' / 'data' / 'dataset' / 'coco_parquet.py'
    spec = importlib.util.spec_from_file_location('deimv2_coco_parquet', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summarize_coco_for_parquet, module.write_coco_parquet


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert wholebody49 COCO JSON files to Parquet with embedded image bytes.'
    )
    parser.add_argument(
        '--dataset-root',
        type=Path,
        default=Path('tools/dataset/wholebody49'),
        help='Directory containing images/ and annotations/.',
    )
    parser.add_argument(
        '--stems',
        nargs='+',
        default=list(DEFAULT_STEMS),
        help='Annotation filename stems to convert from annotations/{stem}.json.',
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=None,
        help='Output directory. Defaults to {dataset-root}/annotations.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Only convert the first N images from each JSON. Useful for smoke tests.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print conversion summaries without writing Parquet files.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing Parquet files.',
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def main():
    args = parse_args()
    summarize_coco_for_parquet, write_coco_parquet = load_coco_parquet_helpers()
    dataset_root = args.dataset_root
    image_root = dataset_root / 'images'
    annotations_dir = dataset_root / 'annotations'
    out_dir = args.out_dir or annotations_dir

    if not image_root.is_dir():
        raise FileNotFoundError(f'Image directory not found: {image_root}')
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f'Annotation directory not found: {annotations_dir}')

    for stem in args.stems:
        json_file = annotations_dir / f'{stem}.json'
        if not json_file.is_file():
            raise FileNotFoundError(f'Annotation JSON not found: {json_file}')

        dataset = load_json(json_file)
        output_file = out_dir / f'{stem}.parquet'
        summary = summarize_coco_for_parquet(dataset, limit=args.limit)

        if args.dry_run:
            print(
                f'[dry-run] {json_file} -> {output_file}: '
                f"images={summary['images']} annotations={summary['annotations']} "
                f"categories={summary['categories']}"
            )
            continue

        result = write_coco_parquet(
            dataset,
            image_root,
            output_file,
            limit=args.limit,
            overwrite=args.overwrite,
            progress=True,
            progress_desc=f'Writing {stem}',
        )
        print(
            f"wrote {result['output_file']}: images={result['images']} "
            f"annotations={result['annotations']} categories={result['categories']}"
        )


if __name__ == '__main__':
    main()
