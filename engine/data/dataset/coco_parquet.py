"""
Utilities for storing COCO-style datasets in Parquet.

The Parquet layout is one row per image. Each row stores image metadata,
the encoded image bytes, and the image's COCO annotations as JSON. Categories
are stored in Parquet file metadata.
"""

import copy
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from faster_coco_eval import COCO
from tqdm import tqdm


CATEGORIES_METADATA_KEY = b'coco_categories_json'
FORMAT_METADATA_KEY = b'coco_parquet_format'
FORMAT_VERSION = 'deimv2.coco-image-row.v1'


def is_parquet_path(path) -> bool:
    return str(path).lower().endswith('.parquet')


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            'pyarrow is required for COCO Parquet support. '
            'Install it with `pip install pyarrow` or sync the project dependencies.'
        ) from exc
    return pa, pq


def _metadata_value(metadata: Optional[Dict[bytes, bytes]], key: bytes) -> Optional[str]:
    if not metadata or key not in metadata:
        return None
    value = metadata[key]
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value


def _to_py(value):
    if hasattr(value, 'as_py'):
        return value.as_py()
    return value


class CocoParquetStore:
    """Lazy random-access reader for one-image-per-row COCO Parquet files."""

    COLUMNS = ['image_id', 'file_name', 'width', 'height', 'image_bytes', 'annotations_json']
    COCO_COLUMNS = ['image_id', 'file_name', 'width', 'height', 'annotations_json']

    def __init__(self, path):
        _, pq = _require_pyarrow()
        self.path = str(path)
        self._pq = pq
        self._parquet_file = None
        self._categories = None
        self._coco = None
        self._ids = None

        metadata = self.parquet_file.metadata
        if metadata.num_row_groups != metadata.num_rows:
            raise ValueError(
                f'{self.path} must use one row group per image for random access. '
                f'Found {metadata.num_row_groups} row groups for {metadata.num_rows} rows.'
            )

    @property
    def parquet_file(self):
        if self._parquet_file is None:
            self._parquet_file = self._pq.ParquetFile(self.path)
        return self._parquet_file

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_parquet_file'] = None
        return state

    def __len__(self):
        return self.parquet_file.metadata.num_rows

    @property
    def categories(self) -> List[dict]:
        if self._categories is None:
            metadata = self.parquet_file.schema_arrow.metadata
            categories_json = _metadata_value(metadata, CATEGORIES_METADATA_KEY)
            if categories_json is None:
                raise ValueError(f'{self.path} is missing {CATEGORIES_METADATA_KEY!r} metadata.')
            self._categories = json.loads(categories_json)
        return copy.deepcopy(self._categories)

    @property
    def ids(self) -> List[int]:
        if self._ids is None:
            ids = []
            for row_group in range(len(self)):
                table = self.parquet_file.read_row_group(row_group, columns=['image_id'])
                ids.append(int(_to_py(table.column('image_id')[0])))
            self._ids = ids
        return list(self._ids)

    def read_item(self, idx: int):
        table = self.parquet_file.read_row_group(idx, columns=self.COLUMNS)
        row = {name: _to_py(table.column(name)[0]) for name in self.COLUMNS}
        image = Image.open(BytesIO(row['image_bytes'])).convert('RGB')
        annotations = json.loads(row['annotations_json'])
        image_id = int(row['image_id'])
        return image, {'image_id': image_id, 'annotations': annotations}

    def build_coco(self) -> COCO:
        if self._coco is not None:
            return self._coco

        images = []
        annotations = []
        for row_group in range(len(self)):
            table = self.parquet_file.read_row_group(row_group, columns=self.COCO_COLUMNS)
            row = {name: _to_py(table.column(name)[0]) for name in self.COCO_COLUMNS}
            image_id = int(row['image_id'])
            images.append({
                'id': image_id,
                'file_name': row['file_name'],
                'width': int(row['width']),
                'height': int(row['height']),
            })
            annotations.extend(json.loads(row['annotations_json']))

        coco = COCO()
        coco.dataset = {
            'images': images,
            'categories': self.categories,
            'annotations': annotations,
        }
        coco.createIndex()
        self._coco = coco
        return self._coco

    def close(self):
        self._parquet_file = None


def load_coco_api_from_parquet(path) -> COCO:
    return CocoParquetStore(path).build_coco()


def write_coco_parquet(
    coco_dataset: dict,
    image_root,
    output_file,
    *,
    limit: Optional[int] = None,
    overwrite: bool = False,
    progress: bool = False,
    progress_desc: Optional[str] = None,
) -> dict:
    """Write a COCO JSON dictionary to the DEIMv2 image-row Parquet layout."""
    pa, pq = _require_pyarrow()

    image_root = Path(image_root)
    output_file = Path(output_file)
    if output_file.exists() and not overwrite:
        raise FileExistsError(f'Output already exists: {output_file}')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    images = list(coco_dataset.get('images', []))
    if limit is not None:
        images = images[:limit]

    selected_ids = {image['id'] for image in images}
    annotations_by_image: Dict[int, List[dict]] = {image['id']: [] for image in images}
    for ann in coco_dataset.get('annotations', []):
        image_id = ann.get('image_id')
        if image_id in selected_ids:
            annotations_by_image[image_id].append(ann)

    categories = coco_dataset.get('categories', [])
    metadata = {
        CATEGORIES_METADATA_KEY: json.dumps(categories, separators=(',', ':')).encode('utf-8'),
        FORMAT_METADATA_KEY: FORMAT_VERSION.encode('utf-8'),
    }
    schema = pa.schema([
        pa.field('image_id', pa.int64()),
        pa.field('file_name', pa.string()),
        pa.field('width', pa.int32()),
        pa.field('height', pa.int32()),
        pa.field('image_bytes', pa.binary()),
        pa.field('annotations_json', pa.string()),
    ]).with_metadata(metadata)

    annotation_count = 0
    with pq.ParquetWriter(output_file, schema=schema, compression='zstd') as writer:
        image_iter = tqdm(
            images,
            desc=progress_desc or f'Writing {output_file.name}',
            dynamic_ncols=True,
            unit='image',
            disable=not progress,
        )
        for image_info in image_iter:
            file_name = image_info['file_name']
            image_path = image_root / file_name
            if not image_path.is_file():
                raise FileNotFoundError(f'Image referenced by COCO JSON was not found: {image_path}')

            anns = annotations_by_image.get(image_info['id'], [])
            annotation_count += len(anns)
            table = pa.Table.from_pydict(
                {
                    'image_id': [int(image_info['id'])],
                    'file_name': [file_name],
                    'width': [int(image_info['width'])],
                    'height': [int(image_info['height'])],
                    'image_bytes': [image_path.read_bytes()],
                    'annotations_json': [json.dumps(anns, separators=(',', ':'))],
                },
                schema=schema,
            )
            writer.write_table(table, row_group_size=1)

    return {
        'images': len(images),
        'annotations': annotation_count,
        'categories': len(categories),
        'output_file': str(output_file),
    }


def summarize_coco_for_parquet(coco_dataset: dict, limit: Optional[int] = None) -> dict:
    images = coco_dataset.get('images', [])
    annotations = coco_dataset.get('annotations', [])
    if limit is None:
        return {
            'images': len(images),
            'annotations': len(annotations),
            'categories': len(coco_dataset.get('categories', [])),
        }

    selected_ids = {image['id'] for image in images[:limit]}
    return {
        'images': min(limit, len(images)),
        'annotations': sum(1 for ann in annotations if ann.get('image_id') in selected_ids),
        'categories': len(coco_dataset.get('categories', [])),
    }
