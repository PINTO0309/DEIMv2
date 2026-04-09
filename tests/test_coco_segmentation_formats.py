import numpy as np
import pytest
import torch

from PIL import Image

import faster_coco_eval.core.mask as coco_mask

from engine.data.dataset.coco_dataset import ConvertCocoPolysToMask
from engine.data.dataset.coco_utils import convert_coco_poly_to_mask


def encode_compressed_rle(mask):
    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def encode_uncompressed_rle(mask):
    flat_mask = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    counts = []
    current = 0
    run_length = 0

    for value in flat_mask:
        if value == current:
            run_length += 1
            continue
        counts.append(run_length)
        run_length = 1
        current = int(value)
    counts.append(run_length)

    return {
        "size": list(mask.shape),
        "counts": counts,
    }


def make_mask():
    mask = np.zeros((6, 7), dtype=np.uint8)
    mask[1:5, 2:6] = 1
    mask[4, 5] = 0
    return mask


def test_convert_coco_poly_to_mask_supports_polygon_compressed_rle_and_empty():
    polygon = [[2, 1, 6, 1, 6, 5, 2, 5]]
    compressed_rle = encode_compressed_rle(make_mask())

    masks, valid = convert_coco_poly_to_mask(
        [polygon, compressed_rle, []],
        height=6,
        width=7,
        return_valid=True,
    )

    assert tuple(masks.shape) == (3, 6, 7)
    assert valid.tolist() == [True, True, False]
    assert int(masks[0].sum()) > 0
    assert int(masks[1].sum()) == int(make_mask().sum())
    assert int(masks[2].sum()) == 0


def test_convert_coco_poly_to_mask_supports_uncompressed_rle():
    mask = make_mask()
    uncompressed_rle = encode_uncompressed_rle(mask)

    masks, valid = convert_coco_poly_to_mask(
        [uncompressed_rle],
        height=6,
        width=7,
        return_valid=True,
    )

    assert valid.tolist() == [True]
    assert tuple(masks.shape) == (1, 6, 7)
    assert int(masks[0].sum()) == int(mask.sum())


def test_convert_coco_poly_to_mask_falls_back_for_malformed_rle():
    malformed_rle = {
        "size": [6, 7],
        "counts": "this-is-not-a-valid-rle",
    }

    masks, valid = convert_coco_poly_to_mask(
        [malformed_rle],
        height=6,
        width=7,
        return_valid=True,
    )

    assert valid.tolist() == [False]
    assert int(masks[0].sum()) == 0


def test_convert_coco_polys_to_mask_marks_compressed_rle_as_valid_for_target_class():
    compressed_rle = encode_compressed_rle(make_mask())
    image = Image.new("RGB", (7, 6), color=0)
    transform = ConvertCocoPolysToMask(
        return_masks=True,
        mask_category_ids=[0],
        segm_eval_category_ids=[0],
    )

    _, target = transform(
        image,
        {
            "image_id": 1,
            "annotations": [
                {
                    "bbox": [2, 1, 4, 4],
                    "category_id": 0,
                    "area": 15.0,
                    "iscrowd": 0,
                    "segmentation": compressed_rle,
                },
                {
                    "bbox": [0, 0, 2, 2],
                    "category_id": 1,
                    "area": 4.0,
                    "iscrowd": 0,
                    "segmentation": compressed_rle,
                },
                {
                    "bbox": [0, 3, 2, 2],
                    "category_id": 0,
                    "area": 4.0,
                    "iscrowd": 0,
                    "segmentation": [],
                },
            ],
        },
    )

    assert target["mask_valid"].tolist() == [True, False, False]
    assert target["segm_eval_valid"].tolist() == [True, False, False]
    assert tuple(target["masks"].shape) == (3, 6, 7)
    assert int(target["masks"][0].sum()) == int(make_mask().sum())
    assert int(target["masks"][1].sum()) == 0
    assert int(target["masks"][2].sum()) == 0
