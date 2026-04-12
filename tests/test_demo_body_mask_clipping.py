import numpy as np
import torch

from demo.wholebody40.demo_deimv2_torch_wholebody40_ins import (
    BODY_CLASS_ID,
    Box,
    clip_binary_mask_to_box,
    prepare_prediction_payload,
)


def test_clip_binary_mask_to_box_zeros_pixels_outside_box():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 2:5] = True
    mask[6, 6] = True

    box = Box(
        classid=BODY_CLASS_ID,
        score=0.9,
        x1=2,
        y1=2,
        x2=4,
        y2=4,
        cx=3,
        cy=3,
    )

    clipped = clip_binary_mask_to_box(mask, box, padding=0)

    assert clipped[2:5, 2:5].all()
    assert not clipped[6, 6]
    assert clipped.sum() == 9


def test_prepare_prediction_payload_clips_each_body_mask_independently():
    mask_store = torch.zeros((2, 8, 8), dtype=torch.float32)
    mask_store[0, 1:4, 1:4] = 1.0
    mask_store[0, 6, 6] = 1.0
    mask_store[1, 4:7, 4:7] = 1.0
    mask_store[1, 0, 0] = 1.0

    boxes = [
        Box(
            classid=BODY_CLASS_ID,
            score=0.9,
            x1=1,
            y1=1,
            x2=3,
            y2=3,
            cx=2,
            cy=2,
            source_idx=0,
        ),
        Box(
            classid=BODY_CLASS_ID,
            score=0.8,
            x1=4,
            y1=4,
            x2=6,
            y2=6,
            cx=5,
            cy=5,
            source_idx=1,
        ),
    ]

    records = prepare_prediction_payload(
        boxes=boxes,
        result={'masks': mask_store},
        mask_threshold=0.5,
        enable_masks=True,
        enable_contours=False,
    )

    assert records[0]['mask_area'] == 9
    assert records[0]['mask_bbox'] == [1, 1, 3, 3]
    assert records[1]['mask_area'] == 9
    assert records[1]['mask_bbox'] == [4, 4, 6, 6]
