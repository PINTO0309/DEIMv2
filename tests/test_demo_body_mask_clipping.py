import numpy as np
import torch
from types import SimpleNamespace

from demo.wholebody40.demo_deimv2_torch_wholebody40_ins import (
    BODY_CLASS_ID,
    Box,
    clip_binary_mask_to_box,
    overlay_body_masks,
    postprocess_body_mask_probs,
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
        mask_bilateral_d=0,
    )

    assert records[0]['mask_area'] == 9
    assert records[0]['mask_bbox'] == [1, 1, 3, 3]
    assert records[1]['mask_area'] == 9
    assert records[1]['mask_bbox'] == [4, 4, 6, 6]


def test_postprocess_body_mask_probs_bilateral_can_fill_single_pixel_hole():
    mask_probs = np.ones((7, 7), dtype=np.float32)
    mask_probs[3, 3] = 0.0

    filtered = postprocess_body_mask_probs(
        mask_probs,
        bilateral_d=5,
        bilateral_sigma_color=1.0,
        bilateral_sigma_space=1.0,
    )

    assert filtered[3, 3] > 0.5


def test_prepare_prediction_payload_uses_bilateral_filter_for_body_mask_json():
    mask_store = torch.zeros((1, 7, 7), dtype=torch.float32)
    mask_store[0, 1:6, 1:6] = 1.0
    mask_store[0, 3, 3] = 0.0

    box = Box(
        classid=BODY_CLASS_ID,
        score=0.9,
        x1=1,
        y1=1,
        x2=5,
        y2=5,
        cx=3,
        cy=3,
        source_idx=0,
    )

    records_without_filter = prepare_prediction_payload(
        boxes=[box],
        result={'masks': mask_store},
        mask_threshold=0.5,
        enable_masks=True,
        enable_contours=False,
        mask_bilateral_d=0,
    )
    records_with_filter = prepare_prediction_payload(
        boxes=[box],
        result={'masks': mask_store},
        mask_threshold=0.5,
        enable_masks=True,
        enable_contours=False,
        mask_bilateral_d=5,
        mask_bilateral_sigma_color=1.0,
        mask_bilateral_sigma_space=1.0,
    )

    assert records_without_filter[0]['mask_area'] == 24
    assert records_with_filter[0]['mask_area'] == 25


def test_overlay_body_masks_uses_bilateral_filter_args():
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    mask_store = torch.zeros((1, 7, 7), dtype=torch.float32)
    mask_store[0, 1:6, 1:6] = 1.0
    mask_store[0, 3, 3] = 0.0
    box = Box(
        classid=BODY_CLASS_ID,
        score=0.9,
        x1=1,
        y1=1,
        x2=5,
        y2=5,
        cx=3,
        cy=3,
        source_idx=0,
    )
    args = SimpleNamespace(
        mask_bilateral_d=5,
        mask_bilateral_sigma_color=1.0,
        mask_bilateral_sigma_space=1.0,
    )

    rendered = overlay_body_masks(
        image=image,
        result={'masks': mask_store},
        boxes=[box],
        args=args,
        mask_threshold=0.5,
        mask_alpha=255,
        disable_render_classids=set(),
    )

    assert rendered[3, 3].any()
