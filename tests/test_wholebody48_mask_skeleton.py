from types import SimpleNamespace

import numpy as np
import torch

from demo.wholebody48 import demo_deimv2_torch_wholebody48_ins as demo


def _box(classid, cx, cy, source_idx=-1, score=0.9):
    return demo.Box(
        classid=classid,
        score=score,
        x1=max(0, cx - 1),
        y1=max(0, cy - 1),
        x2=cx + 1,
        y2=cy + 1,
        cx=cx,
        cy=cy,
        source_idx=source_idx,
    )


def _body(source_idx, score=0.9):
    return demo.Box(
        classid=demo.BODY_CLASS_ID,
        score=score,
        x1=0,
        y1=0,
        x2=9,
        y2=9,
        cx=4,
        cy=4,
        source_idx=source_idx,
    )


def test_draw_skeleton_prefers_same_mask_instance_over_shorter_distance(monkeypatch):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    parent_same_mask = _box(21, 8, 5)
    parent_nearer_other_mask = _box(21, 4, 5)
    child = _box(22, 5, 5)
    boxes = [
        _body(0),
        parent_same_mask,
        parent_nearer_other_mask,
        child,
    ]
    mask_context = {
        id(parent_same_mask): 0,
        id(parent_nearer_other_mask): 1,
        id(child): 0,
    }
    lines = []

    def record_line(_image, pt1, pt2, color, thickness=1):
        lines.append((pt1, pt2))

    monkeypatch.setattr(demo.cv2, 'line', record_line)

    demo.draw_skeleton(
        image=image,
        boxes=boxes,
        max_dist_threshold=300,
        keypoint_mask_instance_map=mask_context,
    )

    assert lines[0] == ((8, 5), (5, 5))


def test_draw_skeleton_without_mask_context_keeps_distance_order(monkeypatch):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    parent_far = _box(21, 8, 5)
    parent_near = _box(21, 4, 5)
    child = _box(22, 5, 5)
    boxes = [
        _body(0),
        parent_far,
        parent_near,
        child,
    ]
    lines = []

    def record_line(_image, pt1, pt2, color, thickness=1):
        lines.append((pt1, pt2))

    monkeypatch.setattr(demo.cv2, 'line', record_line)

    demo.draw_skeleton(
        image=image,
        boxes=boxes,
        max_dist_threshold=300,
    )

    assert lines[0] == ((4, 5), (5, 5))


def test_draw_skeleton_rejects_known_handedness_mismatch_even_with_same_mask(monkeypatch):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    parent = _box(22, 4, 5)
    child = _box(26, 5, 5)
    parent.handedness = 0
    child.handedness = 1
    boxes = [
        _body(0),
        parent,
        child,
    ]
    mask_context = {
        id(parent): 0,
        id(child): 0,
    }
    lines = []

    def record_line(_image, pt1, pt2, color, thickness=1):
        lines.append((pt1, pt2))

    monkeypatch.setattr(demo.cv2, 'line', record_line)

    demo.draw_skeleton(
        image=image,
        boxes=boxes,
        max_dist_threshold=300,
        keypoint_mask_instance_map=mask_context,
    )

    assert lines == []


def test_draw_skeleton_allows_unknown_handedness(monkeypatch):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    parent = _box(22, 4, 5)
    child = _box(26, 5, 5)
    parent.handedness = 0
    child.handedness = -1
    boxes = [
        _body(0),
        parent,
        child,
    ]
    lines = []

    def record_line(_image, pt1, pt2, color, thickness=1):
        lines.append((pt1, pt2))

    monkeypatch.setattr(demo.cv2, 'line', record_line)

    demo.draw_skeleton(
        image=image,
        boxes=boxes,
        max_dist_threshold=300,
    )

    assert lines == [((4, 5), (5, 5))]


def test_build_keypoint_mask_instance_map_uses_highest_center_probability():
    masks = torch.zeros((2, 10, 10), dtype=torch.float32)
    masks[0, 5, 5] = 0.7
    masks[1, 5, 5] = 0.9
    keypoint = _box(21, 5, 5)
    boxes = [
        _body(0, score=0.99),
        _body(1, score=0.50),
        keypoint,
    ]
    args = SimpleNamespace(
        mask_bilateral_d=0,
        mask_bilateral_sigma_color=1.0,
        mask_bilateral_sigma_space=3.0,
    )

    mask_context = demo.build_keypoint_mask_instance_map(
        boxes=boxes,
        result={'masks': masks},
        args=args,
        mask_threshold=0.5,
    )

    assert mask_context[id(keypoint)] == 1
