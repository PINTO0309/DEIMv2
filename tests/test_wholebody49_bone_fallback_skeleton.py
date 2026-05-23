import numpy as np

from demo.wholebody49 import demo_deimv2_torch_wholebody49_ins as demo


def _box(classid, cx, cy, handedness=-1, score=0.9, half_size=1):
    return demo.Box(
        classid=classid,
        score=score,
        x1=max(0, cx - half_size),
        y1=max(0, cy - half_size),
        x2=cx + half_size,
        y2=cy + half_size,
        cx=cx,
        cy=cy,
        handedness=handedness,
    )


def _body(source_idx=-1, x1=0, y1=0, x2=100, y2=100):
    return demo.Box(
        classid=demo.BODY_CLASS_ID,
        score=0.9,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        cx=(x1 + x2) // 2,
        cy=(y1 + y2) // 2,
        source_idx=source_idx,
    )


def _bone(x1=0, y1=0, x2=100, y2=100):
    return demo.Box(
        classid=demo.BONE_CLASS_ID,
        score=0.9,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        cx=(x1 + x2) // 2,
        cy=(y1 + y2) // 2,
    )


def _record_lines(monkeypatch):
    lines = []

    def record_line(_image, pt1, pt2, color, thickness=1):
        lines.append((pt1, pt2))

    monkeypatch.setattr(demo.cv2, 'line', record_line)
    return lines


def test_bone_fallback_draws_when_one_endpoint_has_unknown_handedness(monkeypatch):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    parent = _box(22, 2, 2, handedness=0)
    child = _box(26, 98, 98, handedness=-1)
    lines = _record_lines(monkeypatch)

    demo.draw_skeleton(
        image=image,
        boxes=[_body(), _bone(), parent, child],
        max_dist_threshold=300,
    )

    assert lines == [((2, 2), (98, 98))]


def test_bone_fallback_draws_when_both_endpoints_have_unknown_handedness(monkeypatch):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    parent = _box(22, 98, 2, handedness=-1)
    child = _box(26, 2, 98, handedness=-1)
    lines = _record_lines(monkeypatch)

    demo.draw_skeleton(
        image=image,
        boxes=[_body(), _bone(), parent, child],
        max_dist_threshold=300,
    )

    assert lines == [((98, 2), (2, 98))]


def test_bone_fallback_rejects_weak_bone_geometry(monkeypatch):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    parent = _box(22, 2, 2, handedness=0)
    child = _box(26, 60, 15, handedness=-1)
    lines = _record_lines(monkeypatch)

    demo.draw_skeleton(
        image=image,
        boxes=[_body(), _bone(), parent, child],
        max_dist_threshold=300,
    )

    assert lines == []


def test_bone_fallback_does_not_duplicate_regular_skeleton_line(monkeypatch):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    parent = _box(22, 2, 2, handedness=0)
    child = _box(26, 98, 98, handedness=0)
    lines = _record_lines(monkeypatch)

    demo.draw_skeleton(
        image=image,
        boxes=[_body(), _bone(), parent, child],
        max_dist_threshold=300,
    )

    assert lines == [((2, 2), (98, 98))]


def test_bone_fallback_rejects_different_mask_instances(monkeypatch):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    parent = _box(22, 2, 2, handedness=0)
    child = _box(26, 98, 98, handedness=-1)
    lines = _record_lines(monkeypatch)

    demo.draw_skeleton(
        image=image,
        boxes=[_body(), _bone(), parent, child],
        max_dist_threshold=300,
        keypoint_mask_instance_map={id(parent): 0, id(child): 1},
    )

    assert lines == []


def test_bone_fallback_ignores_known_handedness_mismatch(monkeypatch):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    parent = _box(36, 5, 18, handedness=0)
    child = _box(39, 95, 22, handedness=1)
    lines = _record_lines(monkeypatch)

    demo.draw_skeleton(
        image=image,
        boxes=[_body(), _bone(0, 0, 100, 40), parent, child],
        max_dist_threshold=300,
    )

    assert lines == [((5, 18), (95, 22))]


def test_draw_skeleton_suppresses_duplicate_mask_keypoints_by_body_area_ratio(monkeypatch):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    body = _body(source_idx=7)
    smaller_knee = _box(39, 40, 40, handedness=0, half_size=1)
    larger_knee = _box(39, 20, 40, handedness=0, half_size=5)
    smaller_ankle = _box(42, 41, 70, handedness=0, half_size=1)
    larger_ankle = _box(42, 20, 80, handedness=0, half_size=5)
    lines = _record_lines(monkeypatch)

    demo.draw_skeleton(
        image=image,
        boxes=[body, smaller_knee, larger_knee, smaller_ankle, larger_ankle],
        max_dist_threshold=300,
        keypoint_mask_instance_map={
            id(smaller_knee): 7,
            id(larger_knee): 7,
            id(smaller_ankle): 7,
            id(larger_ankle): 7,
        },
    )

    assert lines == [((20, 40), (20, 80))]


def test_draw_skeleton_suppresses_duplicate_person_keypoints_by_body_area_ratio(monkeypatch):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    body = _body()
    smaller_knee = _box(39, 40, 40, handedness=0, half_size=1)
    larger_knee = _box(39, 20, 40, handedness=0, half_size=5)
    smaller_ankle = _box(42, 41, 70, handedness=0, half_size=1)
    larger_ankle = _box(42, 20, 80, handedness=0, half_size=5)
    lines = _record_lines(monkeypatch)

    demo.draw_skeleton(
        image=image,
        boxes=[body, smaller_knee, larger_knee, smaller_ankle, larger_ankle],
        max_dist_threshold=300,
    )

    assert lines == [((20, 40), (20, 80))]


def test_bone_rescue_draws_mask_mismatched_same_person_edge(monkeypatch):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    knee = _box(39, 30, 30, handedness=0)
    ankle = _box(42, 35, 80, handedness=0)
    lines = _record_lines(monkeypatch)

    demo.draw_skeleton(
        image=image,
        boxes=[_body(), _bone(25, 20, 40, 90), knee, ankle],
        max_dist_threshold=300,
        keypoint_mask_instance_map={id(knee): 0, id(ankle): 1},
    )

    assert lines == [((30, 30), (35, 80))]


def test_bone_rescue_rejects_when_endpoint_is_outside_bone(monkeypatch):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    knee = _box(39, 30, 30, handedness=0)
    ankle = _box(42, 35, 80, handedness=0)
    lines = _record_lines(monkeypatch)

    demo.draw_skeleton(
        image=image,
        boxes=[_body(), _bone(25, 20, 40, 60), knee, ankle],
        max_dist_threshold=300,
        keypoint_mask_instance_map={id(knee): 0, id(ankle): 1},
    )

    assert lines == []


def test_bone_rescue_rejects_different_person_ids(monkeypatch):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    knee = _box(39, 20, 30, handedness=0)
    ankle = _box(42, 80, 80, handedness=0)
    lines = _record_lines(monkeypatch)

    demo.draw_skeleton(
        image=image,
        boxes=[
            _body(x1=0, y1=0, x2=50, y2=100),
            _body(x1=60, y1=0, x2=110, y2=100),
            _bone(10, 20, 90, 90),
            knee,
            ankle,
        ],
        max_dist_threshold=300,
        keypoint_mask_instance_map={id(knee): 0, id(ankle): 1},
    )

    assert lines == []


def test_bone_rescue_rejects_handedness_mismatch(monkeypatch):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    knee = _box(39, 30, 30, handedness=0)
    ankle = _box(42, 35, 80, handedness=1)
    lines = _record_lines(monkeypatch)

    demo.draw_skeleton(
        image=image,
        boxes=[_body(), _bone(25, 20, 40, 90), knee, ankle],
        max_dist_threshold=300,
        keypoint_mask_instance_map={id(knee): 0, id(ankle): 1},
    )

    assert lines == []


def test_bone_rescue_does_not_duplicate_regular_skeleton_line(monkeypatch):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    knee = _box(39, 30, 30, handedness=0)
    ankle = _box(42, 35, 80, handedness=0)
    lines = _record_lines(monkeypatch)

    demo.draw_skeleton(
        image=image,
        boxes=[_body(), _bone(25, 20, 40, 90), knee, ankle],
        max_dist_threshold=300,
        keypoint_mask_instance_map={id(knee): 0, id(ankle): 0},
    )

    assert lines == [((30, 30), (35, 80))]
