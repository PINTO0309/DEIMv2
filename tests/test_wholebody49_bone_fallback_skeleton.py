import numpy as np

from demo.wholebody49 import demo_deimv2_torch_wholebody49_ins as demo


def _box(classid, cx, cy, handedness=-1, score=0.9):
    return demo.Box(
        classid=classid,
        score=score,
        x1=max(0, cx - 1),
        y1=max(0, cy - 1),
        x2=cx + 1,
        y2=cy + 1,
        cx=cx,
        cy=cy,
        handedness=handedness,
    )


def _body():
    return demo.Box(
        classid=demo.BODY_CLASS_ID,
        score=0.9,
        x1=0,
        y1=0,
        x2=100,
        y2=100,
        cx=50,
        cy=50,
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
