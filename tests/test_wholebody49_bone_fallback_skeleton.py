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


def test_draw_skeleton_prefers_clean_keypoint_over_larger_mixed_keypoint(monkeypatch):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    body = _body(source_idx=7)
    clean_knee = _box(39, 40, 40, handedness=0, half_size=1)
    mixed_knee = _box(39, 20, 40, handedness=0, half_size=5)
    ankle = _box(42, 40, 80, handedness=0, half_size=1)
    lines = _record_lines(monkeypatch)

    demo.draw_skeleton(
        image=image,
        boxes=[body, clean_knee, mixed_knee, ankle],
        max_dist_threshold=300,
        keypoint_mask_instance_map={
            id(clean_knee): 7,
            id(mixed_knee): 7,
            id(ankle): 7,
        },
        keypoint_instance_quality_map={
            id(clean_knee): demo.KeypointInstanceQuality(
                is_mixed=False,
                assigned_pixel_share=1.0,
                assigned_pixel_count=4,
                foreign_pixel_count=0,
            ),
            id(mixed_knee): demo.KeypointInstanceQuality(
                is_mixed=True,
                assigned_pixel_share=0.85,
                assigned_pixel_count=85,
                foreign_pixel_count=15,
            ),
        },
    )

    assert lines == [((40, 40), (40, 80))]


def test_draw_skeleton_keeps_area_ratio_order_for_same_mixed_priority(monkeypatch):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    body = _body(source_idx=7)
    smaller_knee = _box(39, 40, 40, handedness=0, half_size=1)
    larger_knee = _box(39, 20, 40, handedness=0, half_size=5)
    ankle = _box(42, 20, 80, handedness=0, half_size=1)
    lines = _record_lines(monkeypatch)

    demo.draw_skeleton(
        image=image,
        boxes=[body, smaller_knee, larger_knee, ankle],
        max_dist_threshold=300,
        keypoint_mask_instance_map={
            id(smaller_knee): 7,
            id(larger_knee): 7,
            id(ankle): 7,
        },
        keypoint_instance_quality_map={
            id(smaller_knee): demo.KeypointInstanceQuality(
                is_mixed=True,
                assigned_pixel_share=0.85,
                assigned_pixel_count=85,
                foreign_pixel_count=15,
            ),
            id(larger_knee): demo.KeypointInstanceQuality(
                is_mixed=True,
                assigned_pixel_share=0.85,
                assigned_pixel_count=85,
                foreign_pixel_count=15,
            ),
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


def test_draw_skeleton_limits_lines_per_keypoint_by_skeleton_degree(monkeypatch):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    shoulder_a = _box(24, 30, 20, handedness=0)
    shoulder_b = _box(24, 70, 20, handedness=1)
    elbow = _box(28, 50, 50, handedness=0)
    wrist = _box(31, 50, 90, handedness=0)
    lines = _record_lines(monkeypatch)

    demo.draw_skeleton(
        image=image,
        boxes=[
            _body(),
            _bone(25, 15, 55, 55),
            _bone(45, 15, 75, 55),
            _bone(45, 45, 55, 95),
            shoulder_a,
            shoulder_b,
            elbow,
            wrist,
        ],
        max_dist_threshold=300,
    )

    elbow_point = (50, 50)
    elbow_line_count = sum(elbow_point in line for line in lines)
    assert len(lines) == 2
    assert elbow_line_count == 2


def test_draw_skeleton_draws_bone_fallback_before_mask_instance_edge(monkeypatch):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    shoulder = _box(22, 70, 20, handedness=1)
    clean_elbow = _box(26, 25, 70, handedness=1)
    mixed_elbow = _box(26, 100, 70, handedness=1)
    lines = _record_lines(monkeypatch)

    demo.draw_skeleton(
        image=image,
        boxes=[_body(), _bone(15, 10, 80, 80), shoulder, clean_elbow, mixed_elbow],
        max_dist_threshold=300,
        keypoint_mask_instance_map={
            id(shoulder): 7,
            id(mixed_elbow): 7,
        },
    )

    assert lines[0] == ((70, 20), (25, 70))


def test_bone_fallback_prefers_clean_candidate_over_mixed_candidate(monkeypatch):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    shoulder = _box(22, 50, 20, handedness=1)
    clean_elbow = _box(26, 20, 70, handedness=1)
    mixed_elbow = _box(26, 95, 60, handedness=1)
    bone = _bone(5, 5, 105, 85)
    shoulder.person_id = 0
    clean_elbow.person_id = 0
    mixed_elbow.person_id = 0
    lines = _record_lines(monkeypatch)

    demo.draw_bone_fallback_skeleton(
        image=image,
        color=(0, 255, 255),
        bone_boxes=[bone],
        classid_to_boxes={
            demo.BONE_CLASS_ID: [bone],
            22: [shoulder],
            26: [clean_elbow, mixed_elbow],
        },
        keypoint_mask_instance_map=None,
        keypoint_instance_quality_map={
            id(clean_elbow): demo.KeypointInstanceQuality(
                is_mixed=False,
                assigned_pixel_share=1.0,
                assigned_pixel_count=4,
                foreign_pixel_count=0,
            ),
            id(mixed_elbow): demo.KeypointInstanceQuality(
                is_mixed=True,
                assigned_pixel_share=0.62,
                assigned_pixel_count=62,
                foreign_pixel_count=38,
            ),
        },
        line_registry=demo.SkeletonLineRegistry(),
    )

    assert lines == [((50, 20), (20, 70))]


def test_instance_skeleton_skips_mixed_keypoint_mask_edge_without_bone_support(monkeypatch):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    shoulder = _box(22, 50, 20, handedness=1)
    mixed_elbow = _box(26, 90, 60, handedness=1)
    lines = _record_lines(monkeypatch)

    demo.draw_skeleton(
        image=image,
        boxes=[_body(), shoulder, mixed_elbow],
        max_dist_threshold=300,
        keypoint_mask_instance_map={
            id(shoulder): 7,
            id(mixed_elbow): 7,
        },
        keypoint_instance_quality_map={
            id(mixed_elbow): demo.KeypointInstanceQuality(
                is_mixed=True,
                assigned_pixel_share=0.62,
                assigned_pixel_count=62,
                foreign_pixel_count=38,
            ),
        },
    )

    assert lines == []


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
