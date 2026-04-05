from pathlib import Path

import pytest
import torch

from engine.core.yaml_config import YAMLConfig
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.matcher import HungarianMatcher
from engine.deim.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


NUM_CLASSES = 41
CENTER_CLASS_IDS = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 35, 36, 37, 38]
REPO_ROOT = Path(__file__).resolve().parents[1]


def make_matcher(**kwargs):
    params = dict(
        weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
        use_focal_loss=False,
        alpha=0.25,
        gamma=2.0,
        change_matcher=False,
        iou_order_alpha=4.0,
        matcher_change_epoch=45,
    )
    params.update(kwargs)
    return HungarianMatcher(**params)


def make_criterion(**kwargs):
    params = dict(
        matcher=make_matcher(),
        weight_dict={
            'loss_mal': 1.0,
            'loss_bbox': 1.0,
            'loss_giou': 1.0,
            'loss_fgl': 1.0,
            'loss_ddf': 1.0,
            'loss_center': 1.0,
        },
        losses=['mal', 'boxes', 'local'],
        alpha=0.75,
        gamma=1.5,
        num_classes=NUM_CLASSES,
        reg_max=4,
    )
    params.update(kwargs)
    return DEIMCriterion(**params)


def make_logits(class_ids, value=8.0):
    logits = torch.full((1, len(class_ids), NUM_CLASSES), -8.0)
    for query_idx, class_id in enumerate(class_ids):
        logits[0, query_idx, class_id] = value
    return logits


def test_loss_boxes_switches_center_targets_to_center_and_wh_terms():
    criterion = make_criterion(
        center_target_class_ids=[21],
        center_bbox_wh_weight=0.25,
        center_giou_weight=0.25,
    )
    outputs = {
        'pred_boxes': torch.tensor(
            [[[0.15, 0.15, 0.10, 0.20], [0.52, 0.52, 0.18, 0.18]]],
            dtype=torch.float32,
        )
    }
    targets = [{
        'boxes': torch.tensor(
            [[0.10, 0.10, 0.20, 0.40], [0.50, 0.50, 0.20, 0.20]],
            dtype=torch.float32,
        ),
        'labels': torch.tensor([21, 5], dtype=torch.int64),
    }]
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]

    losses = criterion.loss_boxes(outputs, targets, indices, num_boxes=2)

    src_boxes = outputs['pred_boxes'][0]
    target_boxes = targets[0]['boxes']
    expected_bbox = (
        (0.0 + 0.0 + abs(0.10 - 0.20) * 0.25 + abs(0.20 - 0.40) * 0.25)
        + (abs(0.52 - 0.50) + abs(0.52 - 0.50) + abs(0.18 - 0.20) + abs(0.18 - 0.20))
    ) / 2.0
    expected_center = (
        torch.norm(src_boxes[0, :2] - target_boxes[0, :2], p=2)
        / max(0.5 * torch.norm(target_boxes[0, 2:], p=2).item(), 0.02)
    ) / 2.0
    giou_diag = torch.diag(
        generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes),
        )
    )
    expected_giou = ((1 - giou_diag[0]) * 0.25 + (1 - giou_diag[1])) / 2.0

    assert losses['loss_bbox'].item() == pytest.approx(expected_bbox, rel=1e-5)
    assert losses['loss_center'].item() == pytest.approx(expected_center.item(), rel=1e-5)
    assert losses['loss_giou'].item() == pytest.approx(expected_giou.item(), rel=1e-5)


def test_loss_boxes_is_backward_compatible_when_center_targets_are_disabled():
    outputs = {
        'pred_boxes': torch.tensor(
            [[[0.15, 0.15, 0.10, 0.20], [0.52, 0.52, 0.18, 0.18]]],
            dtype=torch.float32,
        )
    }
    targets = [{
        'boxes': torch.tensor(
            [[0.10, 0.10, 0.20, 0.40], [0.50, 0.50, 0.20, 0.20]],
            dtype=torch.float32,
        ),
        'labels': torch.tensor([21, 5], dtype=torch.int64),
    }]
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]

    baseline = make_criterion()
    center_disabled = make_criterion(center_target_class_ids=[])

    baseline_losses = baseline.loss_boxes(outputs, targets, indices, num_boxes=2)
    center_disabled_losses = center_disabled.loss_boxes(outputs, targets, indices, num_boxes=2)

    assert baseline_losses.keys() == center_disabled_losses.keys()
    for key in baseline_losses:
        assert baseline_losses[key].item() == pytest.approx(center_disabled_losses[key].item(), rel=1e-6)
    assert 'loss_center' not in baseline_losses


def test_loss_labels_mal_uses_center_quality_for_center_targets():
    outputs = {
        'pred_boxes': torch.tensor([[[0.50, 0.50, 0.05, 0.05]]], dtype=torch.float32),
        'pred_logits': make_logits([21], value=4.0),
    }
    targets = [{
        'boxes': torch.tensor([[0.50, 0.50, 0.20, 0.20]], dtype=torch.float32),
        'labels': torch.tensor([21], dtype=torch.int64),
    }]
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    baseline = make_criterion()
    center_aware = make_criterion(center_target_class_ids=[21])

    baseline_loss = baseline.loss_labels_mal(outputs, targets, indices, num_boxes=1)['loss_mal']
    center_loss = center_aware.loss_labels_mal(outputs, targets, indices, num_boxes=1)['loss_mal']

    assert center_loss.item() < baseline_loss.item()


def test_loss_local_respects_center_local_weight():
    outputs = {
        'pred_boxes': torch.tensor([[[0.50, 0.50, 0.20, 0.20]]], dtype=torch.float32),
        'pred_corners': torch.zeros((1, 1, 20), dtype=torch.float32),
        'ref_points': torch.tensor([[[0.50, 0.50, 0.20, 0.20]]], dtype=torch.float32),
        'up': torch.tensor([0.5], dtype=torch.float32),
        'reg_scale': torch.tensor([4.0], dtype=torch.float32),
    }
    targets = [{
        'boxes': torch.tensor([[0.50, 0.50, 0.20, 0.20]], dtype=torch.float32),
        'labels': torch.tensor([21], dtype=torch.int64),
    }]
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    baseline = make_criterion()
    center_aware = make_criterion(center_target_class_ids=[21], center_local_weight=0.0)

    baseline_loss = baseline.loss_local(outputs, targets, indices, num_boxes=1)['loss_fgl']
    center_loss = center_aware.loss_local(outputs, targets, indices, num_boxes=1)['loss_fgl']

    assert baseline_loss.item() > 0.0
    assert center_loss.item() == pytest.approx(0.0, abs=1e-8)


def test_matcher_uses_center_cost_only_for_configured_classes_before_switch():
    outputs = {
        'pred_logits': make_logits([21, 21, 5, 5]),
        'pred_boxes': torch.tensor(
            [[
                [0.50, 0.50, 0.05, 0.05],
                [0.62, 0.50, 0.20, 0.20],
                [0.80, 0.80, 0.05, 0.05],
                [0.92, 0.80, 0.20, 0.20],
            ]],
            dtype=torch.float32,
        ),
    }
    targets = [{
        'labels': torch.tensor([21, 5], dtype=torch.int64),
        'boxes': torch.tensor(
            [[0.50, 0.50, 0.20, 0.20], [0.80, 0.80, 0.20, 0.20]],
            dtype=torch.float32,
        ),
    }]

    baseline = make_matcher()
    center_aware = make_matcher(center_target_class_ids=[21], center_match_wh_cost_weight=0.25)

    baseline_indices = baseline(outputs, targets, epoch=0)['indices'][0][0].tolist()
    center_indices = center_aware(outputs, targets, epoch=0)['indices'][0][0].tolist()

    assert baseline_indices == [1, 3]
    assert center_indices == [0, 3]


def test_matcher_uses_center_quality_only_for_configured_classes_after_switch():
    outputs = {
        'pred_logits': make_logits([21, 21, 5, 5]),
        'pred_boxes': torch.tensor(
            [[
                [0.50, 0.50, 0.05, 0.05],
                [0.62, 0.50, 0.20, 0.20],
                [0.80, 0.80, 0.05, 0.05],
                [0.92, 0.80, 0.20, 0.20],
            ]],
            dtype=torch.float32,
        ),
    }
    targets = [{
        'labels': torch.tensor([21, 5], dtype=torch.int64),
        'boxes': torch.tensor(
            [[0.50, 0.50, 0.20, 0.20], [0.80, 0.80, 0.20, 0.20]],
            dtype=torch.float32,
        ),
    }]

    baseline = make_matcher(change_matcher=True, matcher_change_epoch=0, iou_order_alpha=4.0)
    center_aware = make_matcher(
        change_matcher=True,
        matcher_change_epoch=0,
        iou_order_alpha=4.0,
        center_target_class_ids=[21],
    )

    baseline_indices = baseline(outputs, targets, epoch=45)['indices'][0][0].tolist()
    center_indices = center_aware(outputs, targets, epoch=45)['indices'][0][0].tolist()

    assert baseline_indices == [1, 3]
    assert center_indices == [0, 3]


def test_matcher_is_backward_compatible_when_center_targets_are_disabled():
    outputs = {
        'pred_logits': make_logits([21, 5]),
        'pred_boxes': torch.tensor(
            [[[0.50, 0.50, 0.20, 0.20], [0.80, 0.80, 0.20, 0.20]]],
            dtype=torch.float32,
        ),
    }
    targets = [{
        'labels': torch.tensor([21, 5], dtype=torch.int64),
        'boxes': torch.tensor(
            [[0.50, 0.50, 0.20, 0.20], [0.80, 0.80, 0.20, 0.20]],
            dtype=torch.float32,
        ),
    }]

    baseline = make_matcher(change_matcher=True, matcher_change_epoch=0)
    center_disabled = make_matcher(change_matcher=True, matcher_change_epoch=0, center_target_class_ids=[])

    baseline_indices = baseline(outputs, targets, epoch=45)['indices'][0]
    center_indices = center_disabled(outputs, targets, epoch=45)['indices'][0]

    assert torch.equal(baseline_indices[0], center_indices[0])
    assert torch.equal(baseline_indices[1], center_indices[1])


def test_yaml_configs_keep_existing_config_unchanged_and_add_new_center_config():
    old_cfg_path = REPO_ROOT / 'configs/deimv2/deimv2_dinov3_x_wholebody40_ins_s08_maskhead256x3.yml'
    new_cfg_path = REPO_ROOT / 'configs/deimv2/deimv2_dinov3_x_wholebody40_ins_s08_maskhead256x3_center.yml'

    old_cfg = YAMLConfig(str(old_cfg_path))
    new_cfg = YAMLConfig(str(new_cfg_path))

    old_criterion = old_cfg.criterion
    new_criterion = new_cfg.criterion

    assert old_cfg.yaml_cfg['output_dir'] != new_cfg.yaml_cfg['output_dir']
    assert getattr(old_criterion, 'center_target_class_ids') == []
    assert getattr(new_criterion, 'center_target_class_ids') == CENTER_CLASS_IDS
    assert new_criterion.weight_dict['loss_center'] == pytest.approx(3.0)
    assert new_criterion.matcher.center_target_class_ids == set(CENTER_CLASS_IDS)


def test_loading_old_extra_state_keeps_current_center_defaults():
    old_criterion = make_criterion()
    state = old_criterion.state_dict()
    state['_extra_state'].pop('center_target_class_ids', None)
    state['_extra_state'].pop('center_distance_min_radius', None)
    state['_extra_state'].pop('center_bbox_wh_weight', None)
    state['_extra_state'].pop('center_giou_weight', None)
    state['_extra_state'].pop('center_local_weight', None)

    new_criterion = make_criterion(
        center_target_class_ids=[21],
        center_distance_min_radius=0.02,
        center_bbox_wh_weight=0.25,
        center_giou_weight=0.25,
        center_local_weight=0.0,
    )
    new_criterion.load_state_dict(state, strict=True)

    assert new_criterion.center_target_class_ids == [21]
    assert new_criterion.center_distance_min_radius == pytest.approx(0.02)
    assert new_criterion.center_bbox_wh_weight == pytest.approx(0.25)
    assert new_criterion.center_giou_weight == pytest.approx(0.25)
    assert new_criterion.center_local_weight == pytest.approx(0.0)
