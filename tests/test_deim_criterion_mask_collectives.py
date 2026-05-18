import torch

import engine.deim.deim_criterion as deim_criterion_module
import engine.misc.dist_utils as dist_utils
from engine.deim.deim_criterion import DEIMCriterion


def test_loss_masks_participates_in_all_reduce_with_no_valid_local_masks(monkeypatch):
    all_reduce_calls = []

    def fake_all_reduce(tensor):
        all_reduce_calls.append(tensor.clone())

    monkeypatch.setattr(deim_criterion_module, "is_dist_available_and_initialized", lambda: True)
    monkeypatch.setattr(deim_criterion_module, "get_world_size", lambda: 3)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    criterion = DEIMCriterion(
        matcher=None,
        weight_dict={},
        losses=["masks"],
        num_classes=2,
    )
    outputs = {
        "mask_embed": torch.ones(1, 2, 4, requires_grad=True),
        "mask_features": torch.ones(1, 4, 2, 2, requires_grad=True),
    }
    targets = [
        {
            "labels": torch.tensor([1]),
            "masks": torch.zeros(1, 4, 4, dtype=torch.uint8),
            "mask_valid": torch.tensor([False]),
        }
    ]
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    losses = criterion.loss_masks(outputs, targets, indices, num_boxes=1)

    assert len(all_reduce_calls) == 1
    assert all_reduce_calls[0].tolist() == [0.0]
    assert losses["loss_mask_bce"].item() == 0.0
    assert losses["loss_mask_dice"].item() == 0.0


def test_loss_boxes_reports_zero_center_loss_without_local_center_targets():
    criterion = DEIMCriterion(
        matcher=None,
        weight_dict={},
        losses=["boxes"],
        num_classes=3,
        center_target_class_ids=[2],
    )
    outputs = {
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]]], requires_grad=True),
    }
    targets = [
        {
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        }
    ]
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    losses = criterion.loss_boxes(outputs, targets, indices, num_boxes=1)

    assert set(losses) == {"loss_bbox", "loss_giou", "loss_center"}
    assert losses["loss_center"].item() == 0.0


def test_reduce_dict_fills_missing_rank_keys(monkeypatch):
    reduced_tensors = []

    def fake_all_reduce(tensor):
        reduced_tensors.append(tensor.clone())

    monkeypatch.setattr(dist_utils, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist_utils, "all_gather", lambda keys: [["loss_a", "loss_b"], ["loss_a"]])
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    reduced = dist_utils.reduce_dict({"loss_a": torch.tensor(3.0)}, avg=False)

    assert list(reduced.keys()) == ["loss_a", "loss_b"]
    assert reduced["loss_a"].item() == 3.0
    assert reduced["loss_b"].item() == 0.0
    assert reduced_tensors[0].tolist() == [3.0, 0.0]
