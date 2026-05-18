import torch

import engine.deim.deim_criterion as deim_criterion_module
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
