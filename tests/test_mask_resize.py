import torch
import torch.nn.functional as F

from engine.data._misc import Mask
from engine.data.transforms._transforms import Resize
from engine.deim.postprocessor import PostProcessor
from engine.misc.mask_resize import resize_masks


def test_resize_masks_identity_preserves_bool_dtype():
    masks = torch.zeros((2, 1, 5, 7), dtype=torch.bool)
    masks[0, 0, 1:4, 2:6] = True

    resized = resize_masks(masks, size=(5, 7), mode='nearest', origin='center')

    assert resized.dtype == torch.bool
    assert torch.equal(resized, masks)


def test_resize_masks_center_differs_from_topleft():
    masks = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)

    center = resize_masks(masks, size=(7, 7), mode='bilinear', origin='center')
    topleft = resize_masks(masks, size=(7, 7), mode='bilinear', origin='topleft')

    assert center.shape == (1, 1, 7, 7)
    assert topleft.shape == (1, 1, 7, 7)
    assert not torch.allclose(center, topleft)


def test_postprocessor_respects_mask_resize_origin():
    outputs = {
        'pred_logits': torch.tensor([[[10.0]]], dtype=torch.float32),
        'pred_boxes': torch.tensor([[[0.5, 0.5, 1.0, 1.0]]], dtype=torch.float32),
        'pred_masks': torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3),
    }
    orig_target_sizes = torch.tensor([[7.0, 7.0]], dtype=torch.float32)

    center_post = PostProcessor(num_classes=1, num_top_queries=1, mask_resize_origin='center')
    topleft_post = PostProcessor(num_classes=1, num_top_queries=1, mask_resize_origin='topleft')

    center_masks = center_post(outputs, orig_target_sizes)[0]['masks']
    topleft_masks = topleft_post(outputs, orig_target_sizes)[0]['masks']

    assert center_masks.shape == (1, 1, 7, 7)
    assert topleft_masks.shape == (1, 1, 7, 7)
    assert not torch.allclose(center_masks, topleft_masks)


def test_resize_transform_uses_center_origin_for_masks():
    mask_tensor = torch.zeros((1, 5, 5), dtype=torch.bool)
    mask_tensor[0, 1:4, 2:5] = True

    transform = Resize(size=(7, 7), mask_resize_origin='center')
    resized = transform(Mask(mask_tensor))
    expected = resize_masks(mask_tensor[:, None], size=(7, 7), mode='nearest', origin='center')[:, 0]

    assert resized.shape == expected.shape
    assert torch.equal(torch.as_tensor(resized), expected)


def test_resize_masks_center_nearest_matches_previous_grid_sample_behavior():
    masks = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)

    y_coords = ((torch.arange(7, dtype=torch.float32) + 0.5) - 3.5) * (5 / 7) + 2.5
    x_coords = ((torch.arange(7, dtype=torch.float32) + 0.5) - 3.5) * (5 / 7) + 2.5
    y_coords = y_coords.clamp(0.0, 4.0)
    x_coords = x_coords.clamp(0.0, 4.0)
    y_norm = (y_coords / 4.0) * 2.0 - 1.0
    x_norm = (x_coords / 4.0) * 2.0 - 1.0
    grid = torch.stack(
        (
            x_norm[None, :].expand(7, 7),
            y_norm[:, None].expand(7, 7),
        ),
        dim=-1,
    ).unsqueeze(0)
    expected = F.grid_sample(
        masks,
        grid,
        mode='nearest',
        padding_mode='zeros',
        align_corners=True,
    )

    actual = resize_masks(masks, size=(7, 7), mode='nearest', origin='center')

    assert torch.equal(actual, expected)
