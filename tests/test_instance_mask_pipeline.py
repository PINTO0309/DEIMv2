import pytest
import torch

from PIL import Image

from engine.data._misc import convert_to_tv_tensor
from engine.data.dataloader import BatchImageCollateFunction
from engine.data.transforms import Compose


def test_sanitize_bounding_boxes_keeps_instance_mask_metadata_in_sync():
    image = Image.new('RGB', (100, 100), color=0)
    target = {
        'boxes': convert_to_tv_tensor(
            torch.tensor([[10, 10, 10, 20], [40, 40, 80, 80]], dtype=torch.float32),
            'boxes',
            spatial_size=(100, 100),
        ),
        'labels': torch.tensor([3, 7], dtype=torch.int64),
        'masks': convert_to_tv_tensor(torch.zeros((2, 100, 100), dtype=torch.uint8), 'masks'),
        'area': torch.tensor([0.0, 1600.0], dtype=torch.float32),
        'iscrowd': torch.tensor([0, 1], dtype=torch.int64),
        'mask_valid': torch.tensor([True, False], dtype=torch.bool),
        'segm_eval_valid': torch.tensor([True, False], dtype=torch.bool),
    }

    transform = Compose(
        ops=[{'type': 'SanitizeBoundingBoxes', 'min_size': 1}],
        policy=None,
    )

    _, sanitized, _ = transform(image, target, type('DatasetState', (), {'epoch': 0})())

    assert sanitized['boxes'].shape[0] == 1
    assert sanitized['labels'].tolist() == [7]
    assert sanitized['masks'].shape[0] == 1
    assert sanitized['area'].tolist() == [1600.0]
    assert sanitized['iscrowd'].tolist() == [1]
    assert sanitized['mask_valid'].tolist() == [False]
    assert sanitized['segm_eval_valid'].tolist() == [False]


@pytest.mark.parametrize(
    ('kwargs', 'expected_message'),
    [
        (
            {'mixup_prob': 0.5, 'mixup_epochs': [0, 1]},
            'MixUp is not supported when instance mask supervision is enabled.',
        ),
        (
            {'copyblend_prob': 0.5, 'copyblend_epochs': [0, 1]},
            'CopyBlend is not supported when instance mask supervision is enabled.',
        ),
    ],
)
def test_collate_rejects_unsupported_instance_mask_augmentations(kwargs, expected_message):
    collate = BatchImageCollateFunction(**kwargs)
    collate.set_epoch(0)

    image = torch.zeros((3, 32, 32), dtype=torch.float32)
    target = {
        'boxes': torch.zeros((1, 4), dtype=torch.float32),
        'labels': torch.zeros((1,), dtype=torch.int64),
        'masks': torch.zeros((1, 32, 32), dtype=torch.uint8),
        'mask_valid': torch.ones((1,), dtype=torch.bool),
        'area': torch.ones((1,), dtype=torch.float32),
        'iscrowd': torch.zeros((1,), dtype=torch.int64),
    }

    with pytest.raises(RuntimeError, match=expected_message):
        collate([(image, target), (image.clone(), {k: v.clone() for k, v in target.items()})])
