import torch

from engine.data._misc import Image
from engine.data.transforms._transforms import RandomHorizontalFlipWithClass


def test_random_horizontal_flip_with_class_flips_image_and_swaps_labels():
    transform = RandomHorizontalFlipWithClass(p=1.0, class_pairs=[[9, 15], [10, 14]])
    image = Image(torch.arange(12, dtype=torch.float32).reshape(1, 3, 4))
    target = {"labels": torch.tensor([9, 15, 10, 7])}

    flipped_image, flipped_target = transform(image, target)

    assert flipped_target["labels"].tolist() == [15, 9, 14, 7]
    assert torch.equal(torch.as_tensor(flipped_image), torch.flip(torch.as_tensor(image), dims=[-1]))


def test_random_horizontal_flip_with_class_supports_legacy_torchvision_hooks():
    transform = RandomHorizontalFlipWithClass(p=0.0)
    seen = []

    transform.check_inputs = None
    transform.make_params = None
    transform.transform = None
    transform._check_inputs = lambda flat_inputs: seen.extend(flat_inputs)
    transform._get_params = lambda flat_inputs: {"count": len(flat_inputs)}
    transform._transform = lambda inpt, params: (inpt, params["count"])

    sentinel = object()
    transform._check_inputs_compat([sentinel])
    params = transform._make_params_compat([sentinel])
    transformed = transform._transform_compat(sentinel, params)

    assert seen == [sentinel]
    assert params == {"count": 1}
    assert transformed == (sentinel, 1)
