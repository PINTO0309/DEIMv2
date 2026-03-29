from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def compute_resized_mask_output_size(
    spatial_size: Sequence[int],
    size: int | Sequence[int],
    max_size: int | None = None,
) -> tuple[int, int]:
    if len(spatial_size) != 2:
        raise ValueError(f'Expected spatial_size=(height, width), got {spatial_size}')

    height, width = int(spatial_size[0]), int(spatial_size[1])
    if isinstance(size, int):
        target_size = int(size)
        if max_size is not None:
            min_original_size = float(min(width, height))
            max_original_size = float(max(width, height))
            if max_original_size / min_original_size * target_size > max_size:
                target_size = int(round(max_size * min_original_size / max_original_size))

        if (width <= height and width == target_size) or (height <= width and height == target_size):
            return height, width

        if width < height:
            out_width = target_size
            out_height = int(target_size * height / width)
        else:
            out_height = target_size
            out_width = int(target_size * width / height)
        return out_height, out_width

    if len(size) == 1:
        return compute_resized_mask_output_size(spatial_size, int(size[0]), max_size=max_size)

    if len(size) != 2:
        raise ValueError(f'Expected size to have length 1 or 2, got {size}')

    return int(size[0]), int(size[1])


def _build_center_resize_grid(
    in_height: int,
    in_width: int,
    out_height: int,
    out_width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y_coords = (
        (torch.arange(out_height, device=device, dtype=dtype) + 0.5) - (out_height / 2.0)
    ) * (in_height / out_height) + (in_height / 2.0)
    x_coords = (
        (torch.arange(out_width, device=device, dtype=dtype) + 0.5) - (out_width / 2.0)
    ) * (in_width / out_width) + (in_width / 2.0)

    y_coords = y_coords.clamp(0.0, float(max(in_height - 1, 0)))
    x_coords = x_coords.clamp(0.0, float(max(in_width - 1, 0)))

    if in_height > 1:
        y_norm = (y_coords / (in_height - 1)) * 2.0 - 1.0
    else:
        y_norm = torch.zeros_like(y_coords)
    if in_width > 1:
        x_norm = (x_coords / (in_width - 1)) * 2.0 - 1.0
    else:
        x_norm = torch.zeros_like(x_coords)

    grid_y = y_norm[:, None].expand(out_height, out_width)
    grid_x = x_norm[None, :].expand(out_height, out_width)
    return torch.stack((grid_x, grid_y), dim=-1)


def resize_masks(
    masks: torch.Tensor,
    size: int | Sequence[int],
    *,
    max_size: int | None = None,
    mode: str = 'nearest',
    origin: str = 'center',
) -> torch.Tensor:
    if masks.ndim != 4:
        raise ValueError(f'Expected masks with shape [N, C, H, W], got {tuple(masks.shape)}')

    out_height, out_width = compute_resized_mask_output_size(masks.shape[-2:], size=size, max_size=max_size)
    in_height, in_width = masks.shape[-2:]
    if (in_height, in_width) == (out_height, out_width):
        return masks

    original_dtype = masks.dtype
    needs_float = (not torch.is_floating_point(masks)) or mode == 'bilinear' or (
        masks.device.type == 'cpu' and masks.dtype in (torch.float16, torch.bfloat16)
    )
    work_masks = masks.float() if needs_float else masks

    if origin == 'topleft':
        align_corners = False if mode == 'bilinear' else None
        resized = F.interpolate(
            work_masks,
            size=(out_height, out_width),
            mode=mode,
            align_corners=align_corners,
        )
    elif origin == 'center':
        sample_mode = 'bilinear' if mode == 'bilinear' else 'nearest'
        grid = _build_center_resize_grid(
            in_height,
            in_width,
            out_height,
            out_width,
            device=work_masks.device,
            dtype=work_masks.dtype,
        )
        grid = grid.unsqueeze(0).expand(work_masks.shape[0], -1, -1, -1)
        resized = F.grid_sample(
            work_masks,
            grid,
            mode=sample_mode,
            padding_mode='zeros',
            align_corners=True,
        )
    else:
        raise ValueError(f'Unsupported mask resize origin: {origin}')

    if original_dtype == torch.bool:
        return resized > 0.5

    if torch.is_floating_point(torch.empty((), dtype=original_dtype)):
        return resized.to(dtype=original_dtype) if resized.dtype != original_dtype else resized

    if mode in ('nearest', 'nearest-exact'):
        return resized.round().to(dtype=original_dtype)

    return resized
