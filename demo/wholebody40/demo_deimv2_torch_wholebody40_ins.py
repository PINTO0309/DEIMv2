"""
PyTorch checkpoint demo for DEIMv2 wholebody40 instance segmentation.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageColor, ImageDraw, ImageFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from engine.core import YAMLConfig


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
BODY_CLASS_ID = 0
DEFAULT_CONFIG = 'configs/deimv2/deimv2_dinov3_x_wholebody40_ins.yml'


def load_class_names(path: Path) -> Dict[int, str]:
    class_names: Dict[int, str] = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_id, class_name = line.split(maxsplit=1)
            class_names[int(class_id)] = class_name
    return class_names


def make_class_colors(class_ids: Sequence[int]) -> Dict[int, Tuple[int, int, int]]:
    palette = [
        '#ff6b6b', '#4ecdc4', '#ffe66d', '#1a535c', '#ff9f1c',
        '#5f0f40', '#9a031e', '#fb8b24', '#0f4c5c', '#2ec4b6',
        '#3a86ff', '#8338ec', '#ff006e', '#8ac926', '#1982c4',
        '#6a4c93', '#e76f51', '#2a9d8f', '#e9c46a', '#264653',
    ]
    colors: Dict[int, Tuple[int, int, int]] = {}
    ids = sorted(set(class_ids))
    for idx, class_id in enumerate(ids):
        colors[class_id] = ImageColor.getrgb(palette[idx % len(palette)])
    return colors


def list_image_paths(images_dir: Path) -> List[Path]:
    image_paths = [
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths, key=lambda path: path.name)


def load_checkpoint_state(resume_path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(resume_path, map_location='cpu')
    if 'ema' in checkpoint and isinstance(checkpoint['ema'], dict) and 'module' in checkpoint['ema']:
        return checkpoint['ema']['module']
    if 'model' in checkpoint:
        return checkpoint['model']
    raise KeyError(f'Checkpoint {resume_path} does not contain `ema.module` or `model`.')


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_transform(image_size: Sequence[int], normalize: bool) -> T.Compose:
    ops: List[object] = [
        T.Resize(tuple(image_size)),
        T.ToTensor(),
    ]
    if normalize:
        ops.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(ops)


def binary_mask_bbox(mask: np.ndarray) -> List[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def prepare_prediction_payload(
    result: Dict[str, torch.Tensor],
    score_threshold: float,
    mask_threshold: float,
) -> List[Dict[str, object]]:
    labels = result['labels'].detach().cpu()
    scores = result['scores'].detach().cpu()
    boxes = result['boxes'].detach().cpu()
    masks = result.get('masks')
    if masks is not None:
        masks = masks.detach().cpu()

    records: List[Dict[str, object]] = []
    for idx in range(len(labels)):
        score = float(scores[idx].item())
        if score < score_threshold:
            continue

        label = int(labels[idx].item())
        box = [float(v) for v in boxes[idx].tolist()]
        record: Dict[str, object] = {
            'label': label,
            'score': score,
            'box_xyxy': box,
        }

        if masks is not None and label == BODY_CLASS_ID:
            binary_mask = masks[idx, 0].numpy() >= mask_threshold
            mask_bbox = binary_mask_bbox(binary_mask)
            if mask_bbox is not None:
                record['mask_area'] = int(binary_mask.sum())
                record['mask_bbox'] = mask_bbox

        records.append(record)
    return records


def overlay_body_masks(
    image: Image.Image,
    result: Dict[str, torch.Tensor],
    score_threshold: float,
    mask_threshold: float,
    body_color: Tuple[int, int, int],
    disable_render_classids: set[int],
) -> Image.Image:
    masks = result.get('masks')
    if masks is None or BODY_CLASS_ID in disable_render_classids:
        return image

    labels = result['labels'].detach().cpu()
    scores = result['scores'].detach().cpu()
    masks = masks.detach().cpu()

    overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    for idx in range(len(labels)):
        if int(labels[idx].item()) != BODY_CLASS_ID:
            continue
        if float(scores[idx].item()) < score_threshold:
            continue
        binary_mask = masks[idx, 0].numpy() >= mask_threshold
        if not binary_mask.any():
            continue
        overlay[binary_mask] = np.array([body_color[0], body_color[1], body_color[2], 96], dtype=np.uint8)

    if overlay[..., 3].max() == 0:
        return image

    base = image.convert('RGBA')
    mask_image = Image.fromarray(overlay, mode='RGBA')
    return Image.alpha_composite(base, mask_image).convert('RGB')


def draw_detections(
    image: Image.Image,
    result: Dict[str, torch.Tensor],
    class_names: Dict[int, str],
    class_colors: Dict[int, Tuple[int, int, int]],
    score_threshold: float,
    disable_render_classids: set[int],
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    labels = result['labels'].detach().cpu()
    scores = result['scores'].detach().cpu()
    boxes = result['boxes'].detach().cpu()

    for idx in range(len(labels)):
        score = float(scores[idx].item())
        if score < score_threshold:
            continue

        class_id = int(labels[idx].item())
        if class_id in disable_render_classids:
            continue

        color = class_colors.get(class_id, (255, 255, 255))
        x1, y1, x2, y2 = [int(round(v)) for v in boxes[idx].tolist()]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        class_name = class_names.get(class_id, str(class_id))
        label_text = f'{class_name} {score:.2f}'
        text_bbox = draw.textbbox((x1, y1), label_text, font=font)
        text_h = text_bbox[3] - text_bbox[1]
        text_w = text_bbox[2] - text_bbox[0]
        bg_y1 = max(0, y1 - text_h - 4)
        bg_y2 = bg_y1 + text_h + 4
        draw.rectangle([x1, bg_y1, x1 + text_w + 6, bg_y2], fill=color)
        draw.text((x1 + 3, bg_y1 + 2), label_text, fill='black', font=font)

    return image


class InferenceModel(nn.Module):
    def __init__(self, cfg: YAMLConfig, state_dict: Dict[str, torch.Tensor], device: torch.device):
        super().__init__()
        cfg.model.load_state_dict(state_dict)
        self.model = cfg.model.eval().to(device)
        self.postprocessor = cfg.postprocessor.eval().to(device)
        self.device = device

    @torch.inference_mode()
    def forward(self, image_tensor: torch.Tensor, orig_target_sizes: torch.Tensor):
        outputs = self.model(image_tensor)
        return self.postprocessor(outputs, orig_target_sizes)


def save_predictions_json(output_dir: Path, image_path: Path, records: List[Dict[str, object]]) -> None:
    pred_dir = output_dir / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'image': image_path.name,
        'predictions': records,
    }
    with (pred_dir / f'{image_path.stem}.json').open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def process_images(args) -> None:
    config_path = Path(args.config)
    resume_path = Path(args.resume)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    if not resume_path.exists():
        raise FileNotFoundError(f'Checkpoint file not found: {resume_path}')
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f'Image directory not found: {images_dir}')

    image_paths = list_image_paths(images_dir)
    if not image_paths:
        raise FileNotFoundError(f'No image files found in {images_dir}')

    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = YAMLConfig(str(config_path), resume=str(resume_path))
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    if 'DINOv3STAs' in cfg.yaml_cfg:
        cfg.yaml_cfg['DINOv3STAs']['weights_path'] = None

    state_dict = load_checkpoint_state(resume_path)
    device = resolve_device(args.device)
    model = InferenceModel(cfg, state_dict, device)

    image_size = cfg.yaml_cfg['eval_spatial_size']
    normalize = bool(cfg.yaml_cfg.get('DINOv3STAs', False))
    transform = build_transform(image_size, normalize)

    class_names = load_class_names(Path(__file__).with_name('classes.txt'))
    class_colors = make_class_colors(class_names.keys())
    body_color = class_colors.get(BODY_CLASS_ID, (255, 107, 107))
    disable_render_classids = set(args.disable_render_classids)

    print(f'Processing {len(image_paths)} images from {images_dir}')
    print(f'Using checkpoint: {resume_path}')
    print(f'Output directory: {output_dir}')

    for idx, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = image.size
        orig_target_sizes = torch.tensor([[orig_w, orig_h]], dtype=torch.float32, device=device)
        image_tensor = transform(image).unsqueeze(0).to(device)

        results = model(image_tensor, orig_target_sizes)
        result = results[0]

        rendered = overlay_body_masks(
            image.copy(),
            result=result,
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
            body_color=body_color,
            disable_render_classids=disable_render_classids,
        )
        rendered = draw_detections(
            rendered,
            result=result,
            class_names=class_names,
            class_colors=class_colors,
            score_threshold=args.score_threshold,
            disable_render_classids=disable_render_classids,
        )
        rendered.save(output_dir / image_path.name)

        if args.save_raw_predictions:
            records = prepare_prediction_payload(
                result=result,
                score_threshold=args.score_threshold,
                mask_threshold=args.mask_threshold,
            )
            save_predictions_json(output_dir, image_path, records)

        if idx % 50 == 0 or idx == len(image_paths):
            print(f'Processed {idx}/{len(image_paths)}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=DEFAULT_CONFIG)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--images_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default=None)
    parser.add_argument('--score_threshold', type=float, default=0.35)
    parser.add_argument('--mask_threshold', type=float, default=0.5)
    parser.add_argument('--disable_render_classids', type=int, nargs='*', default=[])
    parser.add_argument('--save_raw_predictions', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    process_images(parse_args())
