"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import sys
import math
import os
import time
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


def _ddp_debug_enabled() -> bool:
    return os.getenv('DEIM_DEBUG_DDP', '').lower() in ('1', 'true', 'yes', 'on')


def _maybe_sync_for_debug():
    if _ddp_debug_enabled() and torch.cuda.is_available():
        torch.cuda.synchronize()


def _target_debug_summary(targets):
    image_ids = []
    dataset_indices = []
    num_labels = []
    for target in targets:
        image_id = target.get('image_id')
        idx = target.get('idx')
        labels = target.get('labels')
        if torch.is_tensor(image_id) and image_id.numel() > 0:
            image_ids.append(int(image_id.flatten()[0].item()))
        if torch.is_tensor(idx) and idx.numel() > 0:
            dataset_indices.append(int(idx.flatten()[0].item()))
        if torch.is_tensor(labels):
            num_labels.append(int(labels.numel()))
    return f'image_ids={image_ids}, dataset_idx={dataset_indices}, num_labels={num_labels}'


def _log_ddp_phase(phase: str, epoch: int, step: int, started_at=None, targets=None):
    if not _ddp_debug_enabled():
        return

    _maybe_sync_for_debug()
    elapsed = None if started_at is None else time.time() - started_at
    slow_threshold = float(os.getenv('DEIM_DEBUG_DDP_SLOW_SECONDS', '30'))
    always = os.getenv('DEIM_DEBUG_DDP_VERBOSE', '').lower() in ('1', 'true', 'yes', 'on')
    if elapsed is not None and elapsed < slow_threshold and not always:
        return

    message = f'[rank{dist_utils.get_rank()}] epoch={epoch} step={step} phase={phase}'
    if elapsed is not None:
        message += f' elapsed={elapsed:.3f}s'
    if targets is not None:
        message += ' ' + _target_debug_summary(targets)
    try:
        print(message, force=True)
    except TypeError:
        print(message)


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    use_amp: bool = kwargs.get('use_amp', False)
    amp_dtype = kwargs.get('amp_dtype', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    cur_iters = epoch * len(data_loader)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        _log_ddp_phase('batch_loaded', epoch, i, targets=targets)
        phase_start = time.time()
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        _log_ddp_phase('to_device', epoch, i, phase_start, targets=targets)
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if use_amp:
            phase_start = time.time()
            with torch.autocast(device_type=str(device), dtype=amp_dtype, cache_enabled=True):
                outputs = model(samples, targets=targets)
            _log_ddp_phase('forward', epoch, i, phase_start, targets=targets)

            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace('module.', '')
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError('Non-finite pred_boxes detected during AMP forward pass.')

            phase_start = time.time()
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)
            _log_ddp_phase('criterion', epoch, i, phase_start, targets=targets)

            loss = sum(loss_dict.values())
            if scaler is not None:
                phase_start = time.time()
                scaler.scale(loss).backward()
                _log_ddp_phase('backward', epoch, i, phase_start, targets=targets)

                if max_norm > 0:
                    phase_start = time.time()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    _log_ddp_phase('clip_grad', epoch, i, phase_start, targets=targets)

                phase_start = time.time()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                _log_ddp_phase('optimizer', epoch, i, phase_start, targets=targets)
            else:
                optimizer.zero_grad()
                phase_start = time.time()
                loss.backward()
                _log_ddp_phase('backward', epoch, i, phase_start, targets=targets)

                if max_norm > 0:
                    phase_start = time.time()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    _log_ddp_phase('clip_grad', epoch, i, phase_start, targets=targets)

                phase_start = time.time()
                optimizer.step()
                _log_ddp_phase('optimizer', epoch, i, phase_start, targets=targets)

        else:
            phase_start = time.time()
            outputs = model(samples, targets=targets)
            _log_ddp_phase('forward', epoch, i, phase_start, targets=targets)
            phase_start = time.time()
            loss_dict = criterion(outputs, targets, **metas)
            _log_ddp_phase('criterion', epoch, i, phase_start, targets=targets)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            phase_start = time.time()
            loss.backward()
            _log_ddp_phase('backward', epoch, i, phase_start, targets=targets)

            if max_norm > 0:
                phase_start = time.time()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                _log_ddp_phase('clip_grad', epoch, i, phase_start, targets=targets)

            phase_start = time.time()
            optimizer.step()
            _log_ddp_phase('optimizer', epoch, i, phase_start, targets=targets)

        # ema
        if ema is not None:
            phase_start = time.time()
            ema.update(model)
            _log_ddp_phase('ema', epoch, i, phase_start, targets=targets)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        phase_start = time.time()
        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        _log_ddp_phase('loss_reduce', epoch, i, phase_start, targets=targets)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
        center_summary = getattr(coco_evaluator, 'center_eval', {}).get('summary', {})
        if center_summary:
            for key, value in center_summary.items():
                stats[f'center_eval_{key.replace("@", "_at_").replace(".", "_")}'] = value

    return stats, coco_evaluator
