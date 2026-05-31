"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import sys
import math
import time
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


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
    profile_train_steps = kwargs.get('profile_train_steps', None)
    profile_train_warmup = int(kwargs.get('profile_train_warmup', 3) or 0)
    profile_enabled = profile_train_steps is not None and int(profile_train_steps) > 0
    profile_train_steps = int(profile_train_steps or 0)
    profile_sync_cuda = profile_enabled and device.type == 'cuda'

    if profile_enabled:
        print(
            'Training profiler enabled: '
            f'steps={profile_train_steps}, warmup={profile_train_warmup}, sync_cuda={profile_sync_cuda}'
        )
        for name in [
            'prof_h2d',
            'prof_forward',
            'prof_criterion',
            'prof_backward_step',
            'prof_ema',
            'prof_scheduler',
            'prof_reduce_log',
            'prof_step_total',
        ]:
            metric_logger.add_meter(name, SmoothedValue(window_size=20, fmt='{avg:.4f}'))

    def _profile_now():
        if profile_sync_cuda:
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def _profile_record(name, start_time, step_index):
        if not profile_enabled:
            return
        elapsed = _profile_now() - start_time
        if step_index >= profile_train_warmup:
            metric_logger.update(**{name: elapsed})

    cur_iters = epoch * len(data_loader)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        profile_step_start = _profile_now() if profile_enabled else None
        profile_section_start = _profile_now() if profile_enabled else None
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        _profile_record('prof_h2d', profile_section_start, i)
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if use_amp:
            profile_section_start = _profile_now() if profile_enabled else None
            with torch.autocast(device_type=str(device), dtype=amp_dtype, cache_enabled=True):
                outputs = model(samples, targets=targets)
            _profile_record('prof_forward', profile_section_start, i)

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

            profile_section_start = _profile_now() if profile_enabled else None
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)
            _profile_record('prof_criterion', profile_section_start, i)

            loss = sum(loss_dict.values())
            profile_section_start = _profile_now() if profile_enabled else None
            if scaler is not None:
                scaler.scale(loss).backward()

                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                loss.backward()

                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                optimizer.step()
            _profile_record('prof_backward_step', profile_section_start, i)

        else:
            profile_section_start = _profile_now() if profile_enabled else None
            outputs = model(samples, targets=targets)
            _profile_record('prof_forward', profile_section_start, i)

            profile_section_start = _profile_now() if profile_enabled else None
            loss_dict = criterion(outputs, targets, **metas)
            _profile_record('prof_criterion', profile_section_start, i)

            loss : torch.Tensor = sum(loss_dict.values())
            profile_section_start = _profile_now() if profile_enabled else None
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
            _profile_record('prof_backward_step', profile_section_start, i)

        # ema
        profile_section_start = _profile_now() if profile_enabled else None
        if ema is not None:
            ema.update(model)
        _profile_record('prof_ema', profile_section_start, i)

        profile_section_start = _profile_now() if profile_enabled else None
        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()
        _profile_record('prof_scheduler', profile_section_start, i)

        profile_section_start = _profile_now() if profile_enabled else None
        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
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
        _profile_record('prof_reduce_log', profile_section_start, i)
        _profile_record('prof_step_total', profile_step_start, i)

        if profile_enabled and i + 1 >= profile_train_steps:
            print(f'Training profiler reached {profile_train_steps} steps; stopping train epoch early.')
            break

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
