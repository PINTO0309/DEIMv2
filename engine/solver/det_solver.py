"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import time
import json
import datetime
import re
from typing import List, Tuple

import torch

from ..misc import dist_utils, stats

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


class DetSolver(BaseSolver):

    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        for i, (name, param) in enumerate(self.model.named_parameters()):
            if i in [194, 195]:
                print(f"Index {i}: {name} - requires_grad: {param.requires_grad}")

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches,
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        n_parameters = sum([p.numel() for p in self.model.parameters() if not p.requires_grad])
        print(f'number of non-trainable parameters: {n_parameters}')

        top1 = 0
        ttop1 = 0
        best_stat = {'epoch': -1, }
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            self._report_validation(coco_evaluator, self.last_epoch)
            for k in test_stats:
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f'best_stat: {best_stat}')

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if not self.self_lr_scheduler:  # update by epoch
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            self._report_validation(coco_evaluator, epoch)

            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print['epoch'] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                best_stat_print[k] = max(best_stat[k], top1)
                print(f'best_stat: {best_stat_print}')  # global best

                if best_stat['epoch'] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                    ##### For fine-tuning
                    if self.cfg.tuning:
                        # Skip saving weights for the first epoch only
                        if epoch > 0 and test_stats[k][0] > ttop1:
                            ttop1 = test_stats[k][0]
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'tuning_best.pth')

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {'epoch': -1, }
                    self.ema.decay -= 0.0001
                    self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        self._report_validation(coco_evaluator, max(self.last_epoch, 0))

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return

    # --- helper methods for validation console output ---

    def _collect_coco_metrics(self, coco_eval) -> dict:
        stats = getattr(coco_eval, 'stats', None)
        if stats is None or len(stats) == 0:
            return {}

        labels = [
            ('map', 0),
            ('map_50', 1),
            ('map_75', 2),
            ('map_small', 3),
            ('map_medium', 4),
            ('map_large', 5),
            ('mar_1', 6),
            ('mar_10', 7),
            ('mar_100', 8),
            ('mar_small', 9),
            ('mar_medium', 10),
            ('mar_large', 11),
        ]

        metrics = {}
        for name, idx in labels:
            if idx < len(stats):
                metrics[name] = float(stats[idx])

        return metrics

    def _collect_per_class_ap(self, coco_eval) -> List[Tuple[int, float]]:
        eval_dict = getattr(coco_eval, 'eval', None)
        if not eval_dict or 'precision' not in eval_dict:
            return []

        precisions = eval_dict['precision']
        if precisions.size == 0:
            return []

        cat_ids = coco_eval.params.catIds
        if not cat_ids:
            return []

        dataset = getattr(self.val_dataloader, 'dataset', None)
        label2cat = getattr(dataset, 'label2category', None)

        if label2cat is not None:
            cat2label = {cat_id: label for label, cat_id in label2cat.items()}
            num_classes = len(label2cat)
            per_class = [float('nan')] * num_classes
        else:
            cat2label = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
            per_class = [float('nan')] * len(cat_ids)

        for cat_idx, cat_id in enumerate(cat_ids):
            label_idx = cat2label.get(cat_id)
            if label_idx is None:
                continue

            precision = precisions[:, :, cat_idx, 0, -1]
            valid = precision[precision > -1]
            if valid.size == 0:
                ap = float('nan')
            else:
                ap = float(valid.mean())
            per_class[label_idx] = ap

        return list(enumerate(per_class))

    def _resolve_class_names(self) -> List[str]:
        dataset = getattr(self.val_dataloader, 'dataset', None)
        if dataset is not None and hasattr(dataset, 'label2category') and hasattr(dataset, 'category2name'):
            label2category = dataset.label2category
            category2name = dataset.category2name
            names = []
            for label in range(len(label2category)):
                cat_id = label2category[label]
                names.append(category2name.get(cat_id, str(cat_id)))
            return names

        coco_eval = getattr(self.evaluator, 'coco_eval', None)
        if coco_eval and 'bbox' in coco_eval:
            gt = coco_eval['bbox'].cocoGt
            if hasattr(gt, 'cats'):
                cat_ids = coco_eval['bbox'].params.catIds
                return [gt.cats[cid]['name'] for cid in cat_ids]

        return []

    def _print_map_per_class_table(self, per_class, class_names, title: str):
        if not per_class:
            return

        max_id_digits = 3
        name_width = 25
        ap_width = 7
        max_rows_per_col = 20

        total_classes = len(per_class)
        # Adapt layout so long class lists spill into additional columns
        num_cols = max(1, (total_classes + max_rows_per_col - 1) // max_rows_per_col)
        col_height = max_rows_per_col

        def cell_text(idx: int, name: str, ap: float) -> str:
            name = name[:name_width] if name else ''
            ap_val = '-' if ap != ap else f"{ap:.4f}"
            return f"{str(idx).rjust(max_id_digits)}│{name.ljust(name_width)}│{ap_val.rjust(ap_width)}"

        def build_border(left: str, inner: str, between: str, right: str, fill: str) -> str:
            seg = (fill * max_id_digits) + inner + (fill * name_width) + inner + (fill * ap_width)
            return left + (seg + between) * (num_cols - 1) + seg + right

        top = build_border("┏", "┳", "┳", "┓", "━")
        header_sep = build_border("┡", "╇", "╇", "┩", "━")
        bottom = build_border("└", "┴", "┴", "┘", "─")

        def bold(text: str) -> str:
            return f"\033[1m{text}\033[0m"

        header_cell = f"{bold('ID'.rjust(max_id_digits))}┃{bold('Name'.ljust(name_width))}┃{bold('AP'.rjust(ap_width))}"

        def ljust_ansi(text: str, width: int) -> str:
            stripped = re.sub(r"\x1b\[[0-9;]*m", "", text)
            pad = max(0, width - len(stripped))
            return text + (' ' * pad)

        col_width = max_id_digits + 1 + name_width + 1 + ap_width
        header_row = "┃" + "┃".join(ljust_ansi(header_cell, col_width) for _ in range(num_cols)) + "┃"

        entries = []
        for idx, ap in per_class:
            name = class_names[idx] if idx < len(class_names) else ''
            entries.append(cell_text(idx, name, ap))

        empty_cell = (" " * max_id_digits) + "│" + (" " * name_width) + "│" + (" " * ap_width)
        col_lines = []
        for row in range(col_height):
            row_cells = []
            for col in range(num_cols):
                pos = col * col_height + row
                value = entries[pos] if pos < len(entries) else empty_cell
                row_cells.append(value.ljust(col_width))
            col_lines.append("│" + "│".join(row_cells) + "│")

        print(title)
        print("\n".join([top, header_row, header_sep] + col_lines + [bottom]))

    def _print_ap_ar_combined_table(self, metrics: dict, epoch: int):
        def fmt(value):
            return '-' if value is None or value != value else f"{value * 100:06.2f}"

        ap_rows = [
            ("AP @ .5:.95", metrics.get('map')),
            ("AP @     .5", metrics.get('map_50')),
            ("AP @    .75", metrics.get('map_75')),
            ("AP  (small)", metrics.get('map_small')),
            ("AP (medium)", metrics.get('map_medium')),
            ("AP  (large)", metrics.get('map_large')),
        ]

        ar_rows = [
            ("AR maxDets   1", metrics.get('mar_1')),
            ("AR maxDets  10", metrics.get('mar_10')),
            ("AR maxDets 100", metrics.get('mar_100')),
            ("AR     (small)", metrics.get('mar_small')),
            ("AR    (medium)", metrics.get('mar_medium')),
            ("AR     (large)", metrics.get('mar_large')),
        ]

        epoch_width = 5
        label_width = 16
        pct_width = 6

        def border(l, j1, j2, r, fill):
            seg = (fill * epoch_width) + j1 + (fill * label_width) + j1 + (fill * pct_width) + j2 + (fill * label_width) + j1 + (fill * pct_width)
            return l + seg + r

        top = border("┏", "┳", "┳", "┓", "━")
        mid = border("┡", "╇", "╇", "┩", "━")
        bottom = border("└", "┴", "┴", "┘", "─")

        header = f"┃{'Epoch'.rjust(epoch_width)}┃{'Avg. Precision'.ljust(label_width)}┃{'%'.rjust(pct_width)}╇{'Avg. Recall'.ljust(label_width)}┃{'%'.rjust(pct_width)}┃"

        lines = [top, header, mid]
        for ap, ar in zip(ap_rows, ar_rows):
            epoch_str = str(epoch).rjust(epoch_width)
            ap_label = ap[0].ljust(label_width)[:label_width]
            ar_label = ar[0].ljust(label_width)[:label_width]
            ap_pct = fmt(ap[1]).rjust(pct_width)
            ar_pct = fmt(ar[1]).rjust(pct_width)
            lines.append(f"│{epoch_str}│{ap_label}│{ap_pct}╎{ar_label}│{ar_pct}│")

        print("\n".join(lines + [bottom]))
    def _report_validation(self, coco_evaluator, epoch=None):
        if not dist_utils.is_main_process():
            return

        if coco_evaluator is None or 'bbox' not in coco_evaluator.coco_eval:
            return

        coco_eval = coco_evaluator.coco_eval['bbox']
        metrics = self._collect_coco_metrics(coco_eval)

        if metrics:
            epoch_value = 0 if epoch is None else epoch
            self._print_ap_ar_combined_table(metrics, epoch_value)

        per_class = self._collect_per_class_ap(coco_eval)
        if per_class:
            class_names = self._resolve_class_names()
            title = "Per-class mAP:" if metrics else "Per-class mAP (inference):"
            self._print_map_per_class_table(per_class, class_names, title)
