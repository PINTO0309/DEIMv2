"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import math
import numpy as np
import torch

from faster_coco_eval import COCO, COCOeval_faster
import faster_coco_eval.core.mask as mask_util
from ...core import register
from ...misc import dist_utils
__all__ = ['CocoEvaluator',]


@register()
class CocoEvaluator(object):
    __share__ = [
        'segm_eval_category_ids',
        'segm_ignore_missing_masks',
        'center_eval_class_ids',
        'center_eval_thresholds',
        'center_eval_primary_threshold',
        'center_eval_normalizer',
        'center_eval_score_threshold',
    ]

    def __init__(
        self,
        coco_gt,
        iou_types,
        segm_eval_category_ids=None,
        segm_ignore_missing_masks=True,
        center_eval_class_ids=None,
        center_eval_thresholds=None,
        center_eval_primary_threshold=0.5,
        center_eval_normalizer='half_diagonal',
        center_eval_score_threshold=0.0,
    ):
        assert isinstance(iou_types, (list, tuple))
        if isinstance(coco_gt, dict):
            self.coco_gt = {k: copy.deepcopy(v) for k, v in coco_gt.items()}
        else:
            self.coco_gt = copy.deepcopy(coco_gt)
        self.iou_types = iou_types
        self.segm_eval_category_ids = [] if segm_eval_category_ids is None else list(segm_eval_category_ids)
        self.segm_ignore_missing_masks = segm_ignore_missing_masks
        self.center_eval_class_ids = [] if center_eval_class_ids is None else [int(x) for x in center_eval_class_ids]
        self.center_eval_thresholds = [0.25, 0.5, 1.0] if center_eval_thresholds is None else [float(x) for x in center_eval_thresholds]
        self.center_eval_primary_threshold = float(center_eval_primary_threshold)
        self.center_eval_normalizer = str(center_eval_normalizer)
        self.center_eval_score_threshold = float(center_eval_score_threshold)
        if self.center_eval_normalizer != 'half_diagonal':
            raise ValueError(f'Unsupported center_eval_normalizer={center_eval_normalizer}')

        self.coco_eval = {}
        for iou_type in iou_types:
            coco_gt_iou = self._get_coco_gt_for_iou(iou_type)
            self.coco_eval[iou_type] = COCOeval_faster(coco_gt_iou, iouType=iou_type, print_function=print, separate_eval=True)
            if iou_type == 'segm' and self.segm_eval_category_ids:
                self.coco_eval[iou_type].params.catIds = list(self.segm_eval_category_ids)

        self.img_ids = {k: [] for k in iou_types}
        self.eval_imgs = {k: [] for k in iou_types}
        self.center_eval_records = {}
        self.center_eval = self._build_empty_center_eval()

    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            coco_gt_iou = self._get_coco_gt_for_iou(iou_type)
            self.coco_eval[iou_type] = COCOeval_faster(coco_gt_iou, iouType=iou_type, print_function=print, separate_eval=True)
            if iou_type == 'segm' and self.segm_eval_category_ids:
                self.coco_eval[iou_type].params.catIds = list(self.segm_eval_category_ids)
        self.img_ids = {k: [] for k in self.iou_types}
        self.eval_imgs = {k: [] for k in self.iou_types}
        self.center_eval_records = {}
        self.center_eval = self._build_empty_center_eval()

    def _get_coco_gt_for_iou(self, iou_type):
        if isinstance(self.coco_gt, dict):
            if iou_type in self.coco_gt:
                return self.coco_gt[iou_type]
            if 'bbox' in self.coco_gt:
                return self.coco_gt['bbox']
        return self.coco_gt

    def _build_empty_center_eval(self):
        return {
            'summary': {},
            'per_class': {},
            'thresholds': list(self.center_eval_thresholds),
            'class_ids': list(self.center_eval_class_ids),
            'primary_threshold': self.center_eval_primary_threshold,
            'normalizer': self.center_eval_normalizer,
            'score_threshold': self.center_eval_score_threshold,
        }

    def _targeted_predictions(self, prediction):
        labels = prediction['labels'].detach().cpu()
        scores = prediction['scores'].detach().cpu()
        boxes = prediction['boxes'].detach().cpu()
        if not self.center_eval_class_ids:
            return labels.new_zeros((0,), dtype=torch.int64), scores.new_zeros((0,), dtype=torch.float32), boxes.new_zeros((0, 4), dtype=torch.float32)

        keep = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in self.center_eval_class_ids:
            keep |= labels == class_id
        keep &= scores >= self.center_eval_score_threshold
        return labels[keep], scores[keep], boxes[keep]

    def _build_center_eval_record(self, original_id, prediction, coco_gt_bbox):
        labels, scores, boxes = self._targeted_predictions(prediction)
        pred_records = []
        if labels.numel() > 0:
            cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
            cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
            for label, score, center_x, center_y in zip(labels.tolist(), scores.tolist(), cx.tolist(), cy.tolist()):
                pred_records.append({
                    'category_id': int(label),
                    'score': float(score),
                    'center': [float(center_x), float(center_y)],
                })

        gt_records = []
        anns = coco_gt_bbox.imgToAnns.get(original_id, []) if hasattr(coco_gt_bbox, 'imgToAnns') else []
        for ann in anns:
            if int(ann.get('iscrowd', 0)) != 0:
                continue
            category_id = int(ann['category_id'])
            if category_id not in self.center_eval_class_ids:
                continue
            bbox = ann['bbox']
            gt_records.append({
                'category_id': category_id,
                'center': [float(bbox[0] + 0.5 * bbox[2]), float(bbox[1] + 0.5 * bbox[3])],
                'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            })

        self.center_eval_records[int(original_id)] = {
            'predictions': pred_records,
            'targets': gt_records,
        }

    @staticmethod
    def _merge_center_eval_records(record_shards):
        merged = {}
        for shard in record_shards:
            merged.update(shard)
        return merged

    def _normalized_center_distance(self, pred_center, gt_center, gt_bbox):
        dx = float(pred_center[0]) - float(gt_center[0])
        dy = float(pred_center[1]) - float(gt_center[1])
        distance = math.hypot(dx, dy)
        width = float(gt_bbox[2])
        height = float(gt_bbox[3])
        normalizer = max(0.5 * math.hypot(width, height), 1e-6)
        return distance / normalizer

    def _match_center_records(self, predictions, targets, threshold):
        if not predictions:
            return 0, 0, len(targets), []
        predictions = sorted(predictions, key=lambda item: item['score'], reverse=True)
        matched_targets = set()
        tp = 0
        fp = 0
        matched_distances = []
        for pred in predictions:
            best_idx = None
            best_distance = None
            for idx, target in enumerate(targets):
                if idx in matched_targets or int(target['category_id']) != int(pred['category_id']):
                    continue
                distance = self._normalized_center_distance(pred['center'], target['center'], target['bbox'])
                if distance <= threshold and (best_distance is None or distance < best_distance):
                    best_distance = distance
                    best_idx = idx

            if best_idx is None:
                fp += 1
                continue

            matched_targets.add(best_idx)
            tp += 1
            matched_distances.append(best_distance)

        fn = len(targets) - len(matched_targets)
        return tp, fp, fn, matched_distances

    @staticmethod
    def _compute_prf(tp, fp, fn):
        precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        return precision, recall, f1

    def _accumulate_center_eval(self):
        center_eval = self._build_empty_center_eval()
        if not self.center_eval_class_ids:
            return center_eval

        thresholds = list(self.center_eval_thresholds)
        primary_threshold = self.center_eval_primary_threshold
        if primary_threshold not in thresholds:
            thresholds.append(primary_threshold)
            thresholds = sorted(set(thresholds))

        summary = {}
        aggregate_by_threshold = {
            threshold: {'tp': 0, 'fp': 0, 'fn': 0, 'distances': []}
            for threshold in thresholds
        }
        per_class_primary = {
            class_id: {'tp': 0, 'fp': 0, 'fn': 0, 'distances': []}
            for class_id in self.center_eval_class_ids
        }

        pred_count = 0
        gt_count = 0
        for record in self.center_eval_records.values():
            predictions = list(record['predictions'])
            targets = list(record['targets'])
            pred_count += len(predictions)
            gt_count += len(targets)
            for threshold in thresholds:
                tp, fp, fn, distances = self._match_center_records(predictions, targets, threshold)
                aggregate = aggregate_by_threshold[threshold]
                aggregate['tp'] += tp
                aggregate['fp'] += fp
                aggregate['fn'] += fn
                if threshold == primary_threshold:
                    aggregate['distances'].extend(distances)

            for class_id in self.center_eval_class_ids:
                pred_subset = [pred for pred in predictions if int(pred['category_id']) == class_id]
                target_subset = [target for target in targets if int(target['category_id']) == class_id]
                tp, fp, fn, distances = self._match_center_records(pred_subset, target_subset, primary_threshold)
                class_stats = per_class_primary[class_id]
                class_stats['tp'] += tp
                class_stats['fp'] += fp
                class_stats['fn'] += fn
                class_stats['distances'].extend(distances)

        for threshold in thresholds:
            aggregate = aggregate_by_threshold[threshold]
            precision, recall, f1 = self._compute_prf(aggregate['tp'], aggregate['fp'], aggregate['fn'])
            threshold_suffix = f'@{threshold:g}'
            summary[f'precision{threshold_suffix}'] = precision
            summary[f'recall{threshold_suffix}'] = recall
            summary[f'f1{threshold_suffix}'] = f1

        primary_distances = aggregate_by_threshold[primary_threshold]['distances']
        summary['median_distance'] = float(np.median(primary_distances)) if primary_distances else float('nan')
        summary['mean_distance'] = float(np.mean(primary_distances)) if primary_distances else float('nan')
        summary['matched_count'] = int(aggregate_by_threshold[primary_threshold]['tp'])
        summary['gt_count'] = int(gt_count)
        summary['pred_count'] = int(pred_count)

        center_eval['summary'] = summary
        center_eval['per_class'] = {}
        for class_id, class_stats in per_class_primary.items():
            precision, recall, f1 = self._compute_prf(class_stats['tp'], class_stats['fp'], class_stats['fn'])
            distances = class_stats['distances']
            center_eval['per_class'][int(class_id)] = {
                f'precision@{primary_threshold:g}': precision,
                f'recall@{primary_threshold:g}': recall,
                f'f1@{primary_threshold:g}': f1,
                'median_distance': float(np.median(distances)) if distances else float('nan'),
                'matched_count': int(class_stats['tp']),
                'gt_count': int(class_stats['tp'] + class_stats['fn']),
                'pred_count': int(class_stats['tp'] + class_stats['fp']),
            }

        return center_eval


    def update(self, predictions):
        for iou_type in self.iou_types:
            coco_eval = self.coco_eval[iou_type]
            coco_gt_iou = self._get_coco_gt_for_iou(iou_type)
            valid_img_ids = set(coco_gt_iou.imgs.keys()) if hasattr(coco_gt_iou, 'imgs') else None
            predictions_iou = predictions if valid_img_ids is None else {
                img_id: pred for img_id, pred in predictions.items() if img_id in valid_img_ids
            }
            img_ids = list(np.unique(list(predictions_iou.keys())))
            if not img_ids:
                continue
            self.img_ids[iou_type].extend(img_ids)
            results = self.prepare(predictions_iou, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = coco_gt_iou.loadRes(results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)
                    coco_eval.evaluate()

            self.eval_imgs[iou_type].append(np.array(coco_eval._evalImgs_cpp).reshape(len(coco_eval.params.catIds), len(coco_eval.params.areaRng), len(coco_eval.params.imgIds)))

        if self.center_eval_class_ids:
            coco_gt_bbox = self._get_coco_gt_for_iou('bbox')
            valid_img_ids = set(coco_gt_bbox.imgs.keys()) if hasattr(coco_gt_bbox, 'imgs') else None
            predictions_center = predictions if valid_img_ids is None else {
                img_id: pred for img_id, pred in predictions.items() if img_id in valid_img_ids
            }
            for original_id, prediction in predictions_center.items():
                self._build_center_eval_record(original_id, prediction, coco_gt_bbox)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            img_ids, eval_imgs = merge(self.img_ids[iou_type], self.eval_imgs[iou_type])

            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval._evalImgs_cpp = eval_imgs
        if self.center_eval_class_ids:
            gathered = dist_utils.all_gather(self.center_eval_records)
            self.center_eval_records = self._merge_center_eval_records(gathered)

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()
        self.center_eval = self._accumulate_center_eval()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()
        if self.center_eval_class_ids:
            self._summarize_center_eval()

    def _summarize_center_eval(self):
        summary = self.center_eval.get('summary', {})
        if not summary:
            print('Center metric: no center-point evaluation results')
            return

        print('Center metric: wholebody40 joint classes')
        for threshold in self.center_eval.get('thresholds', []):
            suffix = f'@{threshold:g}'
            print(
                '  '
                f'F1{suffix}: {summary.get(f"f1{suffix}", float("nan")):.4f} '
                f'Precision{suffix}: {summary.get(f"precision{suffix}", float("nan")):.4f} '
                f'Recall{suffix}: {summary.get(f"recall{suffix}", float("nan")):.4f}'
            )
        print(
            '  '
            f'median_distance: {summary.get("median_distance", float("nan")):.4f} '
            f'mean_distance: {summary.get("mean_distance", float("nan")):.4f} '
            f'matched: {summary.get("matched_count", 0)} '
            f'gt: {summary.get("gt_count", 0)} '
            f'pred: {summary.get("pred_count", 0)}'
        )

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = convert_to_xywh(prediction["boxes"].detach().cpu()).tolist()
            scores = prediction["scores"].detach().cpu().tolist()
            labels = prediction["labels"].detach().cpu().tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"].detach().cpu()
            labels = prediction["labels"].detach().cpu()
            masks = prediction["masks"].detach().cpu()
            if self.segm_eval_category_ids:
                keep = torch.zeros_like(labels, dtype=torch.bool)
                for cat_id in self.segm_eval_category_ids:
                    keep |= labels == cat_id
                if not keep.any():
                    continue
                scores = scores[keep]
                labels = labels[keep]
                masks = masks[keep]

            masks = masks > 0.5

            scores = scores.tolist()
            labels = labels.tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = convert_to_xywh(prediction["boxes"].detach().cpu()).tolist()
            scores = prediction["scores"].detach().cpu().tolist()
            labels = prediction["labels"].detach().cpu().tolist()
            keypoints = prediction["keypoints"].detach().cpu().flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def merge(img_ids, eval_imgs):
    if not img_ids or not eval_imgs:
        return [], []

    all_img_ids = dist_utils.all_gather(img_ids)
    all_eval_imgs = dist_utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.extend(p)


    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, axis=2)

    # keep only unique (and in sorted order) images. DistributedSampler pads
    # eval datasets when the number of samples is not divisible by world size,
    # so eval_imgs must be filtered with the same indices as img_ids.
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[:, :, idx].ravel()

    return merged_img_ids.tolist(), merged_eval_imgs.tolist()
