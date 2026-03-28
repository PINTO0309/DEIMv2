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
import numpy as np
import torch

from faster_coco_eval import COCO, COCOeval_faster
import faster_coco_eval.core.mask as mask_util
from ...core import register
from ...misc import dist_utils
__all__ = ['CocoEvaluator',]


@register()
class CocoEvaluator(object):
    __share__ = ['segm_eval_category_ids', 'segm_ignore_missing_masks']

    def __init__(self, coco_gt, iou_types, segm_eval_category_ids=None, segm_ignore_missing_masks=True):
        assert isinstance(iou_types, (list, tuple))
        if isinstance(coco_gt, dict):
            self.coco_gt = {k: copy.deepcopy(v) for k, v in coco_gt.items()}
        else:
            self.coco_gt = copy.deepcopy(coco_gt)
        self.iou_types = iou_types
        self.segm_eval_category_ids = [] if segm_eval_category_ids is None else list(segm_eval_category_ids)
        self.segm_ignore_missing_masks = segm_ignore_missing_masks

        self.coco_eval = {}
        for iou_type in iou_types:
            coco_gt_iou = self._get_coco_gt_for_iou(iou_type)
            self.coco_eval[iou_type] = COCOeval_faster(coco_gt_iou, iouType=iou_type, print_function=print, separate_eval=True)
            if iou_type == 'segm' and self.segm_eval_category_ids:
                self.coco_eval[iou_type].params.catIds = list(self.segm_eval_category_ids)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            coco_gt_iou = self._get_coco_gt_for_iou(iou_type)
            self.coco_eval[iou_type] = COCOeval_faster(coco_gt_iou, iouType=iou_type, print_function=print, separate_eval=True)
            if iou_type == 'segm' and self.segm_eval_category_ids:
                self.coco_eval[iou_type].params.catIds = list(self.segm_eval_category_ids)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}

    def _get_coco_gt_for_iou(self, iou_type):
        if isinstance(self.coco_gt, dict):
            if iou_type in self.coco_gt:
                return self.coco_gt[iou_type]
            if 'bbox' in self.coco_gt:
                return self.coco_gt['bbox']
        return self.coco_gt


    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]
            coco_gt_iou = self._get_coco_gt_for_iou(iou_type)

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = coco_gt_iou.loadRes(results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)
                    coco_eval.evaluate()

            self.eval_imgs[iou_type].append(np.array(coco_eval._evalImgs_cpp).reshape(len(coco_eval.params.catIds), len(coco_eval.params.areaRng), len(coco_eval.params.imgIds)))

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            img_ids, eval_imgs = merge(self.img_ids, self.eval_imgs[iou_type])

            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval._evalImgs_cpp = eval_imgs

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

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
    all_img_ids = dist_utils.all_gather(img_ids)
    all_eval_imgs = dist_utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.extend(p)


    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, axis=2).ravel()
    # merged_eval_imgs = np.array(merged_eval_imgs).T.ravel()

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)

    return merged_img_ids.tolist(), merged_eval_imgs.tolist()
