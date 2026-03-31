"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register
from ..misc.mask_resize import resize_masks


__all__ = ['PostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class PostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category',
        'mask_resize_origin',
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
        mask_resize_origin='center',
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False
        self.mask_resize_origin = mask_resize_origin

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    def box_cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Converts bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
        (cx, cy) refers to center of bounding box
        (w, h) are width and height of bounding box
        Args:
            boxes (Tensor[N, 4]): boxes in (cx, cy, w, h) format which will be converted.

        Returns:
            boxes (Tensor(N, 4)): boxes in (x1, y1, x2, y2) format.
        """
        cx = boxes[..., 0:1]
        cy = boxes[..., 1:2]
        w = boxes[..., 2:3]
        h = boxes[..., 3:4]
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        boxes = torch.cat([x1, y1, x2, y2], dim=2)
        return boxes

    def resize_masks(self, mask_logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return resize_masks(
            mask_logits,
            size=size,
            mode='bilinear',
            origin=self.mask_resize_origin,
        )

    def _gather_mask_like(
        self,
        predictions: torch.Tensor | None,
        query_index: torch.Tensor,
        orig_target_sizes: torch.Tensor | None,
    ) -> torch.Tensor | list[torch.Tensor] | None:
        if predictions is None:
            return None

        gathered = []
        for batch_idx in range(predictions.shape[0]):
            logits = predictions[batch_idx, query_index[batch_idx]].unsqueeze(1)
            if orig_target_sizes is not None:
                orig_w, orig_h = orig_target_sizes[batch_idx].tolist()
                logits = self.resize_masks(logits, size=(int(orig_h), int(orig_w)))
            gathered.append(logits.sigmoid())

        if orig_target_sizes is not None:
            return gathered
        return torch.stack(gathered, dim=0)

    # def forward(self, outputs, orig_target_sizes):
    def forward(
        self,
        outputs,
        orig_target_sizes: torch.Tensor=None,
        return_masks: bool=True,
        return_contours: bool=False,
    ):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        pred_masks = outputs.get('pred_masks') if return_masks else None
        pred_contours = outputs.get('pred_mask_contours') if return_contours else None
        bbox_pred = self.box_cxcywh_to_xyxy(boxes)
        if orig_target_sizes is not None:
            bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, flat_index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            if orig_target_sizes is None:
                scores = scores.unsqueeze(-1)
            labels = mod(flat_index, self.num_classes)
            query_index = flat_index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=query_index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, query_index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=query_index)
                boxes = torch.gather(bbox_pred, dim=1, index=query_index.unsqueeze(-1).tile(1, 1, bbox_pred.shape[-1]))
            else:
                query_index = torch.arange(scores.shape[1], device=logits.device).unsqueeze(0).expand(scores.shape[0], -1)
                boxes = bbox_pred

        gathered_masks = self._gather_mask_like(pred_masks, query_index, orig_target_sizes)
        gathered_contours = self._gather_mask_like(pred_contours, query_index, orig_target_sizes)

        if self.deploy_mode:
            if orig_target_sizes is not None and (gathered_masks is not None or gathered_contours is not None):
                raise RuntimeError('Deploy mode does not support resized mask/contour outputs. Export without orig_target_sizes instead.')

            deploy_labels = labels
            deploy_scores = scores
            if deploy_labels.dim() == 2:
                deploy_labels = deploy_labels.unsqueeze(-1)
            if deploy_scores.dim() == 2:
                deploy_scores = deploy_scores.unsqueeze(-1)
            label_xyxy_score = torch.cat([deploy_labels, boxes, deploy_scores], dim=2)

            if gathered_masks is not None or gathered_contours is not None:
                deploy_outputs = [label_xyxy_score]
                if gathered_masks is not None:
                    deploy_outputs.append(gathered_masks)
                if gathered_contours is not None:
                    deploy_outputs.append(gathered_contours)
                return tuple(deploy_outputs)

            if orig_target_sizes is not None:
                return labels, boxes, scores
            return label_xyxy_score

        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        if labels.dim() > 2 and labels.size(-1) == 1:
            labels = labels.squeeze(-1)
        if scores.dim() > 2 and scores.size(-1) == 1:
            scores = scores.squeeze(-1)

        results = []
        for batch_idx, (lab, box, sco) in enumerate(zip(labels, boxes, scores)):
            result = dict(labels=lab, boxes=box, scores=sco)
            if gathered_masks is not None:
                result['masks'] = gathered_masks[batch_idx]
            if gathered_contours is not None:
                result['contours'] = gathered_contours[batch_idx]
            results.append(result)

        return results


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
