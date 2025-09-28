"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register


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
        'remap_mscoco_category'
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

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

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor=None):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        bbox_pred = self.box_cxcywh_to_xyxy(boxes)
        if orig_target_sizes is not None:
            bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            n, a, c = scores.shape
            scores, index = torch.topk(scores.reshape([n,a*c]), self.num_top_queries, dim=-1)
            if orig_target_sizes is None:
                scores = scores.unsqueeze(-1)
            index = index.unsqueeze(-1)
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.repeat(1, 1, bbox_pred.shape[-1]))

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        if self.deploy_mode:
            if orig_target_sizes is not None:
                return labels, boxes, scores
            else:
                return torch.cat([labels, boxes, scores], dim=2)

        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
