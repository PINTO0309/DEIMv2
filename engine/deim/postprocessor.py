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
        'mask_target_class_ids',
        'remap_mscoco_category',
        'mask_resize_origin',
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        mask_target_class_ids=None,
        remap_mscoco_category=False,
        mask_resize_origin='center',
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.mask_target_class_ids = [] if mask_target_class_ids is None else [int(x) for x in mask_target_class_ids]
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

        if orig_target_sizes is None:
            gather_index = query_index[:, :, None, None].expand(
                -1,
                -1,
                predictions.shape[-2],
                predictions.shape[-1],
            )
            return predictions.gather(dim=1, index=gather_index).sigmoid()

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

    def _build_mask_target_mask(self, labels: torch.Tensor) -> torch.Tensor:
        if not self.mask_target_class_ids:
            return torch.ones_like(labels, dtype=torch.bool)
        return torch.stack(
            [labels == class_id for class_id in self.mask_target_class_ids],
            dim=0,
        ).any(dim=0)

    def _gather_mask_embeddings(
        self,
        embeddings: torch.Tensor,
        query_index: torch.Tensor,
    ) -> torch.Tensor:
        gather_index = query_index.unsqueeze(-1).expand(-1, -1, embeddings.shape[-1])
        return embeddings.gather(dim=1, index=gather_index)

    def _compute_sparse_mask_probs_deploy(
        self,
        embeddings: torch.Tensor | None,
        feature_maps: torch.Tensor | None,
        query_index: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if embeddings is None or feature_maps is None:
            return None

        topk_embeddings = self._gather_mask_embeddings(embeddings, query_index)
        selected_positions = torch.nonzero(target_mask, as_tuple=False)
        selected_embeddings = topk_embeddings[selected_positions[:, 0], selected_positions[:, 1]]
        selected_feature_maps = feature_maps[selected_positions[:, 0]]
        selected_probs = torch.sigmoid(torch.einsum('mc,mchw->mhw', selected_embeddings, selected_feature_maps))

        batch_size, num_queries = query_index.shape
        height, width = feature_maps.shape[-2:]
        flat_output = feature_maps.new_zeros((batch_size * num_queries, height, width))
        flat_indices = selected_positions[:, 0] * num_queries + selected_positions[:, 1]
        scatter_index = flat_indices[:, None, None].expand(-1, height, width)
        flat_output = flat_output.scatter(0, scatter_index, selected_probs)
        return flat_output.reshape(batch_size, num_queries, height, width)

    def _compute_sparse_mask_probs(
        self,
        embeddings: torch.Tensor | None,
        feature_maps: torch.Tensor | None,
        query_index: torch.Tensor,
        orig_target_sizes: torch.Tensor | None,
        target_mask: torch.Tensor,
    ) -> torch.Tensor | list[torch.Tensor] | None:
        if embeddings is None or feature_maps is None:
            return None

        if orig_target_sizes is None:
            return self._compute_sparse_mask_probs_deploy(embeddings, feature_maps, query_index, target_mask)

        topk_embeddings = self._gather_mask_embeddings(embeddings, query_index)
        gathered = []
        for batch_idx in range(topk_embeddings.shape[0]):
            orig_w, orig_h = orig_target_sizes[batch_idx].tolist()
            batch_output = feature_maps.new_zeros(
                (topk_embeddings.shape[1], 1, int(orig_h), int(orig_w))
            )
            selected_queries = torch.nonzero(target_mask[batch_idx], as_tuple=False).flatten()
            if selected_queries.numel() > 0:
                selected_embeddings = topk_embeddings[batch_idx, selected_queries]
                selected_logits = torch.einsum('qc,chw->qhw', selected_embeddings, feature_maps[batch_idx])
                resized_masks = self.resize_masks(
                    selected_logits.unsqueeze(1),
                    size=(int(orig_h), int(orig_w)),
                ).sigmoid()
                batch_output[selected_queries] = resized_masks
            gathered.append(batch_output)
        return gathered

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
        mask_embed = outputs.get('mask_embed') if return_masks else None
        mask_features = outputs.get('mask_features') if return_masks else None
        contour_embeds = outputs.get('pred_mask_contour_embeds') if return_contours else None
        contour_features = outputs.get('pred_mask_contour_features') if return_contours else None
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

        target_mask = self._build_mask_target_mask(labels)
        gathered_masks = self._gather_mask_like(pred_masks, query_index, orig_target_sizes)
        if gathered_masks is None and return_masks:
            gathered_masks = self._compute_sparse_mask_probs(
                mask_embed,
                mask_features,
                query_index,
                orig_target_sizes,
                target_mask,
            )

        gathered_contours = self._gather_mask_like(pred_contours, query_index, orig_target_sizes)
        if gathered_contours is None and return_contours:
            gathered_contours = self._compute_sparse_mask_probs(
                contour_embeds,
                contour_features,
                query_index,
                orig_target_sizes,
                target_mask,
            )

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
