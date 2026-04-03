"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision

import copy

from .dfine_utils import bbox2distance
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ..misc.mask_resize import resize_masks
from ..core import register


@register()
class DEIMCriterion(nn.Module):
    """ This class computes the loss for DEIM.
    """
    __share__ = ['num_classes', 'mask_category_ids', 'mask_target_class_ids', 'mask_resize_origin']
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        mask_category_ids=None,
        mask_target_class_ids=None,
        mask_resize_origin='center',
        boxes_weight_format=None,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
        use_boundary_aware_loss=False,
        boundary_aware_width=3,
        boundary_aware_weight=2.0,
        use_contour_detection=False,
        use_distance_transform=False,
        distance_transform_steps=5,
        ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.mask_category_ids = [] if mask_category_ids is None else list(mask_category_ids)
        if mask_target_class_ids is None:
            mask_target_class_ids = self.mask_category_ids
        self.mask_target_class_ids = [] if mask_target_class_ids is None else list(mask_target_class_ids)
        self.mask_resize_origin = mask_resize_origin
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set
        self.use_boundary_aware_loss = use_boundary_aware_loss
        self.boundary_aware_width = boundary_aware_width
        self.boundary_aware_weight = boundary_aware_weight
        self.use_contour_detection = use_contour_detection
        self.use_distance_transform = use_distance_transform
        self.distance_transform_steps = distance_transform_steps

    def get_extra_state(self):
        return {
            'num_classes': self.num_classes,
            'weight_dict': copy.deepcopy(self.weight_dict),
            'losses': copy.deepcopy(self.losses),
            'boxes_weight_format': self.boxes_weight_format,
            'share_matched_indices': self.share_matched_indices,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'reg_max': self.reg_max,
            'mask_category_ids': copy.deepcopy(self.mask_category_ids),
            'mask_target_class_ids': copy.deepcopy(self.mask_target_class_ids),
            'mask_resize_origin': self.mask_resize_origin,
            'mal_alpha': self.mal_alpha,
            'use_uni_set': self.use_uni_set,
            'use_boundary_aware_loss': self.use_boundary_aware_loss,
            'boundary_aware_width': self.boundary_aware_width,
            'boundary_aware_weight': self.boundary_aware_weight,
            'use_contour_detection': self.use_contour_detection,
            'use_distance_transform': self.use_distance_transform,
            'distance_transform_steps': self.distance_transform_steps,
        }

    def set_extra_state(self, state):
        if not state:
            return
        self.num_classes = state.get('num_classes', self.num_classes)
        self.weight_dict = copy.deepcopy(state.get('weight_dict', self.weight_dict))
        self.losses = copy.deepcopy(state.get('losses', self.losses))
        self.boxes_weight_format = state.get('boxes_weight_format', self.boxes_weight_format)
        self.share_matched_indices = state.get('share_matched_indices', self.share_matched_indices)
        self.alpha = state.get('alpha', self.alpha)
        self.gamma = state.get('gamma', self.gamma)
        self.reg_max = state.get('reg_max', self.reg_max)
        self.mask_category_ids = copy.deepcopy(state.get('mask_category_ids', self.mask_category_ids))
        self.mask_target_class_ids = copy.deepcopy(state.get('mask_target_class_ids', self.mask_target_class_ids))
        self.mask_resize_origin = state.get('mask_resize_origin', self.mask_resize_origin)
        self.mal_alpha = state.get('mal_alpha', self.mal_alpha)
        self.use_uni_set = state.get('use_uni_set', self.use_uni_set)
        self.use_boundary_aware_loss = state.get('use_boundary_aware_loss', self.use_boundary_aware_loss)
        self.boundary_aware_width = state.get('boundary_aware_width', self.boundary_aware_width)
        self.boundary_aware_weight = state.get('boundary_aware_weight', self.boundary_aware_weight)
        self.use_contour_detection = state.get('use_contour_detection', self.use_contour_detection)
        self.use_distance_transform = state.get('use_distance_transform', self.use_distance_transform)
        self.distance_transform_steps = state.get('distance_transform_steps', self.distance_transform_steps)

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha != None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        # print(" ### DEIM-gamma{}-alpha{} ### ".format(self.gamma, self.mal_alpha))
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
            and Decoupled Distillation Focal (DDF) Loss. """

        losses = {}
        if 'pred_corners' in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max+1))
            ref_points = outputs['ref_points'][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and 'is_dn' in outputs:
                        self.fgl_targets_dn= bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])
                if self.fgl_targets is None and 'is_dn' not in outputs:
                        self.fgl_targets = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])

            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets

            ious = torch.diag(box_iou(\
                        box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)

            if 'teacher_corners' in outputs:
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max+1))
                target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max+1))
                if not torch.equal(pred_corners, target_corners):
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                    (F.log_softmax(pred_corners / T, dim=1), F.softmax(target_corners.detach() / T, dim=1))).sum(-1)
                    if 'is_dn' not in outputs:
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (self.num_pos + self.num_neg)

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        if 'pred_masks' not in outputs and 'mask_embed' not in outputs:
            return {}

        zero = self._get_mask_zero(outputs)

        src_masks_list = []
        src_contours_list = []
        src_distances_list = []
        target_masks = []
        valid_masks = []
        for batch_idx, (target, (matched_pred_idx, matched_target_idx)) in enumerate(zip(targets, indices)):
            if len(matched_target_idx) == 0:
                continue
            if 'masks' not in target or 'mask_valid' not in target:
                raise KeyError('Body-only mask supervision requires `masks` and `mask_valid` in targets.')
            valid = self._select_valid_mask_matches(target, matched_target_idx)
            if valid.numel() == 0 or not valid.any():
                continue
            selected_pred_idx = matched_pred_idx[valid]
            if 'pred_masks' in outputs:
                src_masks_list.append(outputs['pred_masks'][batch_idx, selected_pred_idx])
            else:
                src_masks_list.append(
                    self._compute_sparse_mask_logits(
                        outputs['mask_embed'],
                        outputs['mask_features'],
                        batch_idx,
                        selected_pred_idx,
                    )
                )
            if self.use_contour_detection:
                if 'pred_mask_contours' in outputs:
                    src_contours_list.append(outputs['pred_mask_contours'][batch_idx, selected_pred_idx])
                elif 'pred_mask_contour_embeds' in outputs and 'pred_mask_contour_features' in outputs:
                    src_contours_list.append(
                        self._compute_sparse_mask_logits(
                            outputs['pred_mask_contour_embeds'],
                            outputs['pred_mask_contour_features'],
                            batch_idx,
                            selected_pred_idx,
                        )
                    )
            if self.use_distance_transform:
                if 'pred_mask_distances' in outputs:
                    src_distances_list.append(outputs['pred_mask_distances'][batch_idx, selected_pred_idx])
                elif 'pred_mask_distance_embeds' in outputs and 'pred_mask_distance_features' in outputs:
                    src_distances_list.append(
                        self._compute_sparse_mask_logits(
                            outputs['pred_mask_distance_embeds'],
                            outputs['pred_mask_distance_features'],
                            batch_idx,
                            selected_pred_idx,
                        )
                    )
            target_masks.append(
                target['masks'][matched_target_idx[valid]]
            )
            valid_masks.append(valid.sum())

        if not src_masks_list:
            return self._build_zero_mask_losses(zero)

        src_masks = torch.cat(src_masks_list, dim=0)
        target_masks = torch.cat(target_masks, dim=0)
        target_masks = resize_masks(
            target_masks[:, None].float(),
            size=src_masks.shape[-2:],
            mode='nearest',
            origin=self.mask_resize_origin,
        )[:, 0].to(device=src_masks.device, dtype=src_masks.dtype)

        num_masks = torch.as_tensor([sum(int(v.item()) for v in valid_masks)], dtype=torch.float, device=src_masks.device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        loss_mask_bce = F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='none')
        loss_mask_bce = loss_mask_bce.flatten(1).mean(1).sum() / num_masks
        loss_mask_dice = self.sigmoid_dice_loss(src_masks, target_masks, num_masks)
        losses = {'loss_mask_bce': loss_mask_bce, 'loss_mask_dice': loss_mask_dice}

        if self.use_boundary_aware_loss:
            boundary_weights = self._build_boundary_weight_map(
                target_masks,
                boundary_width=self.boundary_aware_width,
                boundary_weight=self.boundary_aware_weight,
            )
            loss_mask_boundary = F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='none')
            loss_mask_boundary = (loss_mask_boundary * boundary_weights).flatten(1).mean(1).sum() / num_masks
            losses['loss_mask_boundary'] = loss_mask_boundary

        if self.use_contour_detection:
            if src_contours_list:
                src_contours = torch.cat(src_contours_list, dim=0)
                contour_targets = self._generate_contour_targets(
                    target_masks,
                    target_size=src_contours.shape[-2:],
                )
                loss_mask_contour = F.binary_cross_entropy_with_logits(src_contours, contour_targets, reduction='none')
                loss_mask_contour = loss_mask_contour.flatten(1).mean(1).sum() / num_masks
            else:
                loss_mask_contour = zero
            losses['loss_mask_contour'] = loss_mask_contour

        if self.use_distance_transform:
            if src_distances_list:
                src_distances = torch.cat(src_distances_list, dim=0)
                distance_targets = self._generate_distance_targets(
                    target_masks,
                    target_size=src_distances.shape[-2:],
                    steps=self.distance_transform_steps,
                )
                loss_mask_distance = F.l1_loss(torch.sigmoid(src_distances), distance_targets, reduction='none')
                loss_mask_distance = loss_mask_distance.flatten(1).mean(1).sum() / num_masks
            else:
                loss_mask_distance = zero
            losses['loss_mask_distance'] = loss_mask_distance

        return losses

    def _get_mask_zero(self, outputs):
        for key in (
            'pred_masks',
            'mask_embed',
            'mask_features',
            'pred_mask_contours',
            'pred_mask_contour_embeds',
            'pred_mask_distances',
            'pred_mask_distance_embeds',
        ):
            value = outputs.get(key)
            if torch.is_tensor(value):
                return value.sum() * 0
        raise KeyError('Mask loss requested but no tensor outputs are available to build a zero scalar.')

    def _build_zero_mask_losses(self, zero):
        losses = {'loss_mask_bce': zero, 'loss_mask_dice': zero}
        if self.use_boundary_aware_loss:
            losses['loss_mask_boundary'] = zero
        if self.use_contour_detection:
            losses['loss_mask_contour'] = zero
        if self.use_distance_transform:
            losses['loss_mask_distance'] = zero
        return losses

    def _select_valid_mask_matches(self, target, matched_target_idx):
        valid = target['mask_valid'][matched_target_idx]
        if self.mask_target_class_ids:
            labels = target['labels'][matched_target_idx]
            class_mask = torch.stack(
                [labels == int(class_id) for class_id in self.mask_target_class_ids],
                dim=0,
            ).any(dim=0)
            valid = valid & class_mask
        return valid

    def _compute_sparse_mask_logits(self, embeds, features, batch_idx, selected_pred_idx):
        selected_embeds = embeds[batch_idx, selected_pred_idx]
        feature_map = features[batch_idx]
        return torch.einsum('qc,chw->qhw', selected_embeds, feature_map)

    def _build_boundary_weight_map(
        self,
        target_masks: torch.Tensor,
        boundary_width: int,
        boundary_weight: float,
    ) -> torch.Tensor:
        if boundary_width <= 1:
            return torch.ones_like(target_masks, dtype=torch.float32)

        mask = target_masks[:, None].float()
        pool = nn.MaxPool2d(boundary_width, stride=1, padding=boundary_width // 2)
        dilated = pool(mask)
        eroded = 1 - pool(1 - mask)
        boundary = (dilated - eroded) > 0
        weights = torch.ones_like(mask, dtype=torch.float32)
        weights[boundary] = boundary_weight
        return weights[:, 0]

    def _generate_contour_targets(
        self,
        target_masks: torch.Tensor,
        target_size=None,
        base_resolution: int = 64 * 48,
    ) -> torch.Tensor:
        masks = target_masks[:, None].float()
        if target_size is not None and tuple(masks.shape[-2:]) != tuple(target_size):
            masks = resize_masks(
                masks,
                size=target_size,
                mode='nearest',
                origin=self.mask_resize_origin,
            )

        _, _, height, width = masks.shape
        dy = torch.abs(masks[:, :, 1:, :] - masks[:, :, :-1, :])
        dx = torch.abs(masks[:, :, :, 1:] - masks[:, :, :, :-1])
        dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
        dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
        contours = torch.maximum(dy, dx)

        current_resolution = height * width
        resolution_ratio = current_resolution / float(base_resolution)
        edge_width = max(1, int((resolution_ratio ** 0.5) * 1.5))
        if edge_width > 1:
            kernel_size = 2 * edge_width - 1
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=contours.device, dtype=contours.dtype)
            kernel = kernel / kernel.numel()
            contours = F.conv2d(contours, kernel, padding=kernel_size // 2)
            contours = (contours > 0.1).to(dtype=target_masks.dtype)

        return contours[:, 0].to(dtype=target_masks.dtype)

    def _generate_distance_targets(
        self,
        target_masks: torch.Tensor,
        target_size=None,
        steps: int = 5,
    ) -> torch.Tensor:
        masks = target_masks[:, None].float()
        if target_size is not None and tuple(masks.shape[-2:]) != tuple(target_size):
            masks = resize_masks(
                masks,
                size=target_size,
                mode='nearest',
                origin=self.mask_resize_origin,
            )

        distances = masks.clone()
        for _ in range(max(0, int(steps))):
            dilated = F.max_pool2d(distances, kernel_size=3, stride=1, padding=1)
            distances = distances + (1 - distances) * dilated * 0.5

        return distances[:, 0].to(dtype=target_masks.dtype)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers. """
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                        for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'local': self.loss_local,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, epoch=0, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, epoch=epoch)['indices']
        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if 'aux_outputs' in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs['aux_outputs']
            if 'pre_outputs' in outputs:
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
            for i, aux_outputs in enumerate(aux_outputs_list):
                indices_aux = self.matcher(aux_outputs, targets, epoch=epoch)['indices']
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                indices_enc = self.matcher(aux_outputs, targets, epoch=epoch)['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            assert 'aux_outputs' in outputs, ''

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses, main loss
        losses = {}
        for loss in self.losses:
            use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer. just for dfine
        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']
            for loss in self.losses:
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self.losses:
                    use_uni_set = self.use_uni_set and (loss == 'boxes')
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.
        if 'dn_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer, just for dfine
            if 'dn_pre_outputs' in outputs:
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # For debugging Objects365 pre-train.
        losses = {k:torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', 'mal'):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))

        return dn_match_indices


    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)


    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum', avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
             + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    @staticmethod
    def sigmoid_dice_loss(inputs, targets, num_boxes, eps=1e-6):
        inputs = inputs.sigmoid().flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(1) + targets.sum(1)
        loss = 1 - (numerator + eps) / (denominator + eps)
        return loss.sum() / num_boxes

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1
        step = .5 / (num_layers - 1)
        opt_list = [.5  + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list
