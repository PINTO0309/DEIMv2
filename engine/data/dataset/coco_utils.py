"""
copy and modified https://github.com/pytorch/vision/blob/main/references/detection/coco_utils.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch
import torch.utils.data
import torchvision
import torchvision.transforms.functional as TVF
import faster_coco_eval.core.mask as coco_mask
from faster_coco_eval import COCO
import copy


def decode_coco_segmentation(segmentation, height, width):
    empty_mask = torch.zeros((height, width), dtype=torch.uint8)

    if not segmentation:
        return empty_mask, False

    try:
        if isinstance(segmentation, list):
            rles = coco_mask.frPyObjects(segmentation, height, width)
            mask = coco_mask.decode(rles)
        elif isinstance(segmentation, dict):
            counts = segmentation.get("counts")
            size = segmentation.get("size")
            if not counts or not size:
                return empty_mask, False
            if isinstance(counts, list):
                rles = coco_mask.frPyObjects(segmentation, height, width)
                mask = coco_mask.decode(rles)
            elif isinstance(counts, (str, bytes)):
                mask = coco_mask.decode(segmentation)
            else:
                return empty_mask, False
        else:
            return empty_mask, False
    except Exception:
        return empty_mask, False

    mask = torch.as_tensor(mask, dtype=torch.uint8)
    if mask.ndim == 2:
        if tuple(mask.shape) != (height, width):
            return empty_mask, False
        return mask, True

    if mask.ndim == 3 and tuple(mask.shape[:2]) == (height, width):
        return mask.any(dim=2).to(torch.uint8), True

    return empty_mask, False


def convert_coco_poly_to_mask(segmentations, height, width, return_valid=False):
    masks = []
    valid_flags = []
    for segmentation in segmentations:
        mask, valid = decode_coco_segmentation(segmentation, height, width)
        masks.append(mask)
        valid_flags.append(valid)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)

    if return_valid:
        return masks, torch.as_tensor(valid_flags, dtype=torch.bool)
    return masks


class ConvertCocoPolysToMask:
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]

        anno = target["annotations"]

        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different criteria for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        # img, targets = ds[img_idx]

        img, targets = ds.load_item(img_idx)
        width, height = img.size

        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["width"] = width
        img_dict["height"] = height
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2] # xyxy -> xywh
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)


def get_coco_api_from_dataset_for_segm(dataset, category_ids=None, ignore_missing_masks=False):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset

    if not isinstance(dataset, torchvision.datasets.CocoDetection):
        return convert_to_coco_api(dataset)

    coco_gt = copy.deepcopy(dataset.coco)
    annotations = coco_gt.dataset.get('annotations', [])
    images = coco_gt.dataset.get('images', [])
    categories = coco_gt.dataset.get('categories', [])

    if category_ids is not None:
        category_ids = set(category_ids)
        annotations = [ann for ann in annotations if ann.get('category_id') in category_ids]
        categories = [cat for cat in categories if cat.get('id') in category_ids]

    if ignore_missing_masks:
        annotations = [ann for ann in annotations if ann.get('segmentation')]

    image_ids = {ann['image_id'] for ann in annotations}
    images = [img for img in images if img.get('id') in image_ids]

    coco_gt.dataset = {
        'images': images,
        'categories': categories,
        'annotations': annotations,
    }
    coco_gt.createIndex()
    return coco_gt


def get_coco_api_from_annotation_file(ann_file, category_ids=None, ignore_missing_masks=False):
    coco_gt = COCO(ann_file)
    coco_gt = copy.deepcopy(coco_gt)

    annotations = coco_gt.dataset.get('annotations', [])
    images = coco_gt.dataset.get('images', [])
    categories = coco_gt.dataset.get('categories', [])

    if category_ids is not None:
        category_ids = set(category_ids)
        annotations = [ann for ann in annotations if ann.get('category_id') in category_ids]
        categories = [cat for cat in categories if cat.get('id') in category_ids]

    if ignore_missing_masks:
        annotations = [ann for ann in annotations if ann.get('segmentation')]

    image_ids = {ann['image_id'] for ann in annotations}
    images = [img for img in images if img.get('id') in image_ids]

    coco_gt.dataset = {
        'images': images,
        'categories': categories,
        'annotations': annotations,
    }
    coco_gt.createIndex()
    return coco_gt
