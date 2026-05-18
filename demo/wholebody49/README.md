# DEIMv2 WholeBody49 Demo

PyTorch checkpoint demo for wholebody49 instance segmentation.

wholebody49 extends wholebody48 with `classid=48` (`bone`). Bone boxes are used as evidence for drawing skeleton lines. They are not rendered as normal boxes unless `--enable_bone_bbox_drawing_mode` is specified.

## Image Folder

```bash
uv run python demo/wholebody49/demo_deimv2_torch_wholebody49_ins.py \
-c configs/deimv2/deimv2_dinov3_x_wholebody49_ins_s08_maskhead256x3_center.yml \
-r ckpts/last_full_epoch.pth \
-i images_partial \
-o outputs/demo_wholebody49 \
--score_threshold 0.35 \
--mask_threshold 0.5 \
--enable_bone_drawing_mode \
--enable-masks \
--disable_tracking
```

## Video or Camera

```bash
uv run python demo/wholebody49/demo_deimv2_torch_wholebody49_ins.py \
-c configs/deimv2/deimv2_dinov3_x_wholebody49_ins_s08_maskhead256x3_center.yml \
-r ckpts/last_full_epoch.pth \
-v 0 \
-o outputs/demo_wholebody49_video \
-d cuda \
--enable_bone_drawing_mode \
--enable-masks
```

## Notes

- The checkpoint loader uses `ema.module` first when present, otherwise `model`.
- Body masks are rendered for `classid=0` when `--enable-masks` is set.
- `--enable_bone_drawing_mode` connects wholebody keypoints only when a predicted `bone` box contains a supported keypoint pair.
- `--enable_bone_bbox_drawing_mode` additionally draws the raw `classid=48` bone boxes for debugging.
- `--save_raw_predictions` writes `labels`, `scores`, `boxes`, and body mask metadata; `classid=48` remains in the saved predictions.
