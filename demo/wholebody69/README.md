# DEIMv2 WholeBody69 Demo

This demo runs DEIMv2 WholeBody69 object detection and body-only instance segmentation from a PyTorch checkpoint or an exported ONNX model. The 69-class label set keeps `classid=48` as `bone` and adds hand keypoints at `classid=49-68`.

## Classes

- `classid=0`: body. This is the only class with rendered instance masks.
- `classid=21-44`: body keypoints and left/right body-side attributes.
- `classid=48`: bone box used as evidence for body skeleton drawing.
- `classid=49-68`: hand keypoints.
- `classid=29`: wrist keypoint used as the hand skeleton root.
- `classid=32`: hand box used as the handedness source for hand keypoints.

The class names are listed in `demo/wholebody69/classes.txt`.

## Image Folder

```bash
uv run python demo/wholebody69/demo_deimv2_torch_wholebody69_ins.py \
  -c configs/deimv2/deimv2_dinov3_x_wholebody69_ins_s08_maskhead256x3_center.yml \
  -r ckpts/last_full_epoch.pth \
  -i images_partial \
  -o outputs/demo_wholebody69_images \
  -d cuda \
  --score_threshold 0.35 \
  --keypoint_threshold 0.35 \
  --hand_keypoint_threshold 0.25 \
  --mask_threshold 0.5 \
  --keypoint_drawing_mode dot \
  --enable_bone_drawing_mode \
  --enable-masks \
  --disable_waitKey
```

The script processes `jpg/jpeg/png/bmp/webp` images and preserves the original filenames in `-o/--output_dir`.

## Video Or Camera

```bash
uv run python demo/wholebody69/demo_deimv2_torch_wholebody69_ins.py \
  -c configs/deimv2/deimv2_dinov3_x_wholebody69_ins_s08_maskhead256x3_center.yml \
  -r ckpts/last_full_epoch.pth \
  -v 0 \
  -o outputs/demo_wholebody69_video \
  -d cuda \
  --score_threshold 0.35 \
  --keypoint_threshold 0.35 \
  --hand_keypoint_threshold 0.25 \
  --mask_threshold 0.5 \
  --keypoint_drawing_mode both \
  --enable_bone_drawing_mode \
  --enable-masks
```

During video display, keyboard toggles are available:

- `B`: toggle skeleton drawing.
- `K`: cycle keypoint drawing mode: dot, box, both.
- `N`: toggle adult/child attribute use.
- `G`: toggle gender attribute use.
- `P`: toggle headpose attribute use.
- `H`: toggle left/right body and hand identification.
- `R`: toggle body tracking.
- `T`: toggle track ID overlay.
- `M`: toggle head distance measurement.
- `Esc`: exit.

## ONNX

If `-r/--resume` points to an `.onnx` file, the same script switches to ONNX Runtime automatically.

```bash
uv run python demo/wholebody69/demo_deimv2_torch_wholebody69_ins.py \
  -c configs/deimv2/deimv2_dinov3_x_wholebody69_ins_s08_maskhead256x3_center.yml \
  -r deimv2_dinov3_x_wholebody69_ins_s08_2040query_masks.onnx \
  -i images_partial \
  -o outputs/demo_wholebody69_onnx \
  -d cuda \
  --keypoint_threshold 0.35 \
  --hand_keypoint_threshold 0.25 \
  --enable_bone_drawing_mode \
  --enable-masks
```

Use `-d tensorrt` only when the ONNX Runtime build includes `TensorrtExecutionProvider`. The supported `--inference_type` values are `fp16` and `int8`.

## Rendering Controls

- `--keypoint_threshold` applies to body keypoints.
- `--hand_keypoint_threshold` applies to hand keypoints `classid=49-68`; when omitted, it falls back to `--keypoint_threshold`, then `--score_threshold`.
- `--enable_bone_drawing_mode` uses `classid=48` bone boxes for body skeleton support and draws hand skeleton chains from `29:wrist` to the finger roots `49/53/57/61/65`.
- Hand keypoint edges involving `classid=49-68` do not use `classid=48` bone boxes; both endpoints must be inside the same `classid=32` hand box.
- Hand keypoints `49-68` are always rendered as circles centered on `cx,cy`, regardless of keypoint drawing mode.
- `--enable_bone_bbox_drawing_mode` additionally draws the raw `classid=48` bone boxes for debugging.
- Hand keypoints inherit handedness from the nearest/containing `classid=32` hand box before skeleton matching; the actual bone connection uses `classid=29` wrist and requires compatible handedness.
- `--disable_render_classids 0` hides both body boxes and body masks.
- `--disable_render_classids 49 50 51 52` hides the thumb keypoints, and the same pattern can be used for any hand keypoint class.
- `--disable_left_and_right_label` hides only the rendered `L`/`R` text; left/right colors and association remain enabled.
- `--save_raw_predictions` writes `labels`, `scores`, `boxes`, and body mask metadata to `predictions/*.json`.

## Notes

- Body mask resize uses `--mask_resize_origin topleft` by default. Use `--mask_resize_origin center` when you want center-origin mask resizing.
- ONNX mask and contour resizing converts probability maps back to logits, resizes logits, then applies sigmoid so ONNX inference matches PyTorch checkpoint mask postprocessing more closely.
- `--mask_resize_mode` controls ONNX mask resize interpolation and defaults to `bilinear`.
- `--mask_bilateral_d`, `--mask_bilateral_sigma_color`, and `--mask_bilateral_sigma_space` can smooth body mask probabilities before thresholding.
- Hand keypoints and bone boxes are detection boxes, not instance masks. Only `classid=0` contributes rendered mask overlays.
