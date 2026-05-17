# DEIMv2 WholeBody68 Demo

This demo runs DEIMv2 WholeBody68 object detection and body-only instance segmentation from a PyTorch checkpoint or an exported ONNX model. The 68-class label set adds hand keypoints at `classid=48-67`.

## Classes

- `classid=0`: body. This is the only class with rendered instance masks.
- `classid=21-44`: body keypoints and left/right body-side attributes.
- `classid=48-67`: hand keypoints.
- `classid=29`: wrist keypoint used as the hand skeleton root.
- `classid=32`: hand box used as the handedness source for hand keypoints.

The class names are listed in `demo/wholebody68/classes.txt`.

## Image Folder

```bash
uv run python demo/wholebody68/demo_deimv2_torch_wholebody68_ins.py \
  -c configs/deimv2/deimv2_dinov3_x_wholebody68_ins_s08_maskhead256x3_center.yml \
  -r ckpts/deimv2_dinov3_x_wholebody68_ins_center.pth \
  -i images_partial \
  -o outputs/demo_wholebody68_images \
  -d cuda \
  --score_threshold 0.35 \
  --keypoint_threshold 0.25 \
  --mask_threshold 0.5 \
  --keypoint_drawing_mode dot \
  --enable_bone_drawing_mode \
  --enable-masks \
  --disable_waitKey
```

The script processes `jpg/jpeg/png/bmp/webp` images and preserves the original filenames in `-o/--output_dir`.

## Video Or Camera

```bash
uv run python demo/wholebody68/demo_deimv2_torch_wholebody68_ins.py \
  -c configs/deimv2/deimv2_dinov3_x_wholebody68_ins_s08_maskhead256x3_center.yml \
  -r ckpts/deimv2_dinov3_x_wholebody68_ins_center.pth \
  -v 0 \
  -o outputs/demo_wholebody68_video \
  -d cuda \
  --score_threshold 0.35 \
  --keypoint_threshold 0.25 \
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
uv run python demo/wholebody68/demo_deimv2_torch_wholebody68_ins.py \
  -c configs/deimv2/deimv2_dinov3_x_wholebody68_ins_s08_maskhead256x3_center.yml \
  -r deimv2_dinov3_x_wholebody68_ins_s08_1360query_masks.onnx \
  -i images_partial \
  -o outputs/demo_wholebody68_onnx \
  -d cuda \
  --keypoint_threshold 0.25 \
  --enable_bone_drawing_mode \
  --enable-masks
```

Use `-d tensorrt` only when the ONNX Runtime build includes `TensorrtExecutionProvider`. The supported `--inference_type` values are `fp16` and `int8`.

## Rendering Controls

- `--keypoint_threshold` applies to body keypoints and hand keypoints, including `classid=48-67`.
- `--keypoint_drawing_mode dot|box|both` controls how keypoint boxes are rendered.
- `--enable_bone_drawing_mode` draws the body skeleton and hand skeleton chains from `29:wrist` to the finger roots `48/52/56/60/64`.
- Hand keypoints inherit handedness from the nearest/containing `classid=32` hand box before skeleton matching; the actual bone connection uses `classid=29` wrist and requires compatible handedness.
- `--disable_render_classids 0` hides both body boxes and body masks.
- `--disable_render_classids 48 49 50 51` hides the thumb keypoints, and the same pattern can be used for any hand keypoint class.
- `--disable_left_and_right_label` hides only the rendered `L`/`R` text; left/right colors and association remain enabled.
- `--save_raw_predictions` writes `labels`, `scores`, `boxes`, and body mask metadata to `predictions/*.json`.

## Notes

- Body mask resize uses `--mask_resize_origin topleft` by default. Use `--mask_resize_origin center` if the checkpoint/config expects center-origin mask resizing.
- `--mask_bilateral_d`, `--mask_bilateral_sigma_color`, and `--mask_bilateral_sigma_space` can smooth body mask probabilities before thresholding.
- Hand keypoints are detection boxes, not instance masks. Only `classid=0` contributes rendered mask overlays.
