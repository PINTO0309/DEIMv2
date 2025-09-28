# Export

```bash
WEIGHT=deimv2_dinov3_x_coco
cd ../..
uv run python tools/deployment/export_onnx.py \
-c configs/deimv2/deimv2_dinov3_x_coco.yml \
-r ckpts/${WEIGHT}.pth \
--opset 17

onnxslim ckpts/${WEIGHT}.onnx ckpts/${WEIGHT}.onnx
onnxsim ckpts/${WEIGHT}.onnx ckpts/${WEIGHT}.onnx
```