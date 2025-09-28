# Export

```bash
cd ../..

WEIGHT=deimv2_dinov3_x_coco
QUERIES=300

uv run python tools/deployment/export_onnx.py \
-c configs/deimv2/${WEIGHT}.yml \
-r ckpts/${WEIGHT}.pth \
--opset 17

onnxslim ckpts/${WEIGHT}_${QUERIES}query.onnx ckpts/${WEIGHT}_${QUERIES}query.onnx
onnxsim ckpts/${WEIGHT}_${QUERIES}query.onnx ckpts/${WEIGHT}_${QUERIES}query.onnx

uv run python tools/deployment/export_onnx.py \
-c configs/deimv2/${WEIGHT}.yml \
-r ckpts/${WEIGHT}.pth \
--opset 17 \
--dynamic_batch

onnxslim ckpts/${WEIGHT}_${QUERIES}query_n_batch.onnx ckpts/${WEIGHT}_${QUERIES}query_n_batch.onnx
onnxsim ckpts/${WEIGHT}_${QUERIES}query_n_batch.onnx ckpts/${WEIGHT}_${QUERIES}query_n_batch.onnx
```

```bash
uv run python tools/deployment/make_prep.py
```

<img width="808" height="704" alt="image" src="https://github.com/user-attachments/assets/82606a50-c294-43f2-b617-a653a6ba5424" />
