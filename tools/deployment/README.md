# Export

```bash
cd ../..

WEIGHT=deimv2_dinov3_x_coco
QUERIES=300

uv run python tools/deployment/export_onnx.py \
-c configs/deimv2/${WEIGHT}.yml \
-r ckpts/${WEIGHT}.pth \
--opset 17 \
--simplify

uv run onnxslim ${WEIGHT}_${QUERIES}query.onnx ${WEIGHT}_${QUERIES}query.onnx
uv run onnxsim ${WEIGHT}_${QUERIES}query.onnx ${WEIGHT}_${QUERIES}query.onnx

uv run python tools/deployment/export_onnx.py \
-c configs/deimv2/${WEIGHT}.yml \
-r ckpts/${WEIGHT}.pth \
--opset 17 \
--dynamic_batch \
--simplify

uv run onnxslim ${WEIGHT}_${QUERIES}query_n_batch.onnx ${WEIGHT}_${QUERIES}query_n_batch.onnx
uv run onnxsim ${WEIGHT}_${QUERIES}query_n_batch.onnx ${WEIGHT}_${QUERIES}query_n_batch.onnx
```

```bash
uv run python tools/deployment/make_prep.py
```

<img width="808" height="704" alt="image" src="https://github.com/user-attachments/assets/82606a50-c294-43f2-b617-a653a6ba5424" />

```bash
uv run sit4onnx \
-if deimv2_dinov3_x_coco_300query_n_batch.onnx \
-fs 1 3 640 640 \
-oep cpu

INFO: file: deimv2_dinov3_x_coco_300query_n_batch.onnx
INFO: providers: ['CPUExecutionProvider']
INFO: input_name.1: input_bgr shape: [1, 3, 640, 640] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  4484.5130443573 ms
INFO: avg elapsed time per pred:  448.45130443573 ms
INFO: output_name.1: label_xyxy_score shape: [1, 300, 6] dtype: float32

uv run sit4onnx \
-if deimv2_dinov3_x_coco_300query_n_batch.onnx \
-fs 5 3 640 640 \
-oep cuda

INFO: file: deimv2_dinov3_x_coco_300query_n_batch.onnx
INFO: providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: input_bgr shape: [5, 3, 640, 640] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  1635.331392288208 ms
INFO: avg elapsed time per pred:  163.5331392288208 ms
INFO: output_name.1: label_xyxy_score shape: [5, 300, 6] dtype: float32
```
