# Export

```bash
cd ../..

### X
WEIGHT=deimv2_dinov3_x_wholebody34ft
QUERIES=340

uv run python tools/deployment/export_onnx.py \
-c configs/deimv2/${WEIGHT}.yml \
-r outputs/${WEIGHT}_340/last.pth \
--opset 17

uv run python tools/deployment/export_onnx.py \
-c configs/deimv2/${WEIGHT}.yml \
-r outputs/${WEIGHT}_340/last.pth \
--opset 17 \
--dynamic_batch \
--simplify

uv run onnxslim ${WEIGHT}_${QUERIES}query_n_batch.onnx ${WEIGHT}_${QUERIES}query_n_batch.onnx
uv run onnxsim ${WEIGHT}_${QUERIES}query_n_batch.onnx ${WEIGHT}_${QUERIES}query_n_batch.onnx

### S
WEIGHT=deimv2_dinov3_s_wholebody34ft
H=640
W=640
QUERIES=1750

uv run python tools/deployment/export_onnx.py \
-c configs/deimv2/${WEIGHT}.yml \
-r ckpts/${WEIGHT}.pth \
--opset 17

uv run python tools/deployment/export_onnx.py \
-c configs/deimv2/${WEIGHT}.yml \
-r ckpts/${WEIGHT}.pth \
--opset 17 \
--dynamic_batch \
--simplify
uv run onnxslim ${WEIGHT}_${QUERIES}query_n_batch.onnx ${WEIGHT}_${QUERIES}query_n_batch.onnx
uv run onnxsim ${WEIGHT}_${QUERIES}query_n_batch.onnx ${WEIGHT}_${QUERIES}query_n_batch.onnx

rm ${WEIGHT}_${QUERIES}query.onnx

uv run onnxsim ${WEIGHT}_${QUERIES}query_n_batch.onnx ${WEIGHT}_${QUERIES}query.onnx \
--overwrite-input-shape "images:1,3,${H},${W}"

### N
WEIGHT=deimv2_hgnetv2_n_wholebody34
H=640
W=640
QUERIES=680

uv run python tools/deployment/export_onnx.py \
-c configs/deimv2/${WEIGHT}.yml \
-r ckpts/${WEIGHT}.pth \
--opset 17

uv run python tools/deployment/export_onnx.py \
-c configs/deimv2/${WEIGHT}.yml \
-r ckpts/${WEIGHT}.pth \
--opset 17 \
--dynamic_batch \
--simplify
uv run onnxslim ${WEIGHT}_${QUERIES}query_n_batch.onnx ${WEIGHT}_${QUERIES}query_n_batch.onnx
uv run onnxsim ${WEIGHT}_${QUERIES}query_n_batch.onnx ${WEIGHT}_${QUERIES}query_n_batch.onnx

rm ${WEIGHT}_${QUERIES}query.onnx

uv run onnxsim ${WEIGHT}_${QUERIES}query_n_batch.onnx ${WEIGHT}_${QUERIES}query.onnx \
--overwrite-input-shape "images:1,3,${H},${W}"
```

```bash
uv run python tools/deployment/make_prep.py -m ${WEIGHT}_${QUERIES}query.onnx
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

uv run sit4onnx \
-if deimv2_dinov3_x_coco_300query_n_batch.onnx \
-fs 1 3 640 640 \
-oep tensorrt

INFO: file: deimv2_dinov3_x_coco_300query_n_batch.onnx
INFO: providers: ['TensorrtExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: input_bgr shape: [1, 3, 640, 640] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  90.57736396789551 ms
INFO: avg elapsed time per pred:  9.05773639678955 ms
INFO: output_name.1: label_xyxy_score shape: [1, 300, 6] dtype: float3
```
```bash
uv run sit4onnx \
-if deimv2_hgnetv2_atto_coco_100query_n_batch.onnx \
-oep tensorrt \
-b 1

INFO: file: deimv2_hgnetv2_atto_coco_100query_n_batch.onnx
INFO: providers: ['TensorrtExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: images shape: [1, 3, 320, 320] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  9.71078872680664 ms
INFO: avg elapsed time per pred:  0.9710788726806641 ms
INFO: output_name.1: label_xyxy_score shape: [1, 100, 6] dtype: float32

uv run sit4onnx \
-if deimv2_hgnetv2_atto_coco_100query_n_batch.onnx \
-oep tensorrt \
-b 3

INFO: file: deimv2_hgnetv2_atto_coco_100query_n_batch.onnx
INFO: providers: ['TensorrtExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: images shape: [3, 3, 320, 320] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  12.964248657226562 ms
INFO: avg elapsed time per pred:  1.2964248657226562 ms
INFO: output_name.1: label_xyxy_score shape: [3, 100, 6] dtype: float3

uv run sit4onnx \
-if deimv2_hgnetv2_atto_coco_100query_n_batch.onnx \
-oep cpu \
-b 1

INFO: file: deimv2_hgnetv2_atto_coco_100query_n_batch.onnx
INFO: providers: ['CPUExecutionProvider']
INFO: input_name.1: images shape: [1, 3, 320, 320] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  71.84863090515137 ms
INFO: avg elapsed time per pred:  7.184863090515137 ms
INFO: output_name.1: label_xyxy_score shape: [1, 100, 6] dtype: float32
```
