# Export

> [!warning]
> The pre-trained weights for the coco dataset published in the official paper implementation cannot be used. Only the re-trained weights in this fork repository can be processed successfully. If you don't need to deploy to a framework other than PyTorch, there is no need to retrain. If the output from TFLite, TF.JS, or CoreML is unexpected, try converting it using the model before running `make_prep.py`.

|deimv2_dinov3_x_coco.pth -> onnx|If re-trained by this fork -> onnx|
|:-:|:-:|
|<img width="480" height="360" alt="image" src="https://github.com/user-attachments/assets/a125ba79-e592-4be6-b8c2-914331b7d167" />|<img width="480" height="360" alt="image" src="https://github.com/user-attachments/assets/f5e60ec3-d792-419e-b126-a1ca99988662" />|
|![000000009420](https://github.com/user-attachments/assets/229df7e8-d86e-4e8c-a05c-442a9e4aa501)|![000000009420](https://github.com/user-attachments/assets/a1abdb8f-66c3-45f2-8c6a-95a725ca496f)|

```bash
cd ../..

################################################## X
WEIGHT=deimv2_dinov3_x_wholebody34
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

uv run python tools/deployment/make_prep.py -m ${WEIGHT}_${QUERIES}query.onnx -s 1 3 ${H} ${W}

################################################## S
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

uv run python tools/deployment/make_prep.py -m ${WEIGHT}_${QUERIES}query.onnx -s 1 3 ${H} ${W}

################################################## N
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

uv run python tools/deployment/make_prep.py -m ${WEIGHT}_${QUERIES}query.onnx -s 1 3 ${H} ${W}

################################################## Pico
WEIGHT=deimv2_hgnetv2_pico_wholebody34
H=640
W=640
QUERIES=340

uv run python tools/deployment/export_onnx.py \
-c configs/deimv2/${WEIGHT}.yml \
-r ckpts/${WEIGHT}.pth \
--opset 17 \
--size ${H} ${W}

uv run python tools/deployment/export_onnx.py \
-c configs/deimv2/${WEIGHT}.yml \
-r ckpts/${WEIGHT}.pth \
--opset 17 \
--dynamic_batch \
--simplify \
--size ${H} ${W}
uv run onnxslim ${WEIGHT}_${QUERIES}query_n_batch.onnx ${WEIGHT}_${QUERIES}query_n_batch.onnx
uv run onnxsim ${WEIGHT}_${QUERIES}query_n_batch.onnx ${WEIGHT}_${QUERIES}query_n_batch.onnx

rm ${WEIGHT}_${QUERIES}query.onnx

uv run onnxsim ${WEIGHT}_${QUERIES}query_n_batch.onnx ${WEIGHT}_${QUERIES}query.onnx \
--overwrite-input-shape "images:1,3,${H},${W}"

uv run python tools/deployment/make_prep.py -m ${WEIGHT}_${QUERIES}query.onnx -s 1 3 ${H} ${W}

################################################## Femto
WEIGHT=deimv2_hgnetv2_femto_wholebody34
H=416
W=416
QUERIES=340

uv run python tools/deployment/export_onnx.py \
-c configs/deimv2/${WEIGHT}.yml \
-r ckpts/${WEIGHT}.pth \
--opset 17 \
--size ${H} ${W}

uv run python tools/deployment/export_onnx.py \
-c configs/deimv2/${WEIGHT}.yml \
-r ckpts/${WEIGHT}.pth \
--opset 17 \
--dynamic_batch \
--simplify \
--size ${H} ${W}
uv run onnxslim ${WEIGHT}_${QUERIES}query_n_batch.onnx ${WEIGHT}_${QUERIES}query_n_batch.onnx
uv run onnxsim ${WEIGHT}_${QUERIES}query_n_batch.onnx ${WEIGHT}_${QUERIES}query_n_batch.onnx

rm ${WEIGHT}_${QUERIES}query.onnx

uv run onnxsim ${WEIGHT}_${QUERIES}query_n_batch.onnx ${WEIGHT}_${QUERIES}query.onnx \
--overwrite-input-shape "images:1,3,${H},${W}"

uv run python tools/deployment/make_prep.py -m ${WEIGHT}_${QUERIES}query.onnx -s 1 3 ${H} ${W}

################################################## Atto
WEIGHT=deimv2_hgnetv2_atto_wholebody34
H=160
W=160
QUERIES=170

uv run python tools/deployment/export_onnx.py \
-c configs/deimv2/${WEIGHT}.yml \
-r ckpts/${WEIGHT}.pth \
--opset 17 \
--size ${H} ${W}

uv run python tools/deployment/export_onnx.py \
-c configs/deimv2/${WEIGHT}.yml \
-r ckpts/${WEIGHT}.pth \
--opset 17 \
--dynamic_batch \
--simplify \
--size ${H} ${W}
uv run onnxslim ${WEIGHT}_${QUERIES}query_n_batch.onnx ${WEIGHT}_${QUERIES}query_n_batch.onnx
uv run onnxsim ${WEIGHT}_${QUERIES}query_n_batch.onnx ${WEIGHT}_${QUERIES}query_n_batch.onnx

rm ${WEIGHT}_${QUERIES}query.onnx

uv run onnxsim ${WEIGHT}_${QUERIES}query_n_batch.onnx ${WEIGHT}_${QUERIES}query.onnx \
--overwrite-input-shape "images:1,3,${H},${W}"

uv run python tools/deployment/make_prep.py -m ${WEIGHT}_${QUERIES}query.onnx -s 1 3 ${H} ${W}
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
# LiteRT (TFLite) / TensorFlow saved_model
```bash
uv run onnx2tf -i ${WEIGHT}_${QUERIES}query.onnx
```
<img width="775" height="699" alt="image" src="https://github.com/user-attachments/assets/c5283843-7989-45cd-9ccf-e772b5f9207c" />

# TensorFlow.js + WebGPU
```bash
uv pip install --no-deps \
tensorflowjs==4.22.0 \
tensorflow_decision_forests==1.12.0 \
ydf==0.13.0 \
tensorflow_hub==0.16.1 \
h5py==3.14.0 \
tensorflow==2.19.0 \
wrapt==1.17.3 \
gast==0.6.0 \
astunparse==1.6.3 \
opt_einsum==3.4.0 \
tf-keras==2.19.0 \
jax==0.7.2 \
jaxlib==0.7.2 \
sit4tfjs==1.0.0

sed -i 's/^import tensorflow\.keras as keras$/import tf_keras as keras/' .venv/lib/python3.11/site-packages/tensorflowjs/converters/tf_module_mapper.py
sed -i '97,109c\  import tf_keras as keras' .venv/lib/python3.11/site-packages/tensorflow_hub/__init__.py
sed -i '25,31c import tf_keras as keras' .venv/lib/python3.11/site-packages/tensorflow_hub/keras_layer.py

uv run tensorflowjs_converter \
--input_format tf_saved_model \
--output_format tfjs_graph_model \
saved_model \
tfjs_model

uv run sit4tfjs \
--input_tfjs_file_path ./tfjs_model \
--fixed_shapes "input_bgr:1,640,640,3" \
--execution_provider webgpu
```
```
================================================================================
sit4tfjs - Simple Inference Test for TensorFlow.js
Benchmark tool for TensorFlow.js models
================================================================================

Model path: tfjs_model
Model Information:
  Format: graph-model
  Generated by: 2.17.0
  Converted by: TensorFlow.js Converter v4.22.0
  Inputs:
    input_bgr: [1, -1, -1, 3] (DT_FLOAT)
  Outputs:
    output_0: [1, 340, 6] (DT_FLOAT)
Detected TensorFlow.js model - using browser runtime with WEBGPU backend...
Successfully initialized browser runtime with WEBGPU backend
Model: tfjs_model
Test loops: 100
Batch size: 1
Runtime: TensorFlow.js (Browser - WEBGPU)

Input tensors:
  input_bgr: [1, 640, 640, 3] (DT_FLOAT)

Output tensors:
  output_0: [1, 340, 6] (DT_FLOAT)

Running browser benchmark...
Running 100 inference iterations...
Starting browser benchmark with backend: webgpu
  Completed 10/100 iterations
  Completed 20/100 iterations
  Completed 30/100 iterations
  Completed 40/100 iterations
  Completed 50/100 iterations
  Completed 60/100 iterations
  Completed 70/100 iterations
  Completed 80/100 iterations
  Completed 90/100 iterations
  Completed 100/100 iterations

============================================================
BENCHMARK RESULTS
============================================================
Average inference time: 8.045 ms
Minimum inference time: 6.129 ms
Maximum inference time: 9.714 ms
Standard deviation:     0.814 ms
============================================================
```

# CoreML
> [!warning]
> I don't own any Apple products, so although I have confirmed that the model conversion was successful, I don't have a debugging environment, so the behavior of the output model has not been verified.

- Fixed a critical bug in `coremltools`.
  ```bash
  sed -i '8a import tf_keras' .venv/lib/python3.11/site-packages/coremltools/converters/mil/frontend/tensorflow2/load.py
  sed -i 's/_tf\.keras\./tf_keras./g' .venv/lib/python3.11/site-packages/coremltools/converters/mil/frontend/tensorflow2/load.py
  ```
- Convert
  ```bash
  uv run python tools/deployment/convert_tf_to_coreml.py \
  --input saved_model \
  --sizes 640 640 \
  --output deimv2.mlpackage
  ```
  <img width="763" height="697" alt="image" src="https://github.com/user-attachments/assets/1302cdd6-3aec-46b7-b17c-c4cf4348aba2" />
