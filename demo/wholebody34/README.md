# Demo

## Usage
```
usage: demo_deimv2_onnx_wholebody34_with_edges.py
[-h] [-m MODEL] (-v VIDEO | -i IMAGES_DIR) [-ep {cpu,cuda,tensorrt}]
[-it {fp16,int8}] [-dvw] [-dwk] [-ost OBJECT_SOCRE_THRESHOLD]
[-ast ATTRIBUTE_SOCRE_THRESHOLD] [-kst KEYPOINT_THRESHOLD]
[-kdm {dot,box,both}] [-ebm] [-dnm] [-dgm] [-dlr] [-dhm]
[-drc [DISABLE_RENDER_CLASSIDS ...]] [-efm] [-oyt]
[-bblw BOUNDING_BOX_LINE_WIDTH]

options:
  -h, --help
    show this help message and exit
  -m MODEL, --model MODEL
    ONNX/TFLite file path for DEIMv2.
  -v VIDEO, --video VIDEO
    Video file path or camera index.
  -i IMAGES_DIR, --images_dir IMAGES_DIR
    jpg, png images folder path.
  -ep {cpu,cuda,tensorrt}, --execution_provider {cpu,cuda,tensorrt}
    Execution provider for ONNXRuntime.
  -it {fp16,int8}, --inference_type {fp16,int8}
    Inference type. Default: fp16
  -dvw, --disable_video_writer
    Disable video writer. Eliminates the file I/O load associated with automatic
    recording to MP4. Devices that use a MicroSD card or similar for main storage
    can speed up overall processing.
  -dwk, --disable_waitKey
    Disable cv2.waitKey(). When you want to process a batch of still images,
    disable key-input wait and process them continuously.
  -ost OBJECT_SOCRE_THRESHOLD, --object_socre_threshold OBJECT_SOCRE_THRESHOLD
    The detection score threshold for object detection. Default: 0.35
  -ast ATTRIBUTE_SOCRE_THRESHOLD, --attribute_socre_threshold ATTRIBUTE_SOCRE_THRESHOLD
    The attribute score threshold for object detection. Default: 0.70
  -kst KEYPOINT_THRESHOLD, --keypoint_threshold KEYPOINT_THRESHOLD
    The keypoint score threshold for object detection. Default: 0.25
  -kdm {dot,box,both}, --keypoint_drawing_mode {dot,box,both}
    Key Point Drawing Mode. Default: dot
  -ebm, --enable_bone_drawing_mode
    Enable bone drawing mode. (Press B on the keyboard to switch modes)
  -dnm, --disable_generation_identification_mode
    Disable generation identification mode. (Press N on the keyboard to switch modes)
  -dgm, --disable_gender_identification_mode
    Disable gender identification mode. (Press G on the keyboard to switch modes)
  -dlr, --disable_left_and_right_hand_identification_mode
    Disable left and right hand identification mode. (Press H on the keyboard to switch modes)
  -dhm, --disable_headpose_identification_mode
    Disable HeadPose identification mode. (Press P on the keyboard to switch modes)
  -drc [DISABLE_RENDER_CLASSIDS ...], --disable_render_classids [DISABLE_RENDER_CLASSIDS ...]
    Class ID to disable bounding box drawing. List[int]. e.g. -drc 17 18 19
  -efm, --enable_face_mosaic
    Enable face mosaic.
  -oyt, --output_yolo_format_text
    Output YOLO format texts and images.
  -bblw BOUNDING_BOX_LINE_WIDTH, --bounding_box_line_width BOUNDING_BOX_LINE_WIDTH
    Bounding box line width. Default: 2
```
### Image files
```bash
uv run python demo/wholebody34/demo_deimv2_onnx_wholebody34_with_edges.py \
-m deimv2_dinov3_x_wholebody34_1750query_n_batch.onnx \
-i images_partial \
-ep cuda \
-dwk \
-dgm \
-dnm \
-dhm \
-dlr
```
|Image|Image|
|:-:|:-:|
|![000000009420](https://github.com/user-attachments/assets/a12b8f9d-0277-4a3c-8f06-faa58cfc06f8)|![000000014428](https://github.com/user-attachments/assets/f62fe90f-4933-4702-a0c3-438ded0790cd)|
### USBCam or Video files
```bash
uv run python demo/wholebody34/demo_deimv2_onnx_wholebody34_with_edges.py \
-m deimv2_dinov3_x_wholebody34_1750query_n_batch.onnx \
-v 0 \
-ep tensorrt \
-dwk \
-dgm \
-dnm \
-dhm \
-dlr
```
### 1,750 query
