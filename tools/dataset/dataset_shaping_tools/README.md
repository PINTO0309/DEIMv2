# Dataset Shaping Tools

This directory contains helper scripts for dataset restructuring and annotation conversion.

## `12_make_wholebody40_ins_annotation.py`

This script generates:

- `instances_trainval2017_person_only_no_crowd.json`
- `train_ins.json`
- `val_ins.json`
- `merge_person_masks_report.json`

These generated files are intended to be used as follows in the instance segmentation pipeline:

- `train_ins.json` for training mask supervision
- `val.json` for bbox validation
- `val_ins.json` for segm validation

It copies high-quality person `segmentation` annotations from MS-COCO person-only annotations into the body class (`category_id=0`) of the wholebody40 COCO annotations.

### What the script does

- Merges `instances_train2017_person_only_no_crowd.json` and `instances_val2017_person_only_no_crowd.json`
- Matches wholebody40 body annotations to COCO person annotations on the same source image
- Rescales donor polygons from the original COCO image size to the resized wholebody40 image size
- Uses one-to-one assignment with IoU-based matching
- Recomputes `area` from the imported polygon for matched body annotations
- Sets unmatched body annotations to:
  - `segmentation: []`
  - `area: 0`
- Initializes all non-body annotations to:
  - `segmentation: []`
  - `area: 0`
- Writes a JSON report with matching statistics
- Supports split validation GT usage, where bbox metrics should keep using `val.json` while segm metrics use `val_ins.json`

### Default command

```bash
python tools/dataset/dataset_shaping_tools/12_make_wholebody40_ins_annotation.py
```

### Explicit command example

```bash
python tools/dataset/dataset_shaping_tools/12_make_wholebody40_ins_annotation.py \
  --train-json /media/xxxxx/ExtremeSSD/make_wholebody40/train.json \
  --val-json /media/xxxxx/ExtremeSSD/make_wholebody40/val.json \
  --src-train-json /media/xxxxx/ExtremeSSD/make_wholebody40/instances_train2017_person_only_no_crowd.json \
  --src-val-json /media/xxxxx/ExtremeSSD/make_wholebody40/instances_val2017_person_only_no_crowd.json \
  --src-trainval-json /media/xxxxx/ExtremeSSD/make_wholebody40/instances_trainval2017_person_only_no_crowd.json \
  --train-out /media/xxxxx/ExtremeSSD/make_wholebody40/train_ins.json \
  --val-out /media/xxxxx/ExtremeSSD/make_wholebody40/val_ins.json \
  --report-json /media/xxxxx/ExtremeSSD/make_wholebody40/merge_person_masks_report.json
```

### Notes

- The script assumes wholebody40 body annotations use `category_id=0`.
- The donor COCO person annotations are assumed to use `category_id=1`.
- File matching is based on the 12-digit COCO image id at the beginning of `file_name`.
- Matching is accepted only when IoU is greater than or equal to the configured threshold.
- Progress is displayed with `tqdm`.

### Help

```bash
python tools/dataset/dataset_shaping_tools/12_make_wholebody40_ins_annotation.py --help
```
