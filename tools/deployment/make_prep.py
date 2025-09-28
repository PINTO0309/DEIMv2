#! /usr/bin/env python

import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
from argparse import ArgumentParser
from typing import List
import os

class Model(nn.Module):
    def __init__(self, input_shape: List[int], is_dinov3: bool=True):
        super(Model, self).__init__()
        self.input_shape = tuple(input_shape)

    def forward(self, x: torch.Tensor):
        x = torch.nn.functional.interpolate(
            input=x,
            size=self.input_shape[2:],
        )
        # BGR -> RGB
        x = torch.cat(
            [
                x[:, 2:3, ...],
                x[:, 1:2, ...],
                x[:, 0:1, ...],
            ],
            dim=1,
        )
        x = x * 0.003921569 # / 255.0
        if is_dinov3:
            # mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]
            x = (x - torch.Tensor([0.485, 0.456, 0.406]).reshape([1,3,1,1])) / torch.Tensor([0.229, 0.224, 0.225]).reshape([1,3,1,1])
        return x


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-o',
        '--opset',
        type=int,
        default=17,
        help='onnx opset'
    )
    parser.add_argument(
        '-s',
        '--input_shape',
        type=int,
        nargs=4,
        default=[1,3,640,640],
        help='input shape'
    )
    parser.add_argument(
        '-m',
        '--model_body',
        type=str,
        default="deimv2_dinov3_x_coco_300query.onnx",
        help='Model body'
    )
    args = parser.parse_args()

    MODEL = f'01_prep'
    OPSET=args.opset
    INPUT_SHAPE: List[int] = args.input_shape
    MODEL_BODY: str = args.model_body
    file_name: str = os.path.splitext(os.path.basename(MODEL_BODY))[0]
    is_dinov3: bool = 'dinov3' in file_name

    model = Model(input_shape=INPUT_SHAPE, is_dinov3=is_dinov3)

    onnx_file = f"{MODEL}_{'_'.join(map(str, INPUT_SHAPE))}.onnx"
    x = torch.randn(INPUT_SHAPE)

    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=OPSET,
        input_names = ['input_bgr'],
        output_names=['output_prep'],
        dynamic_axes=\
            {
                'input_bgr': {
                    2: 'H',
                    3: 'W',
                },
            }
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

    torch.onnx.export(
        model,
        args=(x),
        f=f"{MODEL}_{'_'.join(map(str, INPUT_SHAPE))}_n_batch.onnx",
        opset_version=OPSET,
        input_names = ['input_bgr'],
        output_names=['output_prep'],
        dynamic_axes=\
            {
                'input_bgr': {
                    0: 'N',
                    2: 'H',
                    3: 'W',
                },
            }
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

    from sor4onnx import rename
    rename(
        old_new=["/", "prep/"],
        input_onnx_file_path=onnx_file,
        search_mode="prefix_match",
        output_onnx_file_path=onnx_file,
    )

    import os
    from snc4onnx import combine

    combine(
        input_onnx_file_paths = [
            onnx_file,
            MODEL_BODY,
        ],
        srcop_destop = [
            ["output_prep", "images"]
        ],
        output_onnx_file_path=MODEL_BODY,
    )
    combine(
        input_onnx_file_paths = [
            f'{os.path.splitext(os.path.basename(onnx_file))[0]}_n_batch.onnx',
            f'{os.path.splitext(os.path.basename(MODEL_BODY))[0]}_n_batch.onnx',
        ],
        srcop_destop = [
            ["output_prep", "images"]
        ],
        output_onnx_file_path=f'{os.path.splitext(os.path.basename(MODEL_BODY))[0]}_n_batch.onnx',
    )