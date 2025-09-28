"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn

from engine.core import YAMLConfig


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes: torch.Tensor=None):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()

    img_size = cfg.yaml_cfg["eval_spatial_size"]
    dtsettings: dict = cfg.yaml_cfg.get("DEIMTransformer", None)
    num_queries = str(300)
    if dtsettings is not None:
        num_queries = str(dtsettings.get("num_queries", 300))
    data = torch.rand(1, 3, *img_size)
    _ = model(data)

    dynamic_axes = {}
    if args.dynamic_batch:
        dynamic_axes = {
            'images': {0: 'N'},
            'label_xyxy_score': {0: 'N', 1: str(num_queries), 2: '6'},
        }

    output_file = f'{os.path.splitext(os.path.basename(args.config))[0]}_{num_queries}query.onnx'
    fp16_txt = '' if not args.fp16 else '_fp16'

    if not args.dynamic_batch:
        if not args.fp16:
            h, w = args.size
            data = torch.randn(1, 3, h, w)
            _ = model(data)

            torch.onnx.export(
                model,
                (data),
                f'{os.path.splitext(os.path.basename(output_file))[0]}{fp16_txt}.onnx',
                input_names=['images'],
                output_names=['label_xyxy_score'],
                dynamic_axes=None,
                opset_version=17,
            )
        else:
            model.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                h, w = args.size
                data = torch.randn(1, 3, h, w, device="cuda")
                _ = model(data)

                torch.onnx.export(
                    model,
                    (data),
                    f'{os.path.splitext(os.path.basename(output_file))[0]}{fp16_txt}.onnx',
                    input_names=['images'],
                    output_names=['label_xyxy_score'],
                    dynamic_axes=None,
                    opset_version=17,
                )
    else:
        if not args.fp16:
            h, w = args.size
            data = torch.randn(1, 3, h, w)
            _ = model(data)

            torch.onnx.export(
                model,
                (data),
                f'{os.path.splitext(os.path.basename(output_file))[0]}_n_batch{fp16_txt}.onnx',
                input_names=['images'],
                output_names=['label_xyxy_score'],
                dynamic_axes=dynamic_axes,
                opset_version=17,
            )
        else:
            model.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                h, w = args.size
                data = torch.randn(1, 3, h, w, device="cuda")
                _ = model(data)

                torch.onnx.export(
                    model,
                    (data),
                    f'{os.path.splitext(os.path.basename(output_file))[0]}_n_batch{fp16_txt}.onnx',
                    input_names=['images'],
                    output_names=['label_xyxy_score'],
                    dynamic_axes=dynamic_axes,
                    opset_version=17,
                )

    if args.check:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx
        import onnxsim
        import onnxslim
        onnx_model_slim = onnxslim.slim(output_file)
        onnx_model_simplify, check = onnxsim.simplify(onnx_model_slim)
        onnx.save(onnx_model_simplify, output_file)
        print(f'Simplify onnx model {check}...')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/deimv2/deimv2_dinov3_x_coco.yml', type=str)
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--size', '-s', nargs=2, default=[640,640], type=int)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--check',  action='store_true', default=True)
    parser.add_argument('--simplify',  action='store_true', default=True)
    parser.add_argument('--dynamic_batch',  action='store_true')
    parser.add_argument('--fp16', '-f', action='store_true')
    args = parser.parse_args()
    main(args)
