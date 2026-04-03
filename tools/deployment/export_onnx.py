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
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn

from engine.core import YAMLConfig, yaml_utils


@contextmanager
def patch_non_tensor_extra_state(module: nn.Module):
    patched = []
    for submodule in module.modules():
        if type(submodule).get_extra_state is nn.Module.get_extra_state:
            continue
        extra_state = submodule.get_extra_state()
        if torch.is_tensor(extra_state):
            continue
        original = submodule.get_extra_state
        submodule.get_extra_state = lambda: torch.empty(0)
        patched.append((submodule, original))

    try:
        yield
    finally:
        for submodule, original in patched:
            submodule.get_extra_state = original


def main(args, ):
    """main
    """
    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({'resume': args.resume})
    cfg = YAMLConfig(args.config, **update_dict)

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
            self.return_masks = args.with_masks
            self.return_contours = args.with_contours

        def forward(self, images, orig_target_sizes: torch.Tensor=None):
            outputs = self.model(
                images,
                return_masks=self.return_masks,
                return_contours=self.return_contours,
            )
            outputs = self.postprocessor(
                outputs,
                orig_target_sizes,
                return_masks=self.return_masks,
                return_contours=self.return_contours,
            )
            return outputs

    model = Model()

    img_size = cfg.yaml_cfg["eval_spatial_size"]
    dtsettings: dict = cfg.yaml_cfg.get("DEIMTransformer", None)
    num_queries = str(300)
    if dtsettings is not None:
        num_queries = str(dtsettings.get("num_queries", 300))
    data = torch.rand(1, 3, *img_size)
    _ = model(data)

    output_names = ['label_xyxy_score']
    if args.with_masks:
        output_names.append('masks')
    if args.with_contours:
        output_names.append('contours')

    dynamic_axes = {}
    if args.dynamic_batch:
        dynamic_axes = {
            'images': {0: 'N'},
            'label_xyxy_score': {0: 'N', 1: str(num_queries), 2: '6'},
        }
        if args.with_masks:
            dynamic_axes['masks'] = {0: 'N', 1: str(num_queries)}
        if args.with_contours:
            dynamic_axes['contours'] = {0: 'N', 1: str(num_queries)}

    output_file = f'{os.path.splitext(os.path.basename(args.config))[0]}_{num_queries}query'
    if args.with_masks:
        output_file = f'{output_file}_masks'
    if args.with_contours:
        output_file = f'{output_file}_contours'
    fp16_txt = '' if not args.fp16 else '_fp16'
    export_path = f'{output_file}{"_n_batch" if args.dynamic_batch else ""}{fp16_txt}.onnx'

    if not args.dynamic_batch:
        if not args.fp16:
            h, w = args.size
            data = torch.randn(1, 3, h, w)
            _ = model(data)

            with patch_non_tensor_extra_state(model):
                torch.onnx.export(
                    model,
                    (data),
                    export_path,
                    input_names=['images'],
                    output_names=output_names,
                    dynamic_axes=None,
                    opset_version=args.opset,
                )
        else:
            model.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                h, w = args.size
                data = torch.randn(1, 3, h, w, device="cuda")
                _ = model(data)

                with patch_non_tensor_extra_state(model):
                    torch.onnx.export(
                        model,
                        (data),
                        export_path,
                        input_names=['images'],
                        output_names=output_names,
                        dynamic_axes=None,
                        opset_version=args.opset,
                    )
    else:
        if not args.fp16:
            h, w = args.size
            data = torch.randn(1, 3, h, w)
            _ = model(data)

            with patch_non_tensor_extra_state(model):
                torch.onnx.export(
                    model,
                    (data),
                    export_path,
                    input_names=['images'],
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=args.opset,
                )
        else:
            model.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                h, w = args.size
                data = torch.randn(1, 3, h, w, device="cuda")
                _ = model(data)

                with patch_non_tensor_extra_state(model):
                    torch.onnx.export(
                        model,
                        (data),
                        export_path,
                        input_names=['images'],
                        output_names=output_names,
                        dynamic_axes=dynamic_axes,
                        opset_version=args.opset,
                    )

    if args.check:
        import onnx
        onnx_model = onnx.load(export_path)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx
        import onnxsim
        import onnxslim
        if not args.skip_onnxslim:
            onnx_model_slim = onnxslim.slim(export_path)
            onnx_model_simplify, check = onnxsim.simplify(onnx_model_slim)
        else:
            onnx_model_simplify, check = onnxsim.simplify(export_path)
        onnx.save(onnx_model_simplify, export_path)
        print(f'Simplify onnx model {check}...')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/deimv2/deimv2_dinov3_x_coco.yml', type=str)
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--size', '-s', nargs=2, default=[640,640], type=int)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--check',  action='store_true')
    parser.add_argument('--simplify',  action='store_true')
    parser.add_argument('--skip_onnxslim',  action='store_true')
    parser.add_argument('--dynamic_batch',  action='store_true')
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')
    parser.add_argument('--fp16', '-f', action='store_true')
    parser.add_argument('--with-masks', action='store_true')
    parser.add_argument('--with-contours', action='store_true')
    args = parser.parse_args()
    main(args)
