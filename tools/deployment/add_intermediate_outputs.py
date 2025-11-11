#!/usr/bin/env python
"""
Utility script to append intermediate feature tensors to an existing DEIMv2 ONNX graph.

This avoids re-exporting the model by promoting already-present tensors (backbone,
encoder, decoder activations) to graph outputs.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import onnx
from onnx import helper


# Mapping from new user-friendly output names to the tensor names that already
# exist inside the DEIMv2 ONNX graph.
DEFAULT_TENSORS: "OrderedDict[str, str]" = OrderedDict(
    [
        # Backbone (DINOv3 + STA fusion) projections.
        ("backbone_stage2", "/model/backbone/convs.0/Conv_output_0"),
        ("backbone_stage3", "/model/backbone/convs.1/Conv_output_0"),
        ("backbone_stage4", "/model/backbone/convs.2/Conv_output_0"),
        # Hybrid encoder (FPN+PAN) outputs fed into the decoder.
        ("encoder_stage2", "/model/encoder/fpn_blocks.1/cv4/act/Mul_output_0"),
        ("encoder_stage3", "/model/encoder/pan_blocks.0/cv4/act/Mul_output_0"),
        ("encoder_stage4", "/model/encoder/pan_blocks.1/cv4/act/Mul_output_0"),
        # Decoder heads.
        ("decoder_hidden", "/model/decoder/decoder/Add_52_output_0"),
        ("decoder_logits", "/model/decoder/decoder/dec_score_head.5/Add_output_0"),
        ("decoder_boxes", "/model/decoder/decoder/Concat_12_output_0"),
        ("decoder_qualities", "/model/decoder/decoder/lqe_layers.5/Add_output_0"),
    ]
)


def _gather_value_infos(graph: onnx.GraphProto) -> Dict[str, onnx.ValueInfoProto]:
    """Create a quick lookup for every value info in the graph."""
    buckets: Iterable[Iterable[onnx.ValueInfoProto]] = (
        graph.input,
        graph.output,
        graph.value_info,
    )
    lookup: Dict[str, onnx.ValueInfoProto] = {}
    for bucket in buckets:
        for value_info in bucket:
            lookup[value_info.name] = value_info
    return lookup


def _clone_value_info(template: onnx.ValueInfoProto, new_name: str) -> onnx.ValueInfoProto:
    """Duplicate the metadata so ONNX runtimes know the shape/dtype of the new output."""
    clone = onnx.ValueInfoProto()
    clone.CopyFrom(template)
    clone.name = new_name
    return clone


def promote_tensors(
    model: onnx.ModelProto,
    tensor_map: "OrderedDict[str, str]",
    verbose: bool = True,
) -> Tuple[int, int]:
    """Append tensors from `tensor_map` to graph outputs."""
    graph = model.graph
    lookup = _gather_value_infos(graph)
    existing_outputs = {out.name for out in graph.output}
    promoted = 0
    skipped = 0

    for new_name, source_name in tensor_map.items():
        if new_name in existing_outputs:
            skipped += 1
            if verbose:
                print(f"[skip] Output named '{new_name}' already exists.")
            continue

        template = lookup.get(source_name)
        if template is None:
            raise KeyError(
                f"Tensor '{source_name}' not found in graph; cannot promote it to an output."
            )

        # Create Identity node to forward the tensor to a user-friendly output name.
        identity_node = helper.make_node(
            "Identity",
            inputs=[source_name],
            outputs=[new_name],
            name=f"promote::{new_name}",
        )

        graph.node.extend([identity_node])
        graph.output.extend([_clone_value_info(template, new_name)])
        existing_outputs.add(new_name)
        promoted += 1
        if verbose:
            print(f"[add] {new_name}  <--  {source_name}")

    return promoted, skipped


def parse_extra_mappings(pairs: Iterable[str]) -> "OrderedDict[str, str]":
    """Parse custom `name=existing_tensor` pairs."""
    mapping: "OrderedDict[str, str]" = OrderedDict()
    for raw in pairs:
        if "=" not in raw:
            raise argparse.ArgumentTypeError(
                f"Invalid mapping '{raw}'. Expected format new_name=existing_tensor_name"
            )
        new_name, tensor_name = raw.split("=", 1)
        new_name = new_name.strip()
        tensor_name = tensor_name.strip()
        if not new_name or not tensor_name:
            raise argparse.ArgumentTypeError(f"Invalid mapping '{raw}'.")
        mapping[new_name] = tensor_name
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add intermediate DEIMv2 features as ONNX outputs without re-exporting the model."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Path to the source ONNX file (e.g. deimv2_dinov3_x_wholebody34_1750query_n_batch_640x640.onnx).",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Destination path for the augmented ONNX (will be overwritten).",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="NEW=ORIGINAL",
        help="Optional additional tensor promotions (repeatable).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress per-output logging.",
    )
    args = parser.parse_args()

    model = onnx.load(args.input)

    target_map = OrderedDict(DEFAULT_TENSORS)
    if args.extra:
        target_map.update(parse_extra_mappings(args.extra))

    added, skipped = promote_tensors(model, target_map, verbose=not args.quiet)
    onnx.save(model, args.output)

    if not args.quiet:
        print(f"Done. Added {added} outputs (skipped {skipped}).")


if __name__ == "__main__":
    main()
