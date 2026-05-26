#!/usr/bin/env python3
"""Update resume augmentation schedule in a checkpoint and config YAML.

This edits the transform policy saved inside a DEIMv2 resume checkpoint and the
matching schedule values in the config file. Only run this on checkpoints you
trust: editing resume state requires torch.load(..., weights_only=False).
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import torch


DEFAULT_CHECKPOINT = Path("ckpts/last_full_epoch.pth")
DEFAULT_CONFIG = Path("configs/deimv2/deimv2_dinov3_x_wholebody49_ins_s08_maskhead256x3_center.yml")


def _update_checkpoint(checkpoint_path: Path, output_path: Path, end_epoch: int, dry_run: bool = False) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    try:
        policy = checkpoint["resume_meta"]["train_loader_state"]["transform_state"]["policy"]
    except KeyError as exc:
        raise KeyError(
            "checkpoint does not contain resume_meta.train_loader_state.transform_state.policy"
        ) from exc

    policy_epoch = policy.get("epoch")
    if not (isinstance(policy_epoch, list) and len(policy_epoch) == 3):
        raise ValueError(f"expected policy.epoch to be a 3-item list, got: {policy_epoch!r}")

    old_value = policy_epoch[2]
    policy_epoch[2] = end_epoch

    if dry_run:
        print(f"checkpoint: would update policy.epoch[2] {old_value} -> {end_epoch} ({output_path})")
        return

    if output_path == checkpoint_path:
        tmp_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(checkpoint_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, output_path)

    print(f"checkpoint: policy.epoch[2] {old_value} -> {end_epoch} ({output_path})")


def _replace_once(text: str, pattern: str, repl: str, label: str) -> tuple[str, int]:
    updated, count = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"failed to update {label}: expected exactly one match, got {count}")
    return updated, count


def _update_config(config_path: Path, policy_end_epoch: int, collate_end_epoch: int, dry_run: bool = False) -> None:
    text = config_path.read_text()

    # Preserve indentation and inline comments while changing only the trailing epoch value.
    text, _ = _replace_once(
        text,
        r"^(\s*epoch:\s*\[\s*4\s*,\s*29\s*,\s*)\d+(\s*\].*)$",
        rf"\g<1>{policy_end_epoch}\g<2>",
        "train_dataloader.dataset.transforms.policy.epoch",
    )
    text, _ = _replace_once(
        text,
        r"^(\s*stop_epoch:\s*)\d+(\s*(?:#.*)?)$",
        rf"\g<1>{collate_end_epoch}\g<2>",
        "train_dataloader.collate_fn.stop_epoch",
    )
    text, _ = _replace_once(
        text,
        r"^(\s*copyblend_epochs:\s*\[\s*4\s*,\s*)\d+(\s*\].*)$",
        rf"\g<1>{collate_end_epoch}\g<2>",
        "train_dataloader.collate_fn.copyblend_epochs",
    )

    if dry_run:
        print(
            "config: would update "
            f"policy.epoch[2] -> {policy_end_epoch}, "
            f"stop_epoch/copyblend_epochs[1] -> {collate_end_epoch} ({config_path})"
        )
        return

    config_path.write_text(text)
    print(
        "config: updated "
        f"policy.epoch[2] -> {policy_end_epoch}, "
        f"stop_epoch/copyblend_epochs[1] -> {collate_end_epoch} ({config_path})"
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update policy.epoch[2] in a resume checkpoint and update the matching "
            "policy/stop_epoch/copyblend_epochs values in a config YAML."
        )
    )
    parser.add_argument(
        "end_epoch",
        type=_positive_int,
        help="new trailing epoch value for both policy and collate settings, e.g. 54 or 58",
    )
    parser.add_argument(
        "--policy-end-epoch",
        type=_positive_int,
        default=None,
        help="override only policy.epoch[2] in the checkpoint and config",
    )
    parser.add_argument(
        "--collate-end-epoch",
        type=_positive_int,
        default=None,
        help="override only config stop_epoch and copyblend_epochs[1]",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help=f"default: {DEFAULT_CHECKPOINT}")
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=None,
        help="write checkpoint to this path instead of modifying --checkpoint in place",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"default: {DEFAULT_CONFIG}")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="create .bak copies before modifying files in place",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="only update the config YAML",
    )
    parser.add_argument(
        "--checkpoint-only",
        action="store_true",
        help="only update the checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print intended changes without writing files",
    )
    return parser.parse_args()


def _backup(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)
    print(f"backup: {path} -> {backup_path}")


def main() -> None:
    args = parse_args()

    if args.config_only and args.checkpoint_only:
        raise SystemExit("--config-only and --checkpoint-only cannot be used together")

    policy_end_epoch = args.policy_end_epoch if args.policy_end_epoch is not None else args.end_epoch
    collate_end_epoch = args.collate_end_epoch if args.collate_end_epoch is not None else args.end_epoch

    if not args.checkpoint_only:
        if args.backup and not args.dry_run:
            _backup(args.config)
        _update_config(args.config, policy_end_epoch, collate_end_epoch, dry_run=args.dry_run)

    if not args.config_only:
        output_checkpoint: Path = args.output_checkpoint or args.checkpoint
        if args.backup and not args.dry_run and output_checkpoint == args.checkpoint:
            _backup(args.checkpoint)
        _update_checkpoint(args.checkpoint, output_checkpoint, policy_end_epoch, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
