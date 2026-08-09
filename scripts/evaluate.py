from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.data import (
    BiomeDataset,
    class_index_from_split,
    eval_transform,
    NATIVE_H,
    NATIVE_W,
)
from src.evaluate import build_report, collect_predictions, save_report
from src.models.baseline_cnn import BaselineCNN
from src.models.efficientnet import EFFICIENTNET_B0_INPUT, build_efficientnet_b0
from src.utils import SPLITS_DIR, get_device, get_logger

log = get_logger("evaluate_cli")


def build_model(arch: str, num_classes: int) -> torch.nn.Module:
    if arch == "baseline":
        return BaselineCNN(num_classes=num_classes)
    if arch == "efficientnet":
        # pretrained=False -> skip weight download; the checkpoint replaces them anyway
        return build_efficientnet_b0(num_classes=num_classes, pretrained=False)
    raise ValueError(f"unknown arch: {arch}")


def default_image_size(arch: str) -> tuple[int, int]:
    if arch == "efficientnet":
        return EFFICIENTNET_B0_INPUT
    return (NATIVE_H, NATIVE_W)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=["baseline", "efficientnet"], required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, default=SPLITS_DIR / "split.csv")
    ap.add_argument("--which", choices=["val", "test"], default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--image-size", type=int, nargs=2, default=None,
                    metavar=("H", "W"), help="override the default input size")
    ap.add_argument("--run-name", type=str, default=None,
                    help="artifact filename stem; defaults to <arch>_<which>")
    args = ap.parse_args()

    device = get_device()
    log.info("Device: %s", device)

    cidx = class_index_from_split(args.split_csv)
    n_classes = cidx.num_classes

    h, w = tuple(args.image_size) if args.image_size else default_image_size(args.arch)
    tf = eval_transform((h, w))

    df = pd.read_csv(args.split_csv)
    sub = df[df["split"] == args.which].reset_index(drop=True)
    log.info("Evaluating on %s split: %d images across %d classes",
             args.which, len(sub), sub["label"].nunique())
    ds = BiomeDataset(sub, transform=tf)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = build_model(args.arch, n_classes)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    log.info("Loaded checkpoint %s (epoch %s, val_f1_macro %.4f)",
             args.ckpt.name, ckpt.get("epoch"), ckpt.get("val_f1_macro", float("nan")))

    # keep_logits was previously enabled to compute top-5 accuracy; that
    # metric is dropped so we save the memory here.
    preds, targets, _ = collect_predictions(model, loader, device, keep_logits=False)
    report = build_report(preds, targets, cidx.label_to_name)

    run_name = args.run_name or f"{args.arch}_{args.which}"
    paths = save_report(report, run_name, cidx.label_to_name)

    log.info("== %s ==", run_name)
    log.info("Precision (macro) : %.4f", report.precision_macro)
    log.info("Recall    (macro) : %.4f", report.recall_macro)
    log.info("F1        (macro) : %.4f", report.f1_macro)
    log.info("F1        (weighted) : %.4f", report.f1_weighted)
    for k, p in paths.items():
        log.info("  wrote %-9s -> %s", k, p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
