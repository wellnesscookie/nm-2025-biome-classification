"""Train the baseline 3-conv CNN from scratch.
    python -m scripts.train_baseline
    python -m scripts.train_baseline --epochs 5 --batch-size 128 --run-name baseline_v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data import (
    class_index_from_split,
    eval_transform,
    train_transform,
    NATIVE_H,
    NATIVE_W,
)
from src.models.baseline_cnn import BaselineCNN, count_parameters
from src.train import class_weights_from_split, make_loaders, train_model
from src.utils import SPLITS_DIR, get_device, get_logger, seed_everything

log = get_logger("train_baseline")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-csv", type=Path, default=SPLITS_DIR / "split.csv")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.4,
                    help="v2 default: 0.4 (was 0.3) — baseline overfits fast.")
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=8,
                    help="v2 default: 8 (was 6) — sqrt-tempered weights need "
                         "a few more epochs to settle.")
    ap.add_argument("--use-weighted-sampler", action="store_true",
                    help="Opt-in. v1 combined sampler + weighted loss and "
                         "over-corrected; v2 defaults to loss-only.")
    ap.add_argument("--sampler-power", type=float, default=0.5,
                    help="Tempering exponent for the sampler (if enabled). "
                         "0.5=sqrt (default), 1.0=inverse-frequency.")
    ap.add_argument("--class-weight-power", type=float, default=0.5,
                    help="Tempering for CrossEntropy class weights. 0=off, "
                         "0.5=sqrt (default), 1.0=inverse-frequency (v1).")
    ap.add_argument("--label-smoothing", type=float, default=0.05,
                    help="Cross-entropy label smoothing.")
    ap.add_argument("--strong-aug", action="store_true", default=True,
                    help="RandomResizedCrop + RandomErasing (default on for "
                         "v2). Pass --no-strong-aug to disable.")
    ap.add_argument("--no-strong-aug", dest="strong_aug", action="store_false")
    ap.add_argument("--image-size", type=int, nargs=2, default=(NATIVE_H, NATIVE_W))
    ap.add_argument("--run-name", type=str, default="baseline_v2")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    log.info("Device: %s", device)

    cidx = class_index_from_split(args.split_csv)
    n_classes = cidx.num_classes
    log.info("Classes: %d (from %s)", n_classes, args.split_csv.name)

    train_tf = train_transform(tuple(args.image_size), strong=args.strong_aug)
    eval_tf = eval_transform(tuple(args.image_size))

    train_loader, val_loader, _ = make_loaders(
        args.split_csv,
        num_classes=n_classes,
        train_tf=train_tf,
        eval_tf=eval_tf,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=args.use_weighted_sampler,
        sampler_power=args.sampler_power,
    )

    model = BaselineCNN(num_classes=n_classes, base_channels=args.base_channels,
                        dropout=args.dropout)
    log.info("Model params: %d", count_parameters(model))
    log.info(
        "Balancing: sampler=%s (power=%.2f) | class_weight_power=%.2f | "
        "label_smoothing=%.3f | strong_aug=%s",
        args.use_weighted_sampler, args.sampler_power, args.class_weight_power,
        args.label_smoothing, args.strong_aug,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
    )

    class_weights = (
        None if args.class_weight_power == 0.0
        else class_weights_from_split(args.split_csv, n_classes,
                                       power=args.class_weight_power)
    )

    result = train_model(
        model, train_loader, val_loader,
        num_classes=n_classes, device=device,
        epochs=args.epochs, optimizer=optimizer, scheduler=scheduler,
        class_weights=class_weights, run_name=args.run_name,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
    )
    log.info("Best epoch %d  val_f1_macro=%.4f  ckpt=%s",
             result.best_epoch, result.best_val_f1, result.best_ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
