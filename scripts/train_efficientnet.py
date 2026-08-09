"""Fine-tune EfficientNet-B0 on the biome dataset.

Two-phase schedule:

    Phase 1 (head only)  : backbone frozen, LR = --head-lr, --head-epochs epochs
    Phase 2 (all params) : full network unfrozen, LR = --fine-lr with cosine
                           decay over the remaining epochs.

Usage:
    python -m scripts.train_efficientnet
    python -m scripts.train_efficientnet --epochs 12 --head-epochs 2 --batch-size 32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data import class_index_from_split, eval_transform, train_transform
from src.models.efficientnet import EFFICIENTNET_B0_INPUT, build_efficientnet_b0
from src.train import class_weights_from_split, make_loaders, train_model
from src.utils import SPLITS_DIR, get_device, get_logger, seed_everything

log = get_logger("train_efficientnet")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-csv", type=Path, default=SPLITS_DIR / "split.csv")
    ap.add_argument("--epochs", type=int, default=12, help="TOTAL epochs (phase1 + phase2).")
    ap.add_argument("--head-epochs", type=int, default=3, help="Phase-1 epochs with backbone frozen.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--fine-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--use-weighted-sampler", action="store_true",
                    help="Opt-in. v1 combined sampler + weighted loss and "
                         "over-corrected; v2 defaults to loss-only.")
    ap.add_argument("--sampler-power", type=float, default=0.5,
                    help="Tempering exponent for the sampler (if enabled).")
    ap.add_argument("--class-weight-power", type=float, default=0.5,
                    help="Tempering for CrossEntropy class weights. 0=off, "
                         "0.5=sqrt (default), 1.0=inverse-frequency (v1).")
    ap.add_argument("--label-smoothing", type=float, default=0.1,
                    help="Cross-entropy label smoothing; 0.1 is the standard "
                         "value for pretrained image-classifier fine-tunes.")
    ap.add_argument("--image-size", type=int, nargs=2, default=list(EFFICIENTNET_B0_INPUT))
    ap.add_argument("--run-name", type=str, default="efficientnet_v2")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    log.info("Device: %s", device)

    cidx = class_index_from_split(args.split_csv)
    n_classes = cidx.num_classes
    log.info("Classes: %d (from %s)", n_classes, args.split_csv.name)

    train_tf = train_transform(tuple(args.image_size))
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

    class_weights = (
        None if args.class_weight_power == 0.0
        else class_weights_from_split(args.split_csv, n_classes,
                                       power=args.class_weight_power)
    )
    log.info(
        "Balancing: sampler=%s (power=%.2f) | class_weight_power=%.2f | "
        "label_smoothing=%.3f",
        args.use_weighted_sampler, args.sampler_power, args.class_weight_power,
        args.label_smoothing,
    )

    model = build_efficientnet_b0(num_classes=n_classes, pretrained=not args.no_pretrained)

    # --- Phase 1: head only -------------------------------------------------
    if args.head_epochs > 0:
        log.info("Phase 1: training head only for %d epochs (lr=%g)",
                 args.head_epochs, args.head_lr)
        model.freeze_backbone()
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(head_params, lr=args.head_lr, weight_decay=args.weight_decay)
        train_model(
            model, train_loader, val_loader,
            num_classes=n_classes, device=device,
            epochs=args.head_epochs, optimizer=optimizer, scheduler=None,
            class_weights=class_weights, run_name=f"{args.run_name}_head",
            patience=args.patience,
            label_smoothing=args.label_smoothing,
        )

    # --- Phase 2: fine-tune everything -------------------------------------
    fine_epochs = max(args.epochs - args.head_epochs, 1)
    log.info("Phase 2: fine-tuning all params for %d epochs (lr=%g, cosine)",
             fine_epochs, args.fine_lr)
    model.unfreeze_all()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.fine_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fine_epochs)

    result = train_model(
        model, train_loader, val_loader,
        num_classes=n_classes, device=device,
        epochs=fine_epochs, optimizer=optimizer, scheduler=scheduler,
        class_weights=class_weights, run_name=args.run_name,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
    )
    log.info("Best epoch %d  val_f1_macro=%.4f  ckpt=%s",
             result.best_epoch, result.best_val_f1, result.best_ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
