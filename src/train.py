from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .data import BiomeDataset, compute_class_weights, compute_sample_weights
from .utils import CHECKPOINTS_DIR, LOGS_DIR, get_logger

log = get_logger("train")


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_loaders(
    split_csv: Path,
    num_classes: int,
    train_tf,
    eval_tf,
    batch_size: int = 64,
    num_workers: int = 4,
    use_weighted_sampler: bool = False,
    sampler_power: float = 0.5,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders from the persisted split CSV.

    When `use_weighted_sampler=True`, the train loader draws samples with
    probability proportional to `1 / count**sampler_power`. Combined with a
    class-weighted loss this rebalances twice — usually harmful (see v1
    results); prefer either sampler OR loss weight, not both. `sampler_power`
    only kicks in when the sampler is enabled.
    """
    df = pd.read_csv(split_csv)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_ds = BiomeDataset(train_df, transform=train_tf)
    val_ds = BiomeDataset(val_df, transform=eval_tf)
    test_ds = BiomeDataset(test_df, transform=eval_tf)

    sampler: WeightedRandomSampler | None = None
    if use_weighted_sampler:
        w = compute_sample_weights(train_df["label"].tolist(), num_classes,
                                    power=sampler_power)
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    val_loss: float
    val_f1_macro: float
    lr: float
    seconds: float


@dataclass
class TrainResult:
    best_epoch: int
    best_val_f1: float
    best_ckpt: Path
    history: list[EpochStats] = field(default_factory=list)


def _run_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one epoch. If `optimizer` is None -> eval mode."""
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    n_seen = 0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    ctx = torch.enable_grad() if train else torch.inference_mode()
    with ctx:
        for x, y in tqdm(loader, desc=desc, leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            n_seen += bs
            all_preds.append(logits.argmax(1).detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())

    return (
        total_loss / max(n_seen, 1),
        np.concatenate(all_preds),
        np.concatenate(all_targets),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    num_classes: int,
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    class_weights: torch.Tensor | None = None,
    run_name: str = "run",
    patience: int = 6,
    label_smoothing: float = 0.0,
) -> TrainResult:
    """Train `model` on `train_loader`, evaluate on `val_loader` each epoch.

    Selects the checkpoint with the highest val macro-F1. Early-stops if val
    macro-F1 doesn't improve for `patience` consecutive epochs.
    """
    model.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    ckpt_path = CHECKPOINTS_DIR / f"{run_name}_best.pt"
    log_path = LOGS_DIR / f"{run_name}.csv"

    best_f1 = -1.0
    best_epoch = -1
    since_improve = 0
    history: list[EpochStats] = []

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss",
                         "val_f1_macro", "lr", "seconds"])

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, _, _ = _run_epoch(
            model, train_loader, criterion, optimizer, device, f"epoch {epoch} train"
        )
        val_loss, val_preds, val_targets = _run_epoch(
            model, val_loader, criterion, None, device, f"epoch {epoch} val"
        )

        val_f1 = float(f1_score(val_targets, val_preds, average="macro", zero_division=0))
        cur_lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        stats = EpochStats(epoch, train_loss, val_loss, val_f1, cur_lr, dt)
        history.append(stats)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}",
                                     f"{val_f1:.4f}",
                                     f"{cur_lr:.2e}", f"{dt:.1f}"])

        log.info(
            "[%s] ep %02d/%d  train_loss=%.4f  val_loss=%.4f  val_f1=%.4f  lr=%.2e  (%.1fs)",
            run_name, epoch, epochs, train_loss, val_loss, val_f1, cur_lr, dt,
        )

        # Scheduler step (Plateau needs a metric, others don't).
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()

        # Checkpoint selection.
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            since_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_f1_macro": val_f1,
                    "run_name": run_name,
                },
                ckpt_path,
            )
            log.info("  ↳ new best val_f1=%.4f  saved -> %s", val_f1, ckpt_path.name)
        else:
            since_improve += 1
            if since_improve >= patience:
                log.info("Early stopping: val_f1 hasn't improved for %d epochs.", patience)
                break

    return TrainResult(best_epoch=best_epoch, best_val_f1=best_f1,
                       best_ckpt=ckpt_path, history=history)


# ---------------------------------------------------------------------------
# Convenience: derive class-weights tensor from the split CSV
# ---------------------------------------------------------------------------

def class_weights_from_split(
    split_csv: Path,
    num_classes: int,
    *,
    power: float = 0.5,
) -> torch.Tensor:
    """Derive class weights from the training split, tempered by `power`.
    Default `0.5` = sqrt-tempered; see `compute_class_weights` for the
    trade-off. Set `power=0` to disable weighting altogether.
    """
    df = pd.read_csv(split_csv)
    train_labels = df[df["split"] == "train"]["label"].tolist()
    return compute_class_weights(train_labels, num_classes, power=power)
