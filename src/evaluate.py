from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import FIGURES_DIR, LOGS_DIR, get_logger

log = get_logger("eval")


@dataclass
class EvalReport:
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_micro: float
    recall_micro: float
    f1_micro: float
    f1_weighted: float
    per_class: pd.DataFrame           # columns: label, name, precision, recall, f1, support
    confusion: np.ndarray             # raw counts, shape (C, C)
    logits: np.ndarray | None = None  # optional — kept for further analysis


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    keep_logits: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (preds, targets[, logits]) for the entire loader."""
    model.eval()
    model.to(device)
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    logits_out: list[np.ndarray] = []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y_np = y.numpy()
        logits = model(x)
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(y_np)
        if keep_logits:
            logits_out.append(logits.cpu().numpy())
    return (
        np.concatenate(preds),
        np.concatenate(targets),
        np.concatenate(logits_out) if keep_logits else None,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(
    preds: np.ndarray,
    targets: np.ndarray,
    label_to_name: dict[int, str],
    logits: np.ndarray | None = None,
) -> EvalReport:
    num_classes = len(label_to_name)
    labels_range = list(range(num_classes))

    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(
        targets, preds, labels=labels_range, average="macro", zero_division=0
    )
    p_mi, r_mi, f_mi, _ = precision_recall_fscore_support(
        targets, preds, labels=labels_range, average="micro", zero_division=0
    )
    _, _, f_w, _ = precision_recall_fscore_support(
        targets, preds, labels=labels_range, average="weighted", zero_division=0
    )
    p_pc, r_pc, f_pc, s_pc = precision_recall_fscore_support(
        targets, preds, labels=labels_range, average=None, zero_division=0
    )

    per_class = (
        pd.DataFrame({
            "label": labels_range,
            "name": [label_to_name[i] for i in labels_range],
            "precision": p_pc, "recall": r_pc, "f1": f_pc, "support": s_pc,
        })
        .sort_values(["f1", "support"], ascending=[False, False])
        .reset_index(drop=True)
    )

    cm = confusion_matrix(targets, preds, labels=labels_range)

    return EvalReport(
        precision_macro=p_ma, recall_macro=r_ma, f1_macro=f_ma,
        precision_micro=p_mi, recall_micro=r_mi, f1_micro=f_mi,
        f1_weighted=f_w,
        per_class=per_class, confusion=cm, logits=logits,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_report(report: EvalReport, run_name: str, label_to_name: dict[int, str]) -> dict[str, Path]:
    """Write per-class CSV, summary text, and confusion matrix PNG. Returns paths."""
    out: dict[str, Path] = {}

    per_class_path = LOGS_DIR / f"{run_name}_per_class.csv"
    report.per_class.to_csv(per_class_path, index=False)
    out["per_class"] = per_class_path

    summary_path = LOGS_DIR / f"{run_name}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Precision (macro / micro) : {report.precision_macro:.4f} / {report.precision_micro:.4f}\n")
        f.write(f"Recall    (macro / micro) : {report.recall_macro:.4f} / {report.recall_micro:.4f}\n")
        f.write(f"F1        (macro / micro / weighted) : "
                f"{report.f1_macro:.4f} / {report.f1_micro:.4f} / {report.f1_weighted:.4f}\n")
        f.write("\n(see per_class CSV for per-class precision / recall / F1 / support)\n")
    out["summary"] = summary_path

    cm_path = FIGURES_DIR / f"{run_name}_confusion.png"
    _plot_confusion(report.confusion, label_to_name, cm_path, title=f"{run_name} — confusion (normalized)")
    out["confusion"] = cm_path

    return out


def _plot_confusion(cm: np.ndarray, label_to_name: dict[int, str], out: Path, title: str) -> None:
    n = cm.shape[0]
    with np.errstate(all="ignore"):
        cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    names = [label_to_name[i] for i in range(n)]
    fig, ax = plt.subplots(figsize=(max(8, n * 0.35), max(7, n * 0.35)))
    im = ax.imshow(cm_norm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
