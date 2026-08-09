"""
Output: splits/split.csv with columns: path, biome_id, biome_name, label, split (train | val | test)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_class_index  # noqa: E402
from src.utils import DATA_DIR, SPLITS_DIR, get_logger, seed_everything  # noqa: E402

log = get_logger("build_split")


IMG_EXTS = {".jpg", ".jpeg", ".png"}


def scan_images(data_dir: Path) -> pd.DataFrame:
    """Enumerate every image under `data/biome_*/` into a DataFrame."""
    rows: list[dict] = []
    for folder in sorted(data_dir.glob("biome_*")):
        if not folder.is_dir():
            continue
        try:
            biome_id = int(folder.name.split("_", 1)[1])
        except (IndexError, ValueError):
            log.warning("Skipping unexpected folder name: %s", folder.name)
            continue
        for img in folder.iterdir():
            if img.suffix.lower() in IMG_EXTS and img.is_file():
                rows.append({"path": str(img.resolve()), "biome_id": biome_id})
    if not rows:
        raise RuntimeError(f"No images found under {data_dir}")
    return pd.DataFrame(rows)


def stratified_split(
    df: pd.DataFrame,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> pd.DataFrame:
    """Two-stage stratified split -> a 'split' column ('train'/'val'/'test').

    Requires every class to have at least 2 samples (train_test_split needs
    ≥2 per class for stratification).
    """
    counts = df["label"].value_counts()
    tiny = counts[counts < 2]
    if len(tiny) > 0:
        raise ValueError(
            f"{len(tiny)} class(es) have <2 samples and cannot be stratified: "
            f"{tiny.to_dict()}. Either drop them or add data."
        )

    # First: hold out (val + test) from train.
    holdout_frac = val_frac + test_frac
    train_df, holdout_df = train_test_split(
        df, test_size=holdout_frac, stratify=df["label"], random_state=seed
    )
    # Then: split holdout into val and test, keeping the requested ratio.
    rel_test = test_frac / holdout_frac
    val_df, test_df = train_test_split(
        holdout_df, test_size=rel_test, stratify=holdout_df["label"], random_state=seed
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    out = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return out.sort_values(["split", "label", "path"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=SPLITS_DIR / "split.csv")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--min-per-class",
        type=int,
        default=0,
        help="Drop classes with fewer than this many images before splitting.",
    )
    args = ap.parse_args()

    seed_everything(args.seed)

    log.info("Scanning images under %s ...", args.data_dir)
    df = scan_images(args.data_dir)
    log.info("Found %d images across %d biome IDs", len(df), df["biome_id"].nunique())

    cidx = build_class_index(args.data_dir)
    log.info("Discovered %d classes (label indices 0..%d)",
             cidx.num_classes, cidx.num_classes - 1)

    df["biome_name"] = df["biome_id"].map({bid: cidx.label_to_name[cidx.id_to_label[bid]]
                                            for bid in df["biome_id"].unique()})
    df["label"] = df["biome_id"].map(cidx.id_to_label)

    if args.min_per_class > 0:
        keep = df.groupby("label").size()
        keep = keep[keep >= args.min_per_class].index
        before = len(df)
        df = df[df["label"].isin(keep)].reset_index(drop=True)
        log.info("Dropped %d images from %d small classes (< %d imgs)",
                 before - len(df), cidx.num_classes - len(keep), args.min_per_class)

    log.info("Splitting %d imgs into 70/%.0f%%/%.0f%% (val/test) ...",
             len(df), args.val_frac * 100, args.test_frac * 100)
    out = stratified_split(df, args.val_frac, args.test_frac, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    # Report
    log.info("Wrote %s", args.out)
    for split, sub in out.groupby("split"):
        log.info("  %-5s %6d imgs  (%d classes)", split, len(sub), sub["label"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
