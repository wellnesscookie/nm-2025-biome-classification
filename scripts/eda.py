from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_class_index, discover_biome_ids, load_id_to_name  # noqa: E402
from src.utils import DATA_DIR, FIGURES_DIR, LOGS_DIR, get_logger  # noqa: E402

log = get_logger("eda")

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def count_per_folder(data_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for folder in sorted(data_dir.glob("biome_*")):
        if not folder.is_dir():
            continue
        try:
            bid = int(folder.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        n = sum(1 for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
        rows.append({"biome_id": bid, "folder": folder.name, "n_images": n})
    return pd.DataFrame(rows).sort_values("n_images", ascending=False).reset_index(drop=True)


def verify_images(data_dir: Path) -> list[str]:
    """Try to open every image; return paths that failed."""
    bad: list[str] = []
    files = [p for p in data_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    for p in tqdm(files, desc="verify"):
        try:
            with Image.open(p) as im:
                im.verify()
        except (UnidentifiedImageError, OSError):
            bad.append(str(p))
    return bad


def check_spec_mismatches(cidx) -> list[str]:
    """Warn about biome names in the spec that don't resolve on disk."""
    # From specifikacija.txt (verbatim).
    spec_names = {
        "plains", "forest", "mountains", "taiga", "desert", "snowy_tundra",
        "savanna", "birch_forest", "dark_forest", "wooded_hills", "swamp",
        "wooded_mountains", "snowy_mountains", "desert_hills", "beach",
        "river", "taiga_hills", "snowy_taiga", "jungle", "birch_forest_hills",
        "giant_tree_taiga", "sunflower_plains", "savanna_plateau",
        "jungle_hills", "flower_forest", "snowy_beach", "badlands",
        "gravelly_mountains_plus", "frozen_ocean", "snowy_taiga_hills",
        "giant_tree_taiga_hills", "desert_lakes", "gravelly_mountains",
        "dark_forest_hills", "modified_badlands_plateau", "frozen_river",
        "badlands_plateau", "taiga_mountains", "tall_birch_hills",
        "lukewarm_ocean", "snowy_taiga_mountains",
    }
    on_disk = set(cidx.names)
    only_in_spec = spec_names - on_disk
    only_on_disk = on_disk - spec_names
    warnings: list[str] = []
    if only_in_spec:
        warnings.append(f"In spec but not on disk: {sorted(only_in_spec)}")
    if only_on_disk:
        warnings.append(f"On disk but not in spec: {sorted(only_on_disk)}")
    return warnings


def plot_distribution(df: pd.DataFrame, id_to_name: dict[int, str], out: Path) -> None:
    df = df.copy()
    df["name"] = df["biome_id"].map(lambda b: id_to_name.get(b, f"unknown_{b}"))
    df = df.sort_values("n_images", ascending=True)  # horizontal bar, smallest at bottom

    fig, ax = plt.subplots(figsize=(9, max(6, 0.22 * len(df))))
    ax.barh(df["name"], df["n_images"])
    ax.set_xlabel("images")
    ax.set_title(f"Images per biome ({len(df)} classes, total = {int(df['n_images'].sum()):,})")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--check-images", action="store_true",
                    help="Also verify every JPEG opens (slow).")
    args = ap.parse_args()

    log.info("Scanning %s", args.data_dir)
    counts = count_per_folder(args.data_dir)

    id_to_name_all = load_id_to_name()
    cidx = build_class_index(args.data_dir)
    counts["biome_name"] = counts["biome_id"].map(
        lambda b: id_to_name_all.get(b, f"unknown_{b}")
    )
    counts["label"] = counts["biome_id"].map(cidx.id_to_label)
    counts = counts[["label", "biome_id", "biome_name", "folder", "n_images"]]

    # Report.
    total = int(counts["n_images"].sum())
    n_classes = len(counts)
    log.info("Total images: %d", total)
    log.info("Classes: %d  (min=%d, median=%d, max=%d, mean=%.1f)",
             n_classes,
             int(counts["n_images"].min()),
             int(counts["n_images"].median()),
             int(counts["n_images"].max()),
             counts["n_images"].mean())
    imbalance = counts["n_images"].max() / max(counts["n_images"].min(), 1)
    log.info("Imbalance ratio (max/min): %.1fx", imbalance)

    log.info("Discovered biome IDs on disk: %s", discover_biome_ids(args.data_dir))

    for msg in check_spec_mismatches(cidx):
        log.warning(msg)

    csv_out = LOGS_DIR / "class_counts.csv"
    counts.to_csv(csv_out, index=False)
    log.info("Wrote %s", csv_out)

    png_out = FIGURES_DIR / "class_distribution.png"
    plot_distribution(counts, id_to_name_all, png_out)
    log.info("Wrote %s", png_out)

    if args.check_images:
        log.info("Verifying image integrity (this can take a while) ...")
        bad = verify_images(args.data_dir)
        if bad:
            log.warning("Found %d unreadable image(s)", len(bad))
            path = LOGS_DIR / "corrupt_images.txt"
            path.write_text("\n".join(bad), encoding="utf-8")
            log.info("Wrote %s", path)
        else:
            log.info("All images opened successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
