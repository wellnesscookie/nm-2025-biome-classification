from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import LOGS_DIR


_SUMMARY_METRICS = [
    ("prec_macro",     re.compile(r"^Precision \(macro / micro\)\s*:\s*([\d.]+)")),
    ("recall_macro",   re.compile(r"^Recall\s+\(macro / micro\)\s*:\s*([\d.]+)")),
    ("f1_macro",       re.compile(r"^F1\s+\(macro / micro / weighted\)\s*:\s*([\d.]+)")),
    ("f1_weighted",    re.compile(r"^F1\s+\(macro / micro / weighted\)\s*:.*/\s*([\d.]+)\s*$")),
]


def parse_summary(path: Path) -> dict[str, float]:
    """Extract numeric metrics from a `<run>_summary.txt` file."""
    out: dict[str, float] = {}
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        for key, rx in _SUMMARY_METRICS:
            m = rx.search(line)
            if m:
                out[key] = float(m.group(1))
    return out


def best_epoch_from_csv(csv_path: Path) -> tuple[int, float] | None:
    """Return (epoch, val_f1_macro) for the best epoch, or None if missing."""
    if not csv_path.exists():
        return None
    best_ep, best_f1 = -1, -1.0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                f1 = float(row["val_f1_macro"])
                ep = int(row["epoch"])
            except (KeyError, ValueError):
                continue
            if f1 > best_f1:
                best_f1, best_ep = f1, ep
    return (best_ep, best_f1) if best_ep >= 0 else None


def render_table(rows: list[dict]) -> str:
    header = ("| Run | F1 macro | F1 weighted | Prec macro | Recall macro "
              "| Best val F1 (ep) |")
    sep    = "|-----|:--------:|:-----------:|:----------:|:-----------:|:----------------:|"
    lines = [header, sep]
    for r in rows:
        best = r.get("best")
        best_cell = f"{best[1]:.4f} (ep {best[0]})" if best else "—"
        lines.append(
            f"| `{r['name']}` "
            f"| {r['metrics'].get('f1_macro',    float('nan')):.4f} "
            f"| {r['metrics'].get('f1_weighted', float('nan')):.4f} "
            f"| {r['metrics'].get('prec_macro',  float('nan')):.4f} "
            f"| {r['metrics'].get('recall_macro',float('nan')):.4f} "
            f"| {best_cell} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", type=Path, default=LOGS_DIR,
                    help="Directory to scan for *_summary.txt (default: artifacts/logs).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write the table to this file instead of stdout.")
    args = ap.parse_args()

    summaries = sorted(args.logs.glob("*_summary.txt"))
    if not summaries:
        print(f"No *_summary.txt files found under {args.logs}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for s in summaries:
        run_name = s.stem[:-len("_summary")] if s.stem.endswith("_summary") else s.stem
        train_stem = run_name.rsplit("_test", 1)[0].rsplit("_val", 1)[0]
        rows.append({
            "name": run_name,
            "metrics": parse_summary(s),
            "best": best_epoch_from_csv(args.logs / f"{train_stem}.csv"),
        })

    table = render_table(rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
