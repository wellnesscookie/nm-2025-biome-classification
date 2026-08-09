# nm-2025-biome-classification

Multi-class classification of Minecraft biome screenshots (41 classes, ~22 K images).
Baseline: 3-conv CNN from scratch. Main model: fine-tuned EfficientNet-B0.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Reproduce

```bash
python3 -m scripts.eda --check-images
```

```bash
python3 -m scripts.build_split --seed 42
```

```bash
python3 -m scripts.train_baseline --epochs 30 --batch-size 64
```

```bash
python3 -m scripts.train_efficientnet --epochs 12 --head-epochs 3 --batch-size 32
```

```bash
python3 -m scripts.evaluate --arch baseline    --ckpt artifacts/checkpoints/baseline_v2_best.pt --run-name baseline_v2_test
python3 -m scripts.evaluate --arch efficientnet --ckpt artifacts/checkpoints/efficientnet_v2_best.pt --run-name efficientnet_v2_test
```

Outputs go to `artifacts/` (checkpoints, per-epoch logs, per-class CSV,
confusion-matrix PNGs). Everything is deterministic given `--seed`.

To render every completed run as one markdown table for the report:

```bash
python3 -m scripts.summarize_results
```

## Class imbalance (v2 defaults)

The dataset is heavily imbalanced (largest class ≈ 500× the smallest). The
v1 pipeline stacked a `WeightedRandomSampler` on top of a fully
inverse-frequency class-weighted CE, which over-corrected — see
[docs/RESULTS.md](docs/RESULTS.md) for the diagnosis. v2 defaults instead
to:

- weighted sampler **off** (opt back in with `--use-weighted-sampler`);
- class weights **tempered** (`--class-weight-power 0.5`, sqrt-inverse
  frequency — rare vs largest amplification drops from ≈420× to ≈20×);
- label smoothing on (`0.05` baseline, `0.10` EfficientNet);
- baseline uses stronger augmentation (`RandomResizedCrop`,
  `RandomErasing`); disable with `--no-strong-aug`.

## Results

Every completed experiment is logged in [docs/RESULTS.md](docs/RESULTS.md)
with config diff, headline metrics, and a short takeaway. The raw
per-run artifacts (summary txt, per-class CSV, training curve CSV,
confusion PNG) live under [`docs/runs/<version>/`](docs/runs/) and **are
committed** — small files, worth having next to the code they came from.

## Layout

- `src/` — library code (dataset, models, training loop, evaluation)
- `scripts/` — CLI entry points (EDA, split builder, train, evaluate,
  results summariser)
- `splits/split.csv` — persisted 70/15/15 stratified split; committed so
  every experiment is comparable
- `artifacts/` — live training outputs (gitignored)
- `docs/` — results log + committed snapshots of each run
- `data/` — raw dataset (gitignored)
