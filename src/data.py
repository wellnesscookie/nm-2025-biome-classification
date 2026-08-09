from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .utils import DATA_DIR, MAPPING_FILE


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

_MAPPING_RE = re.compile(r"BIOMES\s*=\s*(\{.*?\})", re.DOTALL)


def load_id_to_name(mapping_path: Path = MAPPING_FILE) -> dict[int, str]:
    """Parse `data/mapping` and return {biome_id: biome_name}.

    The file is a Python source snippet, but we parse it with `ast.literal_eval`
    on the extracted dict expression so we never `exec` untrusted code.
    """
    text = Path(mapping_path).read_text(encoding="utf-8")
    m = _MAPPING_RE.search(text)
    if not m:
        raise ValueError(f"Could not find `BIOMES = {{...}}` in {mapping_path}")
    mapping = ast.literal_eval(m.group(1))
    if not isinstance(mapping, dict):
        raise ValueError("Parsed mapping is not a dict")
    return {int(k): str(v) for k, v in mapping.items()}


_DIR_RE = re.compile(r"^biome_(\d+)$")


def discover_biome_ids(data_dir: Path = DATA_DIR) -> list[int]:
    """Return sorted list of biome IDs found as `data/biome_<id>/` folders."""
    ids: list[int] = []
    for p in Path(data_dir).iterdir():
        if not p.is_dir():
            continue
        m = _DIR_RE.match(p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


@dataclass(frozen=True)
class ClassIndex:
    """Immutable class index: biome id / name / label are all resolvable.

    `labels` is the canonical 0..N-1 index used by models. Ordering is
    alphabetical by name so labels are stable no matter where you run this.
    """

    ids: tuple[int, ...]          # e.g. (1, 2, 3, ...)  — dataset-order
    names: tuple[str, ...]         # e.g. ("badlands", "beach", ...) — label-order
    id_to_label: dict[int, int]    # biome_id -> label
    label_to_id: dict[int, int]    # label -> biome_id
    label_to_name: dict[int, str]  # label -> biome_name

    @property
    def num_classes(self) -> int:
        return len(self.names)


def build_class_index(
    data_dir: Path = DATA_DIR,
    mapping_path: Path = MAPPING_FILE,
) -> ClassIndex:
    """Merge disk discovery with the mapping file into a canonical label index.

    Used by the EDA and split-builder scripts, where the raw dataset is
    available. **Training / evaluation should call `class_index_from_split()`
    instead** so the class list is derived from the persisted split — that
    keeps the pipeline portable (e.g. on Kaggle the dataset lives at a
    different path than in the repo, but the split CSV is authoritative).

    Any folder ID missing from `mapping` falls back to the string form of the
    ID as its name and emits no error — that lets you continue while you
    reconcile the aliases mentioned in the spec.
    """
    id_to_name_all = load_id_to_name(mapping_path)
    ids_on_disk = discover_biome_ids(data_dir)

    id_to_name: dict[int, str] = {}
    for bid in ids_on_disk:
        name = id_to_name_all.get(bid, f"unknown_{bid}")
        id_to_name[bid] = name

    # Canonical label order: alphabetical by name (stable across machines).
    ordered = sorted(id_to_name.items(), key=lambda kv: kv[1])
    names = tuple(name for _, name in ordered)
    ids_in_label_order = tuple(bid for bid, _ in ordered)

    id_to_label = {bid: i for i, bid in enumerate(ids_in_label_order)}
    label_to_id = {i: bid for bid, i in id_to_label.items()}
    label_to_name = {i: name for i, name in enumerate(names)}

    return ClassIndex(
        ids=tuple(ids_on_disk),
        names=names,
        id_to_label=id_to_label,
        label_to_id=label_to_id,
        label_to_name=label_to_name,
    )


def class_index_from_split(split_csv: Path) -> ClassIndex:
    """Build a ClassIndex from a persisted split CSV.

    The CSV is authoritative: whatever labels/names were assigned when the
    split was created are the ones we use downstream. No filesystem scan, no
    dependence on the location of the raw dataset — the training and
    evaluation scripts should always use this helper.
    """
    df = pd.read_csv(split_csv, usecols=["label", "biome_id", "biome_name"])
    uniq = (
        df.drop_duplicates(subset=["label"])
          .sort_values("label")
          .reset_index(drop=True)
    )
    if uniq["label"].tolist() != list(range(len(uniq))):
        raise ValueError(
            f"Split CSV labels are not contiguous 0..N-1: got {uniq['label'].tolist()}"
        )
    ids_in_label_order = tuple(int(x) for x in uniq["biome_id"])
    names = tuple(str(x) for x in uniq["biome_name"])
    id_to_label = {bid: i for i, bid in enumerate(ids_in_label_order)}
    label_to_id = dict(enumerate(ids_in_label_order))
    label_to_name = dict(enumerate(names))
    return ClassIndex(
        ids=ids_in_label_order,
        names=names,
        id_to_label=id_to_label,
        label_to_id=label_to_id,
        label_to_name=label_to_name,
    )


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

# ImageNet stats — required for pretrained EfficientNet, harmless for baseline.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Native image size in the dataset (H, W).
NATIVE_H, NATIVE_W = 180, 320


def train_transform(
    image_size: tuple[int, int] | int = (NATIVE_H, NATIVE_W),
    *,
    strong: bool = False,
) -> T.Compose:
    """Augmentations for training. Biome cues are texture/color dominant, so
    aggressive geometric warping is counterproductive — but v1 was too mild
    (the baseline reached train_loss 0.48 vs val_loss 3.4 in 30 epochs).

    `strong=True` swaps `Resize`+small affine for `RandomResizedCrop` and adds
    `RandomErasing`; recommended for the from-scratch baseline where over-
    fitting is the bottleneck. `strong=False` preserves the v1 behavior and
    is a better fit for pretrained fine-tunes (EfficientNet), where you want
    the input distribution to stay close to ImageNet's.
    """
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    ops: list = [
        T.Resize(image_size) if not strong else T.RandomResizedCrop(
            image_size, scale=(0.6, 1.0), ratio=(1.5, 2.0)
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(
            brightness=0.25 if strong else 0.15,
            contrast=0.25 if strong else 0.15,
            saturation=0.25 if strong else 0.15,
            hue=0.03 if strong else 0.02,
        ),
    ]
    if not strong:
        ops.append(T.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)))
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if strong:
        # Random erasing after Normalize (torchvision convention). Modest area
        # so we don't blank out the biome silhouette entirely.
        ops.append(T.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0.0))
    return T.Compose(ops)


def eval_transform(image_size: tuple[int, int] | int = (NATIVE_H, NATIVE_W)) -> T.Compose:
    """Deterministic transform for val/test — resize + normalize only."""
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BiomeDataset(Dataset):
    """Reads (path, label) rows from a DataFrame produced by `build_split.py`.

    Expected columns: `path` (str, relative to project root or absolute) and
    `label` (int, 0..N-1).
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        transform: T.Compose | None = None,
        root: Path | None = None,
    ) -> None:
        required = {"path", "label"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"BiomeDataset frame missing columns: {missing}")
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.root = Path(root) if root else None

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        return (self.root / path) if self.root else path

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[idx]
        img = Image.open(self._resolve(row["path"])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])


# ---------------------------------------------------------------------------
# Class balancing helpers
# ---------------------------------------------------------------------------

def compute_class_weights(
    labels: Sequence[int],
    num_classes: int,
    *,
    power: float = 1.0,
) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)  # avoid /0 for absent classes
    inv = counts ** (-power)                     # power=0 -> all ones
    weights = inv * (num_classes / inv.sum())    # scale to mean 1
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(
    labels: Sequence[int],
    num_classes: int,
    *,
    power: float = 1.0,
) -> torch.Tensor:
    """Per-sample weights for `WeightedRandomSampler`.

    `power` behaves like `compute_class_weights` — 1.0 is pure inverse
    frequency, 0.5 is sqrt-tempered.
    """
    labels_arr = np.asarray(labels)
    counts = np.bincount(labels_arr, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    per_class = counts ** (-power)
    return torch.tensor(per_class[labels_arr], dtype=torch.float64)
