"""Paired image/mask loading for saliency detection."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR
    RESAMPLE_NEAREST = Image.NEAREST


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class SODSample:
    image_path: Path
    mask_path: Path


def expected_duts_layout(root: Path) -> str:
    candidates = [
        root / "DUTS-TR" / "DUTS-TR" / "DUTS-TR-Image",
        root / "DUTS-TR" / "DUTS-TR" / "DUTS-TR-Mask",
        root / "DUTS-TE" / "DUTS-TE" / "DUTS-TE-Image",
        root / "DUTS-TE" / "DUTS-TE" / "DUTS-TE-Mask",
    ]
    return "expected DUTS folders:\n" + "\n".join(f"- {path}" for path in candidates)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _files_by_stem(directory: Path, extensions: set[str]) -> Dict[str, Path]:
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    }


def _mask_dir_for(image_dir: Path) -> Optional[Path]:
    candidates = []
    if "Image" in image_dir.name:
        candidates.append(image_dir.with_name(image_dir.name.replace("Image", "Mask")))
    candidates.extend(path for path in image_dir.parent.iterdir() if path.is_dir() and "mask" in path.name.lower())

    seen = set()
    for candidate in candidates:
        key = candidate.resolve()
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_dir():
            return candidate
    return None


def discover_sod_pairs(data_root: str | Path = "data/DUTS", split_hint: Optional[str] = None) -> list[SODSample]:
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}\n{expected_duts_layout(root)}")

    split_hint = split_hint.lower() if split_hint else None
    image_dirs = [path for path in root.rglob("*-Image") if path.is_dir()]
    image_dirs += [path for path in root.rglob("*_Image") if path.is_dir()]
    if not image_dirs:
        raise RuntimeError(f"no image folders found under: {root}\n{expected_duts_layout(root)}")

    samples = {}
    for image_dir in image_dirs:
        if split_hint and split_hint not in str(image_dir).lower():
            continue

        mask_dir = _mask_dir_for(image_dir)
        if mask_dir is None:
            continue

        images = _files_by_stem(image_dir, IMAGE_EXTENSIONS)
        masks = _files_by_stem(mask_dir, MASK_EXTENSIONS)
        for stem, image_path in images.items():
            if stem in masks:
                samples[str(image_path.resolve())] = SODSample(image_path=image_path, mask_path=masks[stem])

    if not samples:
        raise RuntimeError(f"no paired image/mask files found under: {root}\n{expected_duts_layout(root)}")

    return sorted(samples.values(), key=lambda sample: str(sample.image_path))


def split_samples(
    samples: Sequence[SODSample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Dict[str, list[SODSample]]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("split ratios must sum to 1.0")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    if max_samples is not None:
        shuffled = shuffled[:max_samples]

    n_total = len(shuffled)
    requested_splits = sum(ratio > 0 for ratio in (train_ratio, val_ratio, test_ratio))
    if n_total < requested_splits:
        raise ValueError(f"need at least {requested_splits} samples for the requested split ratios")

    ratios = [train_ratio, val_ratio, test_ratio]
    counts = [int(n_total * ratio) for ratio in ratios]
    positive = [index for index, ratio in enumerate(ratios) if ratio > 0]

    remainder = n_total - sum(counts)
    fractions = [n_total * ratio - int(n_total * ratio) for ratio in ratios]
    for index in sorted(positive, key=lambda i: (fractions[i], ratios[i]), reverse=True):
        if remainder == 0:
            break
        counts[index] += 1
        remainder -= 1

    for index, ratio in enumerate(ratios):
        if ratio > 0 and counts[index] == 0:
            counts[index] = 1

    while sum(counts) > n_total:
        reducible = [i for i in positive if counts[i] > 1]
        counts[max(reducible, key=lambda i: counts[i])] -= 1

    n_train, n_val, _ = counts
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _read_image(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path).convert("RGB"))


def _read_mask(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def _resize_pair(image: Image.Image, mask: Image.Image, size: int, augment: bool) -> Tuple[Image.Image, Image.Image]:
    if not augment:
        return (
            image.resize((size, size), RESAMPLE_BILINEAR),
            mask.resize((size, size), RESAMPLE_NEAREST),
        )

    margin = max(8, size // 8)
    larger = size + margin
    image = image.resize((larger, larger), RESAMPLE_BILINEAR)
    mask = mask.resize((larger, larger), RESAMPLE_NEAREST)
    left = random.randint(0, margin)
    top = random.randint(0, margin)
    box = (left, top, left + size, top + size)
    return image.crop(box), mask.crop(box)


def _augment_pair(image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
        mask = ImageOps.mirror(mask)
    if random.random() < 0.7:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
    if random.random() < 0.4:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
    return image, mask


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = ImageOps.exif_transpose(image.convert("RGB"))
    image = image.resize((image_size, image_size), RESAMPLE_BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def mask_to_tensor(mask: Image.Image, image_size: int, threshold: int = 128) -> torch.Tensor:
    mask = mask.convert("L").resize((image_size, image_size), RESAMPLE_NEAREST)
    array = (np.asarray(mask, dtype=np.uint8) >= threshold).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0).contiguous()


class SaliencyDataset(Dataset):
    def __init__(self, samples: Sequence[SODSample], image_size: int, augment: bool):
        self.samples = list(samples)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        image = _read_image(sample.image_path)
        mask = _read_mask(sample.mask_path)
        image, mask = _resize_pair(image, mask, self.image_size, self.augment)
        if self.augment:
            image, mask = _augment_pair(image, mask)

        return {
            "image": image_to_tensor(image, self.image_size),
            "mask": mask_to_tensor(mask, self.image_size),
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
            "sample_id": sample.image_path.stem,
        }


def _official_split(data_root: str | Path, seed: int, max_samples: Optional[int]) -> Dict[str, list[SODSample]]:
    train_val = discover_sod_pairs(data_root, split_hint="DUTS-TR")
    test = discover_sod_pairs(data_root, split_hint="DUTS-TE")

    if max_samples is not None:
        return split_samples(train_val + test, seed=seed, max_samples=max_samples)

    split = split_samples(train_val, train_ratio=0.85, val_ratio=0.15, test_ratio=0.0, seed=seed)
    return {"train": split["train"], "val": split["val"], "test": test}


def create_sod_dataloaders(
    data_root: str | Path = "data/DUTS",
    image_size: int = 128,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
    split_mode: str = "random",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_samples: Optional[int] = None,
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
) -> Tuple[Dict[str, DataLoader], Dict[str, SaliencyDataset]]:
    seed_everything(seed)

    if split_mode == "random":
        splits = split_samples(
            discover_sod_pairs(data_root),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            max_samples=max_samples,
        )
    elif split_mode == "official":
        splits = _official_split(data_root, seed=seed, max_samples=max_samples)
    else:
        raise ValueError("split_mode must be 'random' or 'official'")

    for name, limit in (("train", train_limit), ("val", val_limit), ("test", test_limit)):
        if limit is not None:
            if limit < 1:
                raise ValueError(f"{name}_limit must be positive")
            splits[name] = splits[name][:limit]

    datasets = {
        "train": SaliencyDataset(splits["train"], image_size=image_size, augment=True),
        "val": SaliencyDataset(splits["val"], image_size=image_size, augment=False),
        "test": SaliencyDataset(splits["test"], image_size=image_size, augment=False),
    }

    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            **loader_kwargs,
        ),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, **loader_kwargs),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, **loader_kwargs),
    }
    return loaders, datasets


def describe_datasets(datasets: Dict[str, SaliencyDataset]) -> str:
    return " | ".join(f"{name}: {len(dataset)}" for name, dataset in datasets.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect paired SOD data.")
    parser.add_argument("--data-root", default="data/DUTS")
    parser.add_argument("--split-mode", choices=["random", "official"], default="random")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = discover_sod_pairs(args.data_root)
    _, datasets = create_sod_dataloaders(
        data_root=args.data_root,
        split_mode=args.split_mode,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print(f"paired samples: {len(pairs)}")
    print(describe_datasets(datasets))


if __name__ == "__main__":
    main()
