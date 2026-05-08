"""Evaluation and inference utilities."""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from data_loader import RESAMPLE_BILINEAR, create_sod_dataloaders, image_to_tensor
from sod_model import MODEL_VARIANTS, build_model, count_trainable_parameters

LOGGER = logging.getLogger(__name__)
DEFAULT_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def safe_torch_load(path: str | Path, map_location: torch.device) -> object:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def soft_iou_score(predictions: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    predictions = predictions.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)
    intersection = (predictions * targets).sum(dim=1)
    union = predictions.sum(dim=1) + targets.sum(dim=1) - intersection
    return ((intersection + eps) / (union + eps)).mean()


def bce_iou_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    iou_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    bce = F.binary_cross_entropy(predictions, targets)
    iou = soft_iou_score(predictions, targets)
    return bce + iou_weight * (1.0 - iou), {"bce": bce, "soft_iou": iou}


@torch.no_grad()
def compute_batch_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Dict[str, float]:
    predicted = (predictions >= threshold).float()
    target = (targets >= 0.5).float()

    dims = (1, 2, 3)
    tp = (predicted * target).sum(dim=dims)
    fp = (predicted * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - predicted) * target).sum(dim=dims)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    mae = torch.abs(predictions - targets).mean(dim=dims)

    return {
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
        "mae": mae.mean().item(),
    }


class MetricAverager:
    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.count = 0

    def update(self, metrics: Dict[str, float], batch_size: int) -> None:
        for name, value in metrics.items():
            self.totals[name] = self.totals.get(name, 0.0) + float(value) * batch_size
        self.count += batch_size

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {}
        return {name: total / self.count for name, total in self.totals.items()}


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        message = "CUDA was requested, but torch.cuda.is_available() is False. Use a GPU runtime or pass --device cpu."
        warnings.warn(message, RuntimeWarning)
        raise RuntimeError(message)
    return torch.device(name)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, object]],
    device: torch.device,
    threshold: float = 0.5,
    iou_weight: float = 0.5,
    max_batches: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, float]:
    model.eval()
    meter = MetricAverager()
    elapsed_total = 0.0
    image_count = 0

    batches = tqdm(dataloader, desc="evaluating", leave=False) if show_progress else dataloader
    for batch_index, batch in enumerate(batches):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        predictions = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        model_elapsed = time.perf_counter() - start

        loss, parts = bce_iou_loss(predictions, masks, iou_weight=iou_weight)
        metrics = compute_batch_metrics(predictions, masks, threshold=threshold)
        metrics.update({"loss": loss.item(), "bce": parts["bce"].item(), "soft_iou": parts["soft_iou"].item()})

        batch_size = images.size(0)
        meter.update(metrics, batch_size)
        elapsed_total += model_elapsed
        image_count += batch_size

    results = meter.compute()
    if image_count:
        results["inference_time_ms"] = 1000.0 * elapsed_total / image_count
    return results


def select_threshold_on_validation(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, object]],
    device: torch.device,
    thresholds: Iterable[float],
    iou_weight: float,
) -> Tuple[float, list[Dict[str, float]]]:
    rows = []
    for threshold in thresholds:
        metrics = evaluate_model(
            model,
            dataloader,
            device,
            threshold=threshold,
            iou_weight=iou_weight,
            show_progress=False,
        )
        row = {"threshold": float(threshold)}
        row.update(metrics)
        rows.append(row)

    if not rows:
        return 0.5, []

    best = max(rows, key=lambda row: (row.get("f1", 0.0), row.get("iou", 0.0), -abs(row["threshold"] - 0.5)))
    return float(best["threshold"]), rows


def tensor_to_image_np(tensor: torch.Tensor) -> np.ndarray:
    return np.clip(tensor.detach().cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)


def tensor_to_mask_np(tensor: torch.Tensor) -> np.ndarray:
    return np.clip(tensor.detach().cpu().squeeze().numpy(), 0.0, 1.0)


def make_overlay_np(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    mask = np.clip(mask, 0.0, 1.0)
    heatmap = np.zeros_like(image)
    heatmap[..., 0] = 1.0
    heatmap[..., 1] = 0.12
    heatmap[..., 2] = 0.04
    return np.clip(image * (1.0 - alpha * mask[..., None]) + heatmap * alpha * mask[..., None], 0.0, 1.0)


@torch.no_grad()
def save_prediction_grid(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, object]],
    device: torch.device,
    output_path: str | Path,
    threshold: float = 0.5,
    max_items: int = 4,
) -> None:
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch = next(iter(dataloader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    predictions = model(images)

    count = min(max_items, images.size(0))
    fig, axes = plt.subplots(count, 4, figsize=(12, 3 * count))
    axes = np.atleast_2d(axes)

    for col, title in enumerate(("Input", "Ground Truth", "Prediction", "Overlay")):
        axes[0, col].set_title(title)

    for row in range(count):
        image = tensor_to_image_np(images[row])
        ground_truth = tensor_to_mask_np(masks[row])
        prediction = tensor_to_mask_np(predictions[row])
        overlay = make_overlay_np(image, prediction)

        axes[row, 0].imshow(image)
        axes[row, 1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(prediction >= threshold, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(overlay)
        for col in range(4):
            axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _checkpoint_config(checkpoint: object) -> Dict[str, object]:
    if not isinstance(checkpoint, dict):
        return {}
    config = checkpoint.get("config") or checkpoint.get("args") or {}
    return dict(config) if isinstance(config, dict) else {}


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    variant: Optional[str] = None,
) -> torch.nn.Module:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    config = _checkpoint_config(checkpoint)
    checkpoint_variant = None
    if isinstance(checkpoint, dict):
        checkpoint_variant = checkpoint.get("variant")
    checkpoint_variant = checkpoint_variant or config.get("variant") or "baseline"

    selected_variant = variant or str(checkpoint_variant)
    if variant and checkpoint_variant and variant != checkpoint_variant:
        raise ValueError(f"checkpoint variant is '{checkpoint_variant}', but '{variant}' was requested")

    model = build_model(selected_variant).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"checkpoint architecture does not match variant '{selected_variant}'. "
            "Use the variant saved in the checkpoint or train a new checkpoint for this architecture."
        ) from exc
    model.eval()
    return model


@torch.no_grad()
def predict_pil_image(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    image_size: int,
) -> Tuple[np.ndarray, float]:
    original_size = image.size
    tensor = image_to_tensor(image, image_size=image_size).unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    prediction = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    mask = tensor_to_mask_np(prediction[0])
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_image = mask_image.resize(original_size, RESAMPLE_BILINEAR)
    return np.asarray(mask_image, dtype=np.float32) / 255.0, elapsed_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained saliency model.")
    parser.add_argument("--data-root", default="data/DUTS")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/exp03_improved_v3_128_final/best_model.pth")
    parser.add_argument("--variant", choices=sorted(MODEL_VARIANTS), default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-sweep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--iou-weight", type=float, default=0.5)
    parser.add_argument("--split-mode", choices=["random", "official"], default="random")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visual-count", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders, datasets = create_sod_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_mode=args.split_mode,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    model = load_model_from_checkpoint(args.checkpoint, device=device, variant=args.variant)

    LOGGER.info("checkpoint: %s", args.checkpoint)
    LOGGER.info("device: %s", device)
    LOGGER.info("test samples: %s", len(datasets["test"]))
    LOGGER.info("parameters: %s", f"{count_trainable_parameters(model):,}")

    selected_threshold = args.threshold
    validation_threshold_metrics = []
    if args.threshold_sweep:
        LOGGER.info("selecting threshold on validation split: %s", ", ".join(f"{t:.2f}" for t in args.thresholds))
        selected_threshold, validation_threshold_metrics = select_threshold_on_validation(
            model,
            loaders["val"],
            device,
            args.thresholds,
            args.iou_weight,
        )
        LOGGER.info("selected threshold: %.2f", selected_threshold)

    metrics = evaluate_model(model, loaders["test"], device, selected_threshold, args.iou_weight)
    metrics["selected_threshold"] = selected_threshold
    metrics["threshold_source"] = "validation" if args.threshold_sweep else "manual"
    if validation_threshold_metrics:
        metrics["validation_threshold_metrics"] = validation_threshold_metrics

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info(json.dumps(metrics, indent=2))
    LOGGER.info("saved metrics: %s", metrics_path)

    if args.visual_count > 0 and len(datasets["test"]) > 0:
        grid_path = output_dir / "sample_predictions.png"
        save_prediction_grid(model, loaders["test"], device, grid_path, selected_threshold, args.visual_count)
        LOGGER.info("saved visualization: %s", grid_path)


if __name__ == "__main__":
    main()
