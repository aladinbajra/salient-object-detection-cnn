"""Train the saliency CNN."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from data_loader import create_sod_dataloaders, describe_datasets, seed_everything
from evaluate import (
    MetricAverager,
    bce_iou_loss,
    compute_batch_metrics,
    get_device,
    safe_torch_load,
    save_prediction_grid,
)
from sod_model import build_model, count_trainable_parameters
from sod_model import MODEL_VARIANTS

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def run_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float,
    iou_weight: float,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    meter = MetricAverager()
    split = "train" if training else "val"

    for batch in tqdm(dataloader, desc=f"epoch {epoch} {split}", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            predictions = model(images)
            loss, parts = bce_iou_loss(predictions, masks, iou_weight=iou_weight)
            if training:
                loss.backward()
                optimizer.step()

        metrics = compute_batch_metrics(predictions.detach(), masks, threshold=threshold)
        metrics.update({"loss": loss.item(), "bce": parts["bce"].item(), "soft_iou": parts["soft_iou"].item()})
        meter.update(metrics, batch_size=images.size(0))

    return meter.compute()


def flatten_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def write_csv(rows: list[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def atomic_torch_save(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    stale_epochs: int,
    history: list[Dict[str, float]],
    config: Dict[str, object],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Dict[str, object]:
    payload = {
        "epoch": epoch,
        "variant": config["variant"],
        "best_val_loss": best_val_loss,
        "stale_epochs": stale_epochs,
        "config": config,
        "history": history,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    return payload


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    stale_epochs: int,
    history: list[Dict[str, float]],
    config: Dict[str, object],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> None:
    atomic_torch_save(
        checkpoint_payload(model, optimizer, epoch, best_val_loss, stale_epochs, history, config, scheduler),
        path,
    )
    LOGGER.info("saved checkpoint: %s", path)


def _checkpoint_config(checkpoint: Dict[str, object]) -> Dict[str, object]:
    config = checkpoint.get("config") or checkpoint.get("args") or {}
    return dict(config) if isinstance(config, dict) else {}


def resume_training(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Dict[str, object],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> tuple[int, float, int, list[Dict[str, float]]]:
    if not checkpoint_path.exists():
        LOGGER.info("starting new run")
        return 1, float("inf"), 0, []

    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"invalid checkpoint format: {checkpoint_path}")

    saved_config = _checkpoint_config(checkpoint)
    saved_variant = checkpoint.get("variant") or saved_config.get("variant")
    if saved_variant and saved_variant != config["variant"]:
        raise ValueError(
            f"checkpoint variant is '{saved_variant}', but current variant is '{config['variant']}'. "
            "Use a different experiment name or pass --no-resume."
        )

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"checkpoint architecture does not match variant '{config['variant']}'. "
            "Use a different experiment name or pass --no-resume for a fresh run."
        ) from exc
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Older checkpoints in this repo stored epoch as zero-based.
    completed_epoch = int(checkpoint.get("epoch", 0))
    if "config" not in checkpoint:
        completed_epoch += 1

    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    stale_epochs = int(checkpoint.get("stale_epochs", checkpoint.get("epochs_without_improvement", 0)))
    history = list(checkpoint.get("history", []))

    LOGGER.info("resumed from epoch %s", completed_epoch)
    return completed_epoch + 1, best_val_loss, stale_epochs, history


def log_epoch(epoch: int, train: Dict[str, float], val: Dict[str, float], best_val_loss: float) -> None:
    LOGGER.info(
        "epoch %s | train loss %.4f iou %.4f f1 %.4f | val loss %.4f iou %.4f f1 %.4f | best %.4f",
        epoch,
        train["loss"],
        train["iou"],
        train["f1"],
        val["loss"],
        val["iou"],
        val["f1"],
        best_val_loss,
    )


def make_scheduler(name: str, optimizer: torch.optim.Optimizer, epochs: int):
    if name == "none":
        return None
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    raise ValueError(f"unknown scheduler '{name}'")


def is_better(metric_name: str, current: float, best: float, min_delta: float) -> bool:
    if metric_name == "val_loss":
        return current < best - min_delta
    return current > best + min_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a from-scratch saliency detector.")
    parser.add_argument("--data-root", default="data/DUTS")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--iou-weight", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--variant", choices=sorted(MODEL_VARIANTS), default="baseline")
    parser.add_argument("--scheduler", choices=["none", "plateau", "cosine"], default="none")
    parser.add_argument("--best-metric", choices=["val_loss", "val_iou", "val_f1"], default="val_loss")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--checkpoint-root", default="outputs/checkpoints")
    parser.add_argument("--log-root", default="outputs/logs")
    parser.add_argument("--visual-root", default="outputs/visualizations")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["random", "official"], default="random")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.experiment_name = args.experiment_name or args.variant
    return args


def main() -> None:
    configure_logging()
    args = parse_args()
    config = vars(args).copy()
    seed_everything(args.seed)

    checkpoint_dir = Path(args.checkpoint_root) / args.experiment_name
    visual_dir = Path(args.visual_root) / args.experiment_name
    history_path = Path(args.log_root) / f"{args.experiment_name}_history.csv"
    last_checkpoint = checkpoint_dir / "last_checkpoint.pth"
    best_checkpoint = checkpoint_dir / "best_model.pth"

    loaders, datasets = create_sod_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        split_mode=args.split_mode,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_samples=args.max_samples,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
    )

    device = get_device(args.device)
    model = build_model(args.variant).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = make_scheduler(args.scheduler, optimizer, args.epochs)

    write_json(config, checkpoint_dir / "config.json")
    write_json(
        {
            "data_root": args.data_root,
            "split_mode": args.split_mode,
            "seed": args.seed,
            "counts": {name: len(dataset) for name, dataset in datasets.items()},
        },
        checkpoint_dir / "dataset.json",
    )

    LOGGER.info("device: %s", device)
    LOGGER.info("dataset: %s", describe_datasets(datasets))
    LOGGER.info("variant: %s", args.variant)
    LOGGER.info("parameters: %s", f"{count_trainable_parameters(model):,}")

    best_score = float("inf") if args.best_metric == "val_loss" else -float("inf")
    start_epoch, best_val_loss, stale_epochs, history = (1, float("inf"), 0, [])
    if args.resume:
        start_epoch, best_val_loss, stale_epochs, history = resume_training(
            last_checkpoint, model, optimizer, device, config, scheduler
        )
        if history:
            last_best = min(row["val_loss"] for row in history)
            best_val_loss = min(best_val_loss, last_best)
            if args.best_metric == "val_loss":
                best_score = best_val_loss
            else:
                best_score = max(row[args.best_metric] for row in history)

    if start_epoch > args.epochs:
        LOGGER.info("training already complete through epoch %s", start_epoch - 1)
    else:
        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                loaders["train"],
                device,
                args.threshold,
                args.iou_weight,
                epoch,
                optimizer,
            )
            val_metrics = run_epoch(model, loaders["val"], device, args.threshold, args.iou_weight, epoch)

            current_score = val_metrics["loss"]
            if args.best_metric == "val_iou":
                current_score = val_metrics["iou"]
            elif args.best_metric == "val_f1":
                current_score = val_metrics["f1"]

            improved = is_better(args.best_metric, current_score, best_score, args.min_delta)
            if improved:
                best_score = current_score
                best_val_loss = val_metrics["loss"]
                stale_epochs = 0
            else:
                stale_epochs += 1

            row: Dict[str, float] = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
            row.update(flatten_metrics("train", train_metrics))
            row.update(flatten_metrics("val", val_metrics))
            history.append(row)
            write_csv(history, history_path)

            if scheduler is not None:
                if args.scheduler == "plateau":
                    scheduler.step(val_metrics["loss"])
                else:
                    scheduler.step()

            save_checkpoint(
                last_checkpoint,
                model,
                optimizer,
                epoch,
                best_val_loss,
                stale_epochs,
                history,
                config,
                scheduler,
            )
            if improved:
                save_checkpoint(
                    best_checkpoint,
                    model,
                    optimizer,
                    epoch,
                    best_val_loss,
                    stale_epochs,
                    history,
                    config,
                    scheduler,
                )

            log_epoch(epoch, train_metrics, val_metrics, best_val_loss)
            if stale_epochs >= args.patience:
                LOGGER.info("early stopping")
                break

    if best_checkpoint.exists() and len(datasets["val"]) > 0:
        checkpoint = safe_torch_load(best_checkpoint, map_location=device)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint["model_state_dict"])
        save_prediction_grid(model, loaders["val"], device, visual_dir / "validation_predictions.png", args.threshold)
        LOGGER.info("saved visualization: %s", visual_dir / "validation_predictions.png")

    LOGGER.info("best checkpoint: %s", best_checkpoint)
    LOGGER.info("history: %s", history_path)


if __name__ == "__main__":
    main()
