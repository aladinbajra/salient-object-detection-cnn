from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from evaluate import get_device, load_model_from_checkpoint, make_overlay_np, predict_pil_image
from sod_model import MODEL_VARIANTS


def default_checkpoint() -> str:
    for path in (
        "outputs/checkpoints/exp03_improved_v3_128_final/best_model.pth",
        "outputs/checkpoints/exp04_improved_v3_224/best_model.pth",
        "outputs/checkpoints/exp02_improved_v2_224/best_model.pth",
        "outputs/checkpoints/exp01_baseline_128/best_model.pth",
    ):
        if Path(path).exists():
            return path
    return "outputs/checkpoints/exp01_baseline_128/best_model.pth"


def default_threshold(checkpoint_path: str) -> float:
    experiment = Path(checkpoint_path).parent.name
    metrics_path = Path("outputs") / f"evaluation_{experiment}" / "metrics.json"
    if not metrics_path.exists():
        return 0.5
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.5
    return float(metrics.get("selected_threshold", 0.5))


@st.cache_resource(show_spinner=False)
def load_cached_model(checkpoint_path: str, device_name: str, variant: str):
    device = get_device(device_name)
    selected_variant = None if variant == "auto" else variant
    return load_model_from_checkpoint(checkpoint_path, device, selected_variant), device


def image_np(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def main() -> None:
    st.set_page_config(page_title="SOD CNN Demo", layout="wide")
    st.title("Salient Object Detection")

    with st.sidebar:
        checkpoint_path = st.text_input("Checkpoint path", default_checkpoint())
        variant = st.selectbox("Model variant", ["auto"] + sorted(MODEL_VARIANTS), index=0)
        image_size = st.selectbox("Input size", [128, 224], index=0)
        threshold = st.slider("Mask threshold", 0.05, 0.95, default_threshold(checkpoint_path), 0.05)
        devices = ["auto", "cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        device_name = st.selectbox("Device", devices, index=0)

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        st.warning(f"Checkpoint not found: {checkpoint}")
        st.code(
            "python train.py --data-root data/DUTS --image-size 128 --batch-size 16 --epochs 25 "
            "--variant improved_v3 --experiment-name exp03_improved_v3_128_final",
            language="bash",
        )
        return

    upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if upload is None:
        st.info("Upload an image to run inference.")
        return

    try:
        image = Image.open(upload).convert("RGB")
    except Exception:
        st.error("Could not read the uploaded image. Use a valid JPG, PNG, BMP, or WEBP file.")
        return

    try:
        model, device = load_cached_model(str(checkpoint), device_name, variant)
        mask, elapsed_ms = predict_pil_image(model, image, device, image_size)
    except Exception as exc:
        st.error(str(exc))
        return

    binary_mask = (mask >= threshold).astype(np.float32)
    overlay = make_overlay_np(image_np(image), mask)

    st.caption(f"Inference time: {elapsed_ms:.2f} ms per image")
    col1, col2, col3 = st.columns(3)
    col1.image(image, caption="Input image", use_container_width=True)
    col2.image(binary_mask, caption="Saliency mask", clamp=True, use_container_width=True)
    col3.image(overlay, caption="Overlay", clamp=True, use_container_width=True)


if __name__ == "__main__":
    main()
