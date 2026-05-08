# Salient Object Detection CNN

From-scratch Salient Object Detection using PyTorch. The model predicts a one-channel saliency mask for an RGB image and can visualize the result as a binary mask or image overlay.

## Files

```text
data_loader.py              dataset discovery, preprocessing, split, augmentation
sod_model.py                baseline and improved CNN variants
train.py                    training loop, metrics, checkpoint save/resume
evaluate.py                 test metrics, prediction grids, inference helpers
app.py                      Streamlit upload demo
```

## Setup

```bash
pip install -r requirements.txt
```

Download DUTS from the official dataset page: https://saliencydetection.net/duts/

Extract the training and test archives under `data/DUTS`. The loader supports the common nested layout:

```text
data/DUTS/DUTS-TR/DUTS-TR/DUTS-TR-Image
data/DUTS/DUTS-TR/DUTS-TR/DUTS-TR-Mask
data/DUTS/DUTS-TE/DUTS-TE/DUTS-TE-Image
data/DUTS/DUTS-TE/DUTS-TE/DUTS-TE-Mask
```

Inspect the dataset:

```bash
python data_loader.py --data-root data/DUTS
```

## Train

Baseline:

```bash
python train.py --data-root data/DUTS --image-size 128 --batch-size 4 --epochs 15 --variant baseline --experiment-name exp01_baseline_128
```

Improved experiment:

```bash
python train.py --data-root data/DUTS --image-size 224 --batch-size 8 --epochs 25 --variant improved_v2 --experiment-name exp02_improved_v2_224 --learning-rate 5e-4 --iou-weight 1.0 --scheduler plateau --device cuda
```

Final selected experiment:

`improved_v3` is a stronger from-scratch U-Net experiment. It does not replace the required baseline; it is the best real-performing completed model.

```bash
python train.py --data-root data/DUTS --image-size 128 --batch-size 16 --epochs 25 --variant improved_v3 --experiment-name exp03_improved_v3_128_final --learning-rate 5e-4 --iou-weight 0.75 --scheduler plateau --device cuda
```

Additional high-resolution experiment:

```bash
python train.py --data-root data/DUTS --image-size 224 --batch-size 16 --epochs 25 --variant improved_v3 --experiment-name exp04_improved_v3_224 --learning-rate 3e-4 --iou-weight 0.75 --scheduler plateau --device cuda
```

By default, training uses a reproducible 70/15/15 split. Use `--split-mode official` to train/validate on DUTS-TR and test on DUTS-TE.

Checkpointing is automatic. Each epoch writes:

```text
outputs/checkpoints/<experiment>/last_checkpoint.pth
outputs/checkpoints/<experiment>/best_model.pth
outputs/checkpoints/<experiment>/config.json
outputs/checkpoints/<experiment>/dataset.json
outputs/logs/<experiment>_history.csv
```

Restart the same command to resume from `last_checkpoint.pth`. Use `--no-resume` for a fresh run.

## Evaluate

```bash
python evaluate.py --data-root data/DUTS --checkpoint outputs/checkpoints/exp01_baseline_128/best_model.pth --output-dir outputs/evaluation_exp01_baseline_128
```

Selected run:

```bash
python evaluate.py --data-root data/DUTS --image-size 128 --checkpoint outputs/checkpoints/exp03_improved_v3_128_final/best_model.pth --output-dir outputs/evaluation_exp03_improved_v3_128_final
```

Outputs:

```text
outputs/evaluation/metrics.json
outputs/evaluation/sample_predictions.png
```

Metrics: IoU, precision, recall, F1, MAE, loss, BCE, soft IoU, and model inference time per image.

By default, evaluation sweeps thresholds `0.30` through `0.60` on the validation split, saves `selected_threshold` in `metrics.json`, and applies that threshold once on the held-out test split. Use `--no-threshold-sweep` to evaluate a fixed `--threshold`.

Current real held-out results:

| Experiment | Model | Image Size | Device | IoU | Precision | Recall | F1 | MAE | Selected |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `exp01_baseline_128` | baseline | 128 | CPU | 0.6237 | 0.7220 | 0.8224 | 0.7422 | 0.1390 | No |
| `exp02_improved_v2_224` | improved v2 | 224 | Colab CUDA | 0.6927 | 0.8297 | 0.8056 | 0.7849 | 0.0855 | No |
| `exp03_improved_v3_128_final` | improved v3 | 128 | Colab CUDA | 0.7883 | 0.8757 | 0.8841 | 0.8629 | 0.0585 | Yes |
| `exp04_improved_v3_224` | improved v3 | 224 | Colab CUDA | 0.7826 | 0.8516 | 0.9018 | 0.8598 | 0.0643 | No |

`exp03_improved_v3_128_final` is the final selected model because it has the best real held-out test result, not because it has the highest input resolution. It used 7,763,041 trainable parameters, 25 epochs, and validation-selected threshold `0.60`. Full test metrics: IoU `0.7883202339688392`, Precision `0.8757337848209354`, Recall `0.8840669330260525`, F1 `0.8629409748397462`, MAE `0.05853735704663886`, loss `0.26766617425194345`, BCE `0.14678792832802012`, soft IoU `0.7582435099637672`, inference `2.416306752991204 ms/image`.

Final artifacts:

```text
outputs/checkpoints/exp03_improved_v3_128_final/best_model.pth
outputs/checkpoints/exp03_improved_v3_128_final/last_checkpoint.pth
outputs/evaluation_exp03_improved_v3_128_final/metrics.json
outputs/evaluation_exp03_improved_v3_128_final/sample_predictions.png
outputs/logs/exp03_improved_v3_128_final_history.csv
outputs/visualizations/exp03_improved_v3_128_final/validation_predictions.png
```

## Demo

```bash
python -m streamlit run app.py
```

The app shows the uploaded image, predicted saliency mask, overlay, and inference time.

## Smoke Test

Use this before long training runs:

```bash
python train.py --max-samples 12 --epochs 1 --batch-size 2 --image-size 64 --experiment-name smoke --no-resume
python evaluate.py --max-samples 12 --batch-size 2 --image-size 64 --checkpoint outputs/checkpoints/smoke/best_model.pth --output-dir outputs/evaluation_smoke
```

The smoke test verifies the pipeline only. It is not a model-quality result.

## References

- PyTorch BCE Loss documentation: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
- PyTorch DataLoader documentation: https://pytorch.org/docs/stable/data.html
- PyTorch torchvision transforms documentation: https://pytorch.org/vision/stable/transforms
