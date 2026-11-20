# Training Ground

Train YOLO26 or RF-DETR models for object detection and instance segmentation using Roboflow datasets. The wizard trains a model, exports ONNX, evaluates the selected checkpoint, and uploads the run artifacts to GCS.

## Run locally

From a checkout with `uv` installed, run the following. The project uses Python 3.14.

```sh
uv sync --locked
uv run training-ground wizard
```

On Ubuntu, install `libxcb-cursor-dev` and `libgl1-mesa-dev` if OpenCV requires system libraries.

The wizard saves each run under `runs/` and prints the checkpoint, ONNX, and evaluation paths before upload. Training batches are chosen automatically on CUDA. Use `wizard --batch-size 8` to set a fixed size.

## Evaluate and review predictions

```sh
uv run training-ground evaluate datasets/DATASET runs/RUN/weights/best.pt --backend yolo26 --task boxes
uv run training-ground predict-dir images predictions --checkpoint-path runs/RUN/checkpoint_best_ema.pth --backend rfdetr
uv run training-ground metrics runs/RUN/metrics.csv
uv run training-ground upload runs/RUN
```

Replace `DATASET` and `RUN` with your dataset and run directory names. Use `uv run training-ground COMMAND --help` for command options.

- **Evaluation** writes overlays, plots, and metrics. COCO AP uses a confidence floor of `0.001`. Precision, recall, F1, and overlays use `--threshold` or the model’s operating defaults.
- **Directory prediction** saves images by detected class, preserving source subfolders. Add `--upload` to send images and predictions to Roboflow for review.
- **RF-DETR** offers model defaults or a fixed cap of 50 queries and 20 outputs per image, plus a choice of mask resolution for segmentation. Compatible STM checkpoints automatically use the camera’s preprocessing and per-class thresholds. Set `checkpoint_path` in `tag-bad-images.ipynb` to tag examples from an STM evaluation.
