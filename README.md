# Training Ground

Train YOLO26 or RF-DETR models for object detection and instance segmentation using Roboflow datasets. The wizard trains a model, exports ONNX, evaluates the selected checkpoint, and uploads the run artifacts to GCS.

## Install on an Ubuntu GPU VM

Paste this command to install the tool and Python 3.14. It installs `uv` if needed.

```sh
sudo apt update && sudo apt-get install -y libxcb-cursor-dev libgl1-mesa-dev ca-certificates curl git \
&& export PATH="$HOME/.local/bin:$PATH" \
&& (command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh) \
&& uv tool install --python 3.14 --from git+https://github.com/timwedde/training-ground.git training-ground
```

Then run `training-ground wizard`.

## Run from a checkout

From a checkout with `uv` installed, run the following. The project uses Python 3.14.

```sh
uv sync --locked
uv run training-ground wizard
```

The wizard saves each run under `runs/` and prints the checkpoint, ONNX, and evaluation paths before upload. On CUDA, YOLO26 auto-batching targets 95% of available VRAM. Use `wizard --batch-size 8` to set a fixed size.

## Train on native-resolution tiles

For CityWizard or other high-resolution COCO datasets, select **Boxes → YOLO26 → Native-resolution tiles** in the wizard. To train from the local export and save the run locally:

```sh
uv run training-ground wizard --dataset-path 'Waste Detection 5928.v34i.coco'
```

Training uses overlapping 640 × 640 crops at native pixel scale. The loader requests up to 16 workers based on available CPUs, reserving two for the trainer. Standalone evaluation of a tiled checkpoint needs `--backend yolo26 --task boxes --tiled`. Evaluation merges same-class duplicates across tiles before scoring against the original COCO labels. It reports tile counts and inference time and uses validation to choose the merge IoU and confidence unless the sweep is disabled. An explicit `--threshold` overrides the recommended confidence.

The ONNX model takes one 640 × 640 tile and returns NMS-free detection rows. The grid, preprocessing, output columns, merge rules, and ordered labels are saved in the evaluation's `recommended_settings.json` for the TensorRT runtime.

## Evaluate and review predictions

```sh
uv run training-ground evaluate datasets/DATASET runs/RUN/weights/best.pt --backend yolo26 --task boxes
uv run training-ground predict-dir images predictions --checkpoint-path runs/RUN/checkpoint_best_ema.pth --backend rfdetr
uv run training-ground metrics runs/RUN/metrics.csv
uv run training-ground upload runs/RUN
```

Replace `DATASET` and `RUN` with your dataset and run directory names. Use `uv run training-ground COMMAND --help` for command options.

- **Evaluation** writes overlays, plots, and metrics. COCO AP uses a confidence floor of `0.001`. Precision, recall, F1, and overlays use `--threshold` or the model’s operating defaults.
- **Recommended settings** are saved in `recommended_settings.json` with the evaluation. The report lists the checkpoint’s model settings and recommends confidence cutoffs and, where supported, YOLO NMS IoU using F1 on the validation split. Use `evaluate --no-settings-sweep` to skip the extra inference passes.
- **Directory prediction** saves images by detected class, preserving source subfolders. Add `--upload` to send images and predictions to Roboflow for review.
- **RF-DETR** offers model defaults or a fixed cap of 50 queries and 20 outputs per image, plus a choice of mask resolution for segmentation. Compatible STM checkpoints automatically use the camera’s preprocessing and per-class thresholds. Set `checkpoint_path` in `tag-bad-images.ipynb` to tag examples from an STM evaluation.
