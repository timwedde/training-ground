# training-ground

`training-ground` trains segmentation models from Roboflow datasets with two backends:

- `YOLO26`
- `RF-DETR Seg Nano`

The wizard keeps the same high-level flow for both backends: choose a Roboflow project/version, choose the training backend, choose GPU sizing, train, export ONNX, plot metrics, evaluate, and upload artifacts.

## Install `uv`

```bash
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install the Tool

```bash
sudo apt update && sudo apt-get install -y libxcb-cursor-dev libgl1-mesa-dev \
&& uv tool install --from git+https://github.com/timwedde/training-ground.git training-ground
```

## Run the Tool

```bash
training-ground --help
```

## Wizard

```bash
training-ground wizard
```

When `YOLO26` is selected, the wizard asks for a model size:

- `Nano`
- `Small`
- `Medium`
- `Large`
- `XLarge`

These map to:

- `yolo26n-seg.pt`
- `yolo26s-seg.pt`
- `yolo26m-seg.pt`
- `yolo26l-seg.pt`
- `yolo26x-seg.pt`

## Running Inference on a Directory

The `predict-dir` command runs prediction on every image in a directory tree, draws annotated bounding boxes and segmentation masks on them, and saves the resulting files. It also supports uploading processed samples to Roboflow as part of an Active Learning workflow.

```bash
training-ground predict-dir [OPTIONS] INPUT_DIR OUTPUT_DIR
```

### Options

* `--checkpoint-path PATH`: Model checkpoint path (defaults to `best.pt`).
* `--threshold FLOAT`: Prediction confidence threshold (defaults to `0.5`).
* `--backend [yolo26|rfdetr]`: Model backend to use (defaults to `rfdetr`).
* `--model-size TEXT`: Model size (nano, small, medium, large, etc.). Required for non-nano RF-DETR checkpoints.
* `--upload`: Enables uploading processed samples to Roboflow.
* `--project TEXT`: Explicit Roboflow project ID for upload (e.g. `stick-detection`).

### High-Performance Asynchronous Upload Pipeline

When the `--upload` flag is provided:
- **Immediate Project Selector**: The tool automatically attempts to resolve the Roboflow project name from your checkpoint's runs folder metadata (`metadata.json` / `upload_metadata.json`). If it cannot be resolved automatically, you are immediately prompted with a list of your workspace projects sorted by their last updated date. This happens before the model is loaded or compiled so the rest of the execution runs without interruption.
- **Asynchronous Execution Model**: Model inference and overlay saving run at full GPU/CPU speeds on the main thread, while a concurrent background thread pool (`max_workers=8`) manages temporary COCO annotation generation, polygon coordinate tracing via OpenCV, and quiet uploads.
- **Active Learning Support**: Images with predicted masks are uploaded together with their generated COCO JSON polygon segmentations directly to the **Annotate** tab in Roboflow (`is_prediction=True`) for human review. Images without predictions are uploaded bare.

## CLI examples

```bash
training-ground wizard
training-ground evaluate datasets/<dataset> runs/yolo26-nano/weights/best.pt --backend yolo26
training-ground evaluate datasets/<dataset> runs/checkpoint_best_ema.pth --backend rfdetr
training-ground predict-dir ./images ./predictions --checkpoint-path runs/yolo26-nano/weights/best.pt --backend yolo26
training-ground predict-dir ./false-negatives ./predictions --checkpoint-path model.pth --backend rfdetr --model-size nano --upload
training-ground upload runs/yolo26-nano --backend yolo26
training-ground upload runs --backend rfdetr
```
