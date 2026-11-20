# training-ground

`training-ground` trains object-detection or instance-segmentation models from Roboflow datasets with two backends:

- `YOLO26`
- `RF-DETR`

The wizard keeps the same high-level flow for both backends: choose a Roboflow project/version, choose **Boxes** or **Segmentation**, choose the training backend and size, choose GPU sizing, train, export ONNX, plot metrics, evaluate, and upload artifacts.

For RF-DETR, the wizard offers the selected model’s default query and output limits or a fixed limit of 50 queries and 20 outputs per image. For segmentation, the default mask downsample ratio `4` produces masks at one quarter of the input resolution per axis. Ratio `2` doubles the linear mask resolution and uses roughly four times as many mask-head pixels. Ratio `1` produces full-resolution masks and uses roughly sixteen times as many mask-head pixels as ratio `4`.

## Install `uv`

```bash
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install the Tool

```bash
sudo apt update && sudo apt-get install -y libxcb-cursor-dev libgl1-mesa-dev \
&& uv tool install --python 3.14 --from git+https://github.com/timwedde/training-ground.git training-ground
```

## Run the Tool

```bash
training-ground --help
```

## Wizard

```bash
training-ground wizard
```

Each training run gets a new directory under `runs/`. The wizard prints the checkpoint, ONNX model, and evaluation report paths before upload. Standalone evaluations also create a fresh report for each invocation.

Evaluation reports use a confidence floor of `0.001` for COCO AP. Precision, recall, F1, and overlays use `--threshold` or the model’s operating defaults. STM evaluation uses the camera geometry and 20-output cap in both sets of metrics.

When `YOLO26` is selected, the wizard asks for a model size:

- `Nano`
- `Small`
- `Medium`
- `Large`
- `XLarge`

For **Boxes**, these map to `yolo26n.pt` through `yolo26x.pt`. For **Segmentation**, they map to:

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

Outputs are grouped as `OUTPUT_DIR/<class>/<source-relative-path>`, with only that class annotated. Images containing multiple classes appear in each matching folder; images without detections go into `OUTPUT_DIR/no_detections/`. Source subfolders are preserved. Uploads still happen once per source image with all predictions.

### Options

* `--checkpoint-path PATH`: Model checkpoint path (defaults to `best.pt`).
* `--threshold FLOAT`: Override confidence thresholds for enabled classes. Defaults to the camera's per-class thresholds for compatible RF-DETR Medium STM checkpoints, or `0.5` for other models.
* `--backend [yolo26|rfdetr]`: Model backend to use (defaults to `rfdetr`).
* `--task [boxes|segmentation]`: Model task (defaults to `segmentation`).
* `--model-size TEXT`: Model size fallback for legacy RF-DETR checkpoints without embedded model metadata (defaults to nano).
* `--upload`: Enables uploading processed samples to Roboflow.
* `--project TEXT`: Explicit Roboflow project ID for upload (e.g. `stick-detection`).

RF-DETR checkpoints with embedded model metadata automatically restore their architecture, including Medium models with `/2` mask resolution. Both `predict-dir` and `evaluate` preserve the saved mask resolution and select the inference device for the current machine. No `--model-size` override is needed for these checkpoints.

For Medium STM checkpoints such as `model.pth` (432 input, `/2` masks, 20 selections, and class slots 1–3 named pole/stick/tree), prediction and evaluation automatically use the settings from `../fw-jetson/camera`'s Medium STM v3.6/v3.7 profile:

* Confidence: pole **0.90**, stick **0.55**, tree **0.60**, including scores equal to the threshold.
* Disable garbage/background slots before selecting the top 20 query/class pairs; no NMS.
* Preserve aspect ratio with centered black letterboxing to 432×432 and ImageNet normalization.
* Use 216×216 mask logits, the camera's centered width-based mask crop, bilinear resizing to the source image, and a strict logit `> 0` mask cutoff.

An explicit `--threshold` replaces the three enabled-class confidence thresholds; it does not enable garbage/background. Other model architectures and class layouts retain their usual inference behavior. These are matching inference settings, not bit-for-bit TensorRT output: local PyTorch inference uses float32 while the camera engine uses FP16. The camera's width-based mask crop assumes landscape frames; portrait images follow that same camera rule.

CLI overlays and metrics retain one detection per selected query/class pair (the camera groups these into instances with multiple class confidences). Small images use bilinear upsampling; production-sized images use the camera's downsampling filter.

### High-Performance Asynchronous Upload Pipeline

When the `--upload` flag is provided:
- **Immediate Project Selector**: The tool automatically attempts to resolve the Roboflow project name from your checkpoint's runs folder metadata (`metadata.json` / `upload_metadata.json`). If it cannot be resolved automatically, you are immediately prompted with a list of your workspace projects sorted by their last updated date. This happens before the model is loaded or compiled so the rest of the execution runs without interruption.
- **Asynchronous Execution Model**: The main thread runs inference and saves overlays while up to eight background workers upload images. Inference waits for a free upload slot when Roboflow falls behind. If any upload fails, the command reports the failures and exits with status 1.
- **Active Learning Support**: Images with predictions are uploaded with generated COCO box or polygon annotations directly to the **Annotate** tab in Roboflow (`is_prediction=True`) for human review. Images without predictions are uploaded bare.

## CLI examples

```bash
training-ground wizard
training-ground evaluate "datasets/DATASET" runs/YOLO_BOX_RUN/weights/best.pt --backend yolo26 --task boxes
training-ground evaluate "datasets/DATASET" runs/YOLO_SEG_RUN/weights/best.pt --backend yolo26 --task segmentation
training-ground evaluate "datasets/DATASET" runs/RFDETR_RUN/checkpoint_best_ema.pth --backend rfdetr
training-ground metrics runs/RFDETR_RUN/metrics.csv
training-ground predict-dir ./images ./predictions --checkpoint-path runs/YOLO_BOX_RUN/weights/best.pt --backend yolo26 --task boxes
training-ground predict-dir ./false-negatives ./predictions --checkpoint-path model.pth --backend rfdetr --upload
training-ground upload runs/YOLO_BOX_RUN
training-ground upload runs/RFDETR_RUN
```

Replace the uppercase names with the dataset and run directories printed by the wizard. To tag examples from an RF-DETR evaluation in `tag-bad-images.ipynb`, set `checkpoint_path` to the checkpoint you evaluated.
