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

## CLI examples

```bash
training-ground wizard
training-ground evaluate datasets/<dataset> runs/yolo26-nano/weights/best.pt --backend yolo26
training-ground evaluate datasets/<dataset> runs/checkpoint_best_ema.pth --backend rfdetr
training-ground predict-dir ./images ./predictions --checkpoint-path runs/yolo26-nano/weights/best.pt --backend yolo26
training-ground upload runs/yolo26-nano --backend yolo26
training-ground upload runs --backend rfdetr
```
