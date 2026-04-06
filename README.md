# training-ground

## Usage

### Install `uv`

```bash
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install the Tool

```bash
sudo apt update && sudo apt-get install -y libxcb-cursor-dev libgl1-mesa-dev \
&& uv tool install --from git+https://github.com/timwedde/training-ground.git training-ground \
&& training-ground wizard
```

### Run the Tool

```bash
training-ground --help
```
