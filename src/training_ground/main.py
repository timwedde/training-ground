import asyncio
import concurrent.futures
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import time
import zipfile
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final

import requests
import roboflow as roboflow_sdk
from roboflow.adapters import rfapi
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Center, Horizontal, Middle, Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Checkbox,
    Header,
    Input,
    Label,
    LoadingIndicator,
    OptionList,
    ProgressBar,
    RichLog,
    Sparkline,
    Static,
)

AUTH_URL: Final = "https://app.roboflow.com/auth-cli"
DOWNLOAD_ROOT: Final = Path.cwd() / "datasets"
WEIGHTS_ROOT: Final = Path.cwd() / "weights"
TRAINING_ROOT: Final = Path.cwd() / "training_runs"
AUTH_KEY: Final = "A"
DATASET_FORMAT: Final = "coco"
DATASET_FORMAT_LABEL: Final = "COCO Segmentation"
TRAINING_LOG_LINES: Final = 1000
GPU_HISTORY_LENGTH: Final = 90


@dataclass(slots=True)
class RoboflowSession:
    logged_in: bool
    workspace_name: str | None = None
    workspace_slug: str | None = None

    @property
    def identity_label(self) -> str:
        if self.workspace_slug:
            return self.workspace_slug
        return "unknown"


@dataclass(slots=True)
class ProjectChoice:
    name: str
    slug: str
    project_type: str
    images: int
    version_count: int


@dataclass(slots=True)
class VersionChoice:
    number: int
    images: int
    created_at: str
    exports: list[str]
    splits_summary: str


@dataclass(slots=True)
class ModelChoice:
    label: str
    size_key: str
    class_name: str
    default_resolution: int
    filename: str
    url: str
    md5_hash: str


@dataclass(slots=True)
class TrainingConfigChoice:
    image_size: int
    epochs: int
    batch_size: int
    num_workers: int
    output_dir: str
    early_stopping: bool
    early_stopping_patience: int
    early_stopping_min_delta: float
    early_stopping_use_ema: bool


@dataclass(slots=True)
class GpuSnapshot:
    index: int
    name: str
    utilization: float
    memory_used_mb: float
    memory_total_mb: float

    @property
    def memory_percent(self) -> float:
        if self.memory_total_mb <= 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100.0


@dataclass(slots=True)
class EnvironmentHealth:
    platform_label: str
    accelerator: str
    gpu_names: list[str] = field(default_factory=list)
    torch_installed: bool = False
    torch_version: str | None = None
    torch_cuda_available: bool = False
    torch_mps_available: bool = False
    torch_cuda_built: bool = False
    torch_mps_built: bool = False
    rfdetr_installed: bool = False
    training_ready: bool = False
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def state_label(self) -> str:
        if self.training_ready:
            return "READY"
        if self.issues:
            return "NOT READY"
        return "CHECK"


MODEL_CHOICES: Final[tuple[ModelChoice, ...]] = (
    ModelChoice(
        "Nano",
        "rfdetr-seg-nano",
        "RFDETRSegNano",
        312,
        "rf-detr-seg-nano.pt",
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-n-ft.pth",
        "9995497791d0ff1664a1d9ddee9cfd20",
    ),
    ModelChoice(
        "Small",
        "rfdetr-seg-small",
        "RFDETRSegSmall",
        384,
        "rf-detr-seg-small.pt",
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth",
        "0a2a3006381d0c42853907e700eadd08",
    ),
    ModelChoice(
        "Medium",
        "rfdetr-seg-medium",
        "RFDETRSegMedium",
        432,
        "rf-detr-seg-medium.pt",
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-m-ft.pth",
        "a49af1562c3719227ad43d0ca53b4c7a",
    ),
    ModelChoice(
        "Large",
        "rfdetr-seg-large",
        "RFDETRSegLarge",
        504,
        "rf-detr-seg-large.pt",
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-l-ft.pth",
        "275f7b094909544ed2841c94a677d07e",
    ),
    ModelChoice(
        "XLarge",
        "rfdetr-seg-xlarge",
        "RFDETRSegXLarge",
        624,
        "rf-detr-seg-xlarge.pt",
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-xl-ft.pth",
        "3693b35d0eea86ebb3e0444f4a611fba",
    ),
    ModelChoice(
        "2XLarge",
        "rfdetr-seg-2xlarge",
        "RFDETRSeg2XLarge",
        768,
        "rf-detr-seg-xxlarge.pt",
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-2xl-ft.pth",
        "040bc3412af840fa8a47e0ff69b552ba",
    ),
)


def roboflow_config_path() -> Path:
    if os.name == "nt":
        default_path = Path.home() / "roboflow" / "config.json"
    else:
        default_path = Path.home() / ".config" / "roboflow" / "config.json"

    return Path(os.getenv("ROBOFLOW_CONFIG_DIR", str(default_path)))


def load_roboflow_session() -> RoboflowSession:
    config_path = roboflow_config_path()
    if not config_path.exists():
        return RoboflowSession(logged_in=False)

    try:
        payload = json.loads(config_path.read_text())
    except json.JSONDecodeError, OSError:
        return RoboflowSession(logged_in=False)

    workspace_slug = payload.get("RF_WORKSPACE")
    workspace_name = None

    workspaces = payload.get("workspaces", {})
    if isinstance(workspaces, dict) and isinstance(workspace_slug, str):
        for workspace in workspaces.values():
            if not isinstance(workspace, dict):
                continue
            if workspace.get("url") != workspace_slug:
                continue

            candidate_name = workspace.get("name") or workspace.get("workspace")
            workspace_name = candidate_name if isinstance(candidate_name, str) else None
            break

    return RoboflowSession(
        logged_in=isinstance(workspace_slug, str) and bool(workspace_slug),
        workspace_name=workspace_name,
        workspace_slug=workspace_slug if isinstance(workspace_slug, str) else None,
    )


def login_with_roboflow(token: str, *, force: bool) -> RoboflowSession:
    original_getpass = roboflow_sdk.getpass
    original_write_line = roboflow_sdk.write_line

    try:
        roboflow_sdk.write_line = lambda *_args, **_kwargs: None
        roboflow_sdk.getpass = lambda _prompt="": token
        roboflow_sdk.login(force=force)
    finally:
        roboflow_sdk.getpass = original_getpass
        roboflow_sdk.write_line = original_write_line

    session = load_roboflow_session()
    if not session.logged_in:
        raise RuntimeError("Roboflow did not create a login session.")
    return session


def with_roboflow_silenced(callable_obj, *args, **kwargs):
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            return callable_obj(*args, **kwargs)


def fetch_workspace() -> object:
    client = with_roboflow_silenced(roboflow_sdk.Roboflow)
    return with_roboflow_silenced(client.workspace)


def fetch_projects() -> list[ProjectChoice]:
    workspace = fetch_workspace()
    project_list = getattr(workspace, "project_list", [])
    projects: list[ProjectChoice] = []

    for info in project_list:
        if not isinstance(info, dict):
            continue

        version_count = int(info.get("versions") or 0)
        if version_count <= 0:
            continue

        project_id = str(info.get("id") or "")
        project_slug = project_id.rsplit("/", 1)[-1]
        if not project_slug:
            continue

        projects.append(
            ProjectChoice(
                name=str(info.get("name") or project_slug),
                slug=project_slug,
                project_type=str(info.get("type") or "unknown"),
                images=int(info.get("images") or 0),
                version_count=version_count,
            )
        )

    projects.sort(key=lambda project: project.name.lower())
    return projects


def build_version_choices(version_info: list[object]) -> list[VersionChoice]:
    versions: list[VersionChoice] = []
    for info in version_info:
        if not isinstance(info, dict):
            continue

        number = parse_version_number(str(info.get("id") or ""))
        created_raw = info.get("created")
        created_at = "unknown date"
        if isinstance(created_raw, (int, float)):
            created_at = datetime.fromtimestamp(created_raw).strftime("%Y-%m-%d")

        exports = [item for item in info.get("exports", []) if isinstance(item, str)]

        versions.append(
            VersionChoice(
                number=number,
                images=int(info.get("images") or 0),
                created_at=created_at,
                exports=exports,
                splits_summary=format_version_splits(info.get("splits")),
            )
        )

    versions.sort(key=lambda version: version.number, reverse=True)
    return versions


def parse_version_number(version_id: str) -> int:
    try:
        return int(version_id.rsplit("/", 1)[-1])
    except TypeError, ValueError:
        return 0


def format_version_splits(splits: object) -> str:
    if not isinstance(splits, dict):
        return "no split data"

    parts: list[str] = []
    for name in ("train", "valid", "test"):
        value = splits.get(name)
        if isinstance(value, int):
            parts.append(f"{name} {value}")

    return ", ".join(parts) if parts else "no split data"


def fetch_versions(project_slug: str) -> list[VersionChoice]:
    workspace = fetch_workspace()
    project = with_roboflow_silenced(workspace.project, project_slug)
    version_info = with_roboflow_silenced(project.get_version_information)
    return build_version_choices(version_info)


def fetch_projects_with_versions() -> tuple[
    list[ProjectChoice], dict[str, list[VersionChoice]]
]:
    projects = fetch_projects()
    versions_by_project: dict[str, list[VersionChoice]] = {}
    if not projects:
        return [], versions_by_project

    max_workers = min(8, len(projects))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_slug = {
            executor.submit(fetch_versions, project.slug): project.slug
            for project in projects
        }
        for future in concurrent.futures.as_completed(future_to_slug):
            slug = future_to_slug[future]
            versions = future.result()
            if versions:
                versions_by_project[slug] = versions

    filtered_projects = [
        project for project in projects if project.slug in versions_by_project
    ]
    return filtered_projects, versions_by_project


def report_progress(
    callback,
    phase: str,
    message: str,
    progress: float,
    total: float | None,
) -> None:
    if callback is not None:
        callback(phase, message, progress, total)


def compute_md5(filepath: Path) -> str:
    digest = hashlib.md5()
    with filepath.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TrainingLogStream:
    def __init__(self, file_handle, callback=None) -> None:
        self.file_handle = file_handle
        self.callback = callback
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0

        self.file_handle.write(text)
        self.file_handle.flush()
        self._buffer += text.replace("\r", "\n")

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            report_progress(self.callback, "log", line.rstrip(), 0, None)

        return len(text)

    def flush(self) -> None:
        self.file_handle.flush()
        if self._buffer:
            report_progress(self.callback, "log", self._buffer.rstrip(), 0, None)
            self._buffer = ""


def query_gpu_snapshots() -> list[GpuSnapshot]:
    command = shutil.which("nvidia-smi")
    if command is None:
        return []

    try:
        result = subprocess.run(
            [
                command,
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return []
    if result.returncode != 0:
        return []

    snapshots: list[GpuSnapshot] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            snapshots.append(
                GpuSnapshot(
                    index=int(parts[0]),
                    name=parts[1],
                    utilization=float(parts[2]),
                    memory_used_mb=float(parts[3]),
                    memory_total_mb=float(parts[4]),
                )
            )
        except ValueError:
            continue

    return snapshots


def run_environment_health_check() -> EnvironmentHealth:
    system = platform.system()
    machine = platform.machine()
    health = EnvironmentHealth(
        platform_label=f"{system} ({machine})",
        accelerator="Unknown",
    )

    snapshots = query_gpu_snapshots()
    if snapshots:
        health.gpu_names = [snapshot.name for snapshot in snapshots]

    health.rfdetr_installed = importlib.util.find_spec("rfdetr") is not None
    if not health.rfdetr_installed:
        health.issues.append("RF-DETR is not installed in the current environment.")

    if importlib.util.find_spec("torch") is None:
        health.accelerator = "No torch runtime"
        health.issues.append("PyTorch is not installed in the current environment.")
        if system == "Darwin":
            health.warnings.append(
                "MPS support could not be verified because torch is missing."
            )
        elif system == "Linux":
            health.warnings.append(
                "CUDA support could not be verified because torch is missing."
            )
        return health

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        health.accelerator = "Torch import failed"
        health.issues.append(f"PyTorch failed to import: {exc}")
        return health

    health.torch_installed = True
    health.torch_version = getattr(torch, "__version__", "unknown")

    cuda_api = getattr(torch, "cuda", None)
    if cuda_api is not None:
        try:
            health.torch_cuda_available = bool(cuda_api.is_available())
        except Exception:
            health.torch_cuda_available = False
        health.torch_cuda_built = getattr(torch.version, "cuda", None) is not None

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is not None:
        try:
            health.torch_mps_built = bool(mps_backend.is_built())
        except Exception:
            health.torch_mps_built = False
        try:
            health.torch_mps_available = bool(mps_backend.is_available())
        except Exception:
            health.torch_mps_available = False

    if system == "Darwin":
        health.accelerator = "MPS"
        if machine != "arm64":
            health.issues.append("MPS training requires Apple Silicon on macOS.")
        if not health.torch_mps_built:
            health.issues.append("This PyTorch build does not include MPS support.")
        elif not health.torch_mps_available:
            health.issues.append("PyTorch MPS support is present but not available.")
        if machine == "arm64" and not health.gpu_names:
            health.gpu_names = ["Apple Silicon GPU"]
        health.training_ready = (
            health.rfdetr_installed
            and health.torch_installed
            and health.torch_mps_built
            and health.torch_mps_available
        )
    elif system == "Linux":
        health.accelerator = "CUDA"
        if not health.gpu_names:
            health.issues.append("No NVIDIA GPUs were detected with nvidia-smi.")
        if not health.torch_cuda_built:
            health.issues.append("This PyTorch build does not include CUDA support.")
        elif not health.torch_cuda_available:
            health.issues.append("PyTorch CUDA support is present but unavailable.")
        health.training_ready = (
            health.rfdetr_installed
            and health.torch_installed
            and bool(health.gpu_names)
            and health.torch_cuda_built
            and health.torch_cuda_available
        )
    else:
        health.accelerator = "Unsupported"
        health.issues.append(
            "Training Ground currently supports MPS on macOS or CUDA on Linux."
        )

    if system == "Linux" and health.gpu_names and not health.torch_cuda_available:
        health.warnings.append(
            "NVIDIA GPUs are present, but the current torch install cannot use CUDA."
        )
    if system == "Darwin" and machine == "arm64" and not health.torch_mps_available:
        health.warnings.append(
            "Apple Silicon is present, but the current torch install cannot use MPS."
        )

    return health


def default_training_config(
    project: ProjectChoice,
    version: VersionChoice,
    model: ModelChoice,
) -> TrainingConfigChoice:
    run_name = f"{project.slug}-v{version.number}-{model.size_key}"
    output_dir = TRAINING_ROOT / run_name
    return TrainingConfigChoice(
        image_size=model.default_resolution,
        epochs=50,
        batch_size=4,
        num_workers=2,
        output_dir=str(output_dir),
        early_stopping=True,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        early_stopping_use_ema=False,
    )


def download_dataset(
    project_slug: str,
    version_number: int,
    callback=None,
) -> Path:
    workspace = fetch_workspace()
    project = with_roboflow_silenced(workspace.project, project_slug)
    version = with_roboflow_silenced(project.version, version_number)

    target = DOWNLOAD_ROOT / project_slug / f"version-{version_number}" / DATASET_FORMAT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "roboflow.zip"

    existing_files = [path for path in target.iterdir() if path.name != "roboflow.zip"]
    if existing_files and not zip_path.exists():
        report_progress(
            callback,
            "complete",
            f"Dataset already exists at {target}",
            1,
            1,
        )
        return target

    api_key = getattr(version, "_Version__api_key")
    workspace_slug = getattr(version, "workspace")
    version_project_slug = getattr(version, "project")
    version_id = str(getattr(version, "version"))

    report_progress(callback, "export", "Preparing COCO export...", 0, 100)
    export_info = rfapi.get_version_export(
        api_key=api_key,
        workspace_url=workspace_slug,
        project_url=version_project_slug,
        version=version_id,
        format=DATASET_FORMAT,
    )

    while export_info.get("ready") is False:
        progress = float(export_info.get("progress") or 0.0)
        report_progress(
            callback,
            "export",
            f"Preparing COCO export... {progress * 100:.0f}%",
            progress * 100,
            100,
        )
        time.sleep(1)
        export_info = rfapi.get_version_export(
            api_key=api_key,
            workspace_url=workspace_slug,
            project_url=version_project_slug,
            version=version_id,
            format=DATASET_FORMAT,
        )

    link = export_info["export"]["link"]
    with requests.get(link, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        total_bytes = response.headers.get("content-length")
        total = int(total_bytes) if total_bytes is not None else None
        written = 0

        report_progress(callback, "download", "Downloading dataset...", 0, total)
        with zip_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                report_progress(
                    callback,
                    "download",
                    "Downloading dataset...",
                    written,
                    total,
                )

    with zipfile.ZipFile(zip_path, "r") as archive:
        members = archive.infolist()
        total_members = max(len(members), 1)
        report_progress(callback, "extract", "Extracting dataset...", 0, total_members)
        for index, member in enumerate(members, start=1):
            try:
                archive.extract(member, target)
            except zipfile.error as exc:
                raise RuntimeError("Error unzipping download") from exc
            report_progress(
                callback,
                "extract",
                "Extracting dataset...",
                index,
                total_members,
            )

    zip_path.unlink(missing_ok=True)
    report_progress(callback, "complete", f"Dataset downloaded to {target}", 1, 1)
    return target


def download_model_weights(model: ModelChoice, callback=None) -> Path:
    WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    target = WEIGHTS_ROOT / model.filename
    temp_target = WEIGHTS_ROOT / f"{model.filename}.tmp"

    if target.exists() and compute_md5(target).lower() == model.md5_hash.lower():
        report_progress(
            callback,
            "complete",
            f"Pretrained weights already exist at {target}",
            1,
            1,
        )
        return target

    report_progress(
        callback, "weights", f"Downloading {model.label} pretrained weights...", 0, 1
    )
    with requests.get(model.url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        total_bytes = response.headers.get("content-length")
        total = int(total_bytes) if total_bytes is not None else None
        written = 0

        with temp_target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                report_progress(
                    callback,
                    "weights",
                    f"Downloading {model.label} pretrained weights...",
                    written,
                    total,
                )

    if model.md5_hash:
        actual_md5 = compute_md5(temp_target)
        if actual_md5.lower() != model.md5_hash.lower():
            temp_target.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded weights failed validation for {model.filename}"
            )

    temp_target.replace(target)
    report_progress(
        callback, "complete", f"Pretrained weights downloaded to {target}", 1, 1
    )
    return target


def run_training(
    model: ModelChoice,
    dataset_path: Path,
    weights_path: Path,
    config: TrainingConfigChoice,
    callback=None,
) -> Path:
    report_progress(
        callback,
        "training",
        f"Loading RF-DETR training stack for {model.label}...",
        0,
        1,
    )
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"

    report_progress(
        callback,
        "training",
        f"Starting training for {model.label}...",
        0,
        1,
    )
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_stream = TrainingLogStream(log_handle, callback)
        with redirect_stdout(log_stream), redirect_stderr(log_stream):
            detr_module = importlib.import_module("rfdetr.detr")
            model_class = getattr(detr_module, model.class_name)
            trainer = model_class(
                pretrain_weights=str(weights_path),
                resolution=config.image_size,
            )
            trainer.train(
                dataset_dir=str(dataset_path),
                output_dir=str(output_dir),
                dataset_file="roboflow",
                epochs=config.epochs,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                early_stopping=config.early_stopping,
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_min_delta=config.early_stopping_min_delta,
                early_stopping_use_ema=config.early_stopping_use_ema,
                progress_bar=False,
            )
        log_stream.flush()
    report_progress(
        callback,
        "complete",
        f"Training completed. Outputs written to {output_dir}",
        1,
        1,
    )
    return output_dir


class AuthScreen(ModalScreen[RoboflowSession | None]):
    BINDINGS = [("escape", "cancel", "Cancel"), ("ctrl+c", "app.quit", "Quit")]

    CSS = """
    AuthScreen {
        align: center middle;
    }

    #auth-shell {
        width: 84;
        max-width: 92%;
        padding: 1 2;
        border: round $surface;
        background: $panel;
    }

    #auth-title {
        text-style: bold;
        content-align: center middle;
        margin-bottom: 1;
    }

    #auth-subtitle {
        color: $text-muted;
        margin-bottom: 1;
    }

    #auth-status {
        min-height: 3;
        margin: 1 0;
        padding: 0 1;
        border: round $primary-background;
        color: $text;
    }

    #auth-help {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(self, *, force: bool, required: bool) -> None:
        super().__init__()
        self.force = force
        self.required = required
        self._login_in_flight = False

    def compose(self) -> ComposeResult:
        title = "Reauthenticate Roboflow" if self.force else "Log In To Roboflow"
        subtitle = (
            "Paste a fresh auth token to replace the existing Roboflow session."
            if self.force
            else "Authenticate with Roboflow before proceeding to dataset setup."
        )

        with Center():
            with Middle():
                with Vertical(id="auth-shell"):
                    yield Label(title, id="auth-title")
                    yield Static(subtitle, id="auth-subtitle")
                    yield Input(
                        placeholder="Roboflow auth token from app.roboflow.com/auth-cli",
                        password=True,
                        id="auth-token",
                    )
                    yield Static("", id="auth-status")
                    yield Static(
                        f"Open {AUTH_URL} to generate the token used here, then press Enter.",
                        id="auth-help",
                    )

    def on_mount(self) -> None:
        self.query_one("#auth-token", Input).focus()
        if self.force:
            self.set_status("Reauthentication will replace the saved Roboflow session.")
        else:
            self.set_status(
                "Open the Roboflow auth page, copy the token, then continue."
            )

    @on(Input.Submitted, "#auth-token")
    async def submit_token(self) -> None:
        await self.run_login()

    def action_cancel(self) -> None:
        if self.required and not load_roboflow_session().logged_in:
            self.app.exit()
            return
        self.dismiss(None)

    async def run_login(self) -> None:
        if self._login_in_flight:
            return

        token = self.query_one("#auth-token", Input).value.strip()
        if not token:
            self.set_status("An auth token is required.", error=True)
            self.query_one("#auth-token", Input).focus()
            return

        self._login_in_flight = True
        self.query_one("#auth-token", Input).disabled = True
        self.set_status("Running roboflow.login()...")

        try:
            session = await asyncio.to_thread(
                login_with_roboflow, token, force=self.force
            )
        except Exception as exc:
            self.set_status(f"Login failed: {exc}", error=True)
        else:
            self.dismiss(session)
        finally:
            self._login_in_flight = False
            self.query_one("#auth-token", Input).disabled = False

    def set_status(self, message: str, *, error: bool = False) -> None:
        status = self.query_one("#auth-status", Static)
        status.update(message)
        if error:
            status.styles.border = ("round", "red")
            status.styles.color = "red"
        else:
            status.styles.border = ("round", "cyan")
            status.styles.color = "white"


class TrainingConfigScreen(ModalScreen[TrainingConfigChoice | None]):
    BINDINGS = [("escape", "cancel", "Cancel"), ("ctrl+c", "app.quit", "Quit")]

    CSS = """
    TrainingConfigScreen {
        align: center middle;
    }

    #config-shell {
        width: 88;
        max-width: 92%;
        padding: 1 2;
        border: round $surface;
        background: $panel;
    }

    .config-label {
        margin-top: 1;
    }

    #config-status {
        min-height: 3;
        margin-top: 1;
        padding: 0 1;
        border: round $surface;
    }

    #config-actions {
        height: auto;
        margin-top: 1;
    }

    Button {
        margin-right: 1;
    }
    """

    def __init__(self, initial: TrainingConfigChoice) -> None:
        super().__init__()
        self.initial = initial

    def compose(self) -> ComposeResult:
        with Center():
            with Middle():
                with Vertical(id="config-shell"):
                    yield Label("Training Configuration")
                    yield Static(
                        "Set the main RF-DETR segmentation training parameters before training starts.",
                    )
                    yield Label("Image Size", classes="config-label")
                    yield Input(str(self.initial.image_size), id="image-size")
                    yield Label("Epochs", classes="config-label")
                    yield Input(str(self.initial.epochs), id="epochs")
                    yield Label("Batch Size", classes="config-label")
                    yield Input(str(self.initial.batch_size), id="batch-size")
                    yield Label("Workers", classes="config-label")
                    yield Input(str(self.initial.num_workers), id="num-workers")
                    yield Label("Output Directory", classes="config-label")
                    yield Input(self.initial.output_dir, id="output-dir")
                    yield Checkbox(
                        "Enable early stopping",
                        value=self.initial.early_stopping,
                        id="early-stopping",
                    )
                    yield Label("Early Stopping Patience", classes="config-label")
                    yield Input(
                        str(self.initial.early_stopping_patience),
                        id="early-stopping-patience",
                    )
                    yield Label("Early Stopping Min Delta", classes="config-label")
                    yield Input(
                        str(self.initial.early_stopping_min_delta),
                        id="early-stopping-min-delta",
                    )
                    yield Checkbox(
                        "Use EMA metric for early stopping",
                        value=self.initial.early_stopping_use_ema,
                        id="early-stopping-use-ema",
                    )
                    with Horizontal(id="config-actions"):
                        yield Button(
                            "Start Training", id="start-training", variant="primary"
                        )
                        yield Button("Cancel", id="cancel-training")
                    yield Static("", id="config-status")

    def on_mount(self) -> None:
        self.query_one("#image-size", Input).focus()
        self.set_status("Adjust the values and start training.")

    @on(Button.Pressed, "#start-training")
    def press_start(self) -> None:
        self.submit()

    @on(Button.Pressed, "#cancel-training")
    def press_cancel(self) -> None:
        self.dismiss(None)

    @on(Input.Submitted)
    def submit_on_enter(self) -> None:
        self.submit()

    def submit(self) -> None:
        try:
            config = TrainingConfigChoice(
                image_size=int(self.query_one("#image-size", Input).value),
                epochs=int(self.query_one("#epochs", Input).value),
                batch_size=int(self.query_one("#batch-size", Input).value),
                num_workers=int(self.query_one("#num-workers", Input).value),
                output_dir=self.query_one("#output-dir", Input).value.strip(),
                early_stopping=self.query_one("#early-stopping", Checkbox).value,
                early_stopping_patience=int(
                    self.query_one("#early-stopping-patience", Input).value
                ),
                early_stopping_min_delta=float(
                    self.query_one("#early-stopping-min-delta", Input).value
                ),
                early_stopping_use_ema=self.query_one(
                    "#early-stopping-use-ema", Checkbox
                ).value,
            )
        except ValueError:
            self.set_status(
                "Image size, epochs, batch size, workers, patience, and min delta must be valid numbers.",
                error=True,
            )
            return

        if (
            config.image_size <= 0
            or config.epochs <= 0
            or config.batch_size <= 0
            or config.num_workers < 0
        ):
            self.set_status(
                "Image size, epochs, and batch size must be positive. Workers cannot be negative.",
                error=True,
            )
            return
        if not config.output_dir:
            self.set_status("An output directory is required.", error=True)
            return
        if config.early_stopping and config.early_stopping_patience <= 0:
            self.set_status(
                "Early stopping patience must be positive when early stopping is enabled.",
                error=True,
            )
            return

        self.dismiss(config)

    def set_status(self, message: str, *, error: bool = False) -> None:
        status = self.query_one("#config-status", Static)
        status.update(message)
        if error:
            status.styles.border = ("round", "red")
            status.styles.color = "red"
        else:
            status.styles.border = ("round", "cyan")
            status.styles.color = "white"


class TrainingDetailsScreen(ModalScreen[None]):
    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("i", "dismiss", "Close"),
        ("ctrl+c", "app.quit", "Quit"),
    ]

    CSS = """
    TrainingDetailsScreen {
        align: center middle;
    }

    #training-details-shell {
        width: 140;
        padding: 1 2;
        border: round $surface;
        background: $panel;
    }

    #training-details-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #training-details-body {
        border: round $surface-lighten-1;
        padding: 1;
    }

    #training-details-help {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(self, summary: str) -> None:
        super().__init__()
        self.summary = summary

    def compose(self) -> ComposeResult:
        with Center():
            with Middle():
                with Vertical(id="training-details-shell"):
                    yield Label("Training Run Details", id="training-details-title")
                    yield Static(self.summary, id="training-details-body")
                    yield Static("Press Escape to close.", id="training-details-help")

    def action_dismiss(self) -> None:
        self.dismiss(None)


class MainScreen(Screen[None]):
    BINDINGS = [
        ("a", "authenticate", "Auth"),
        ("i", "toggle_training_details", "Run Details"),
        ("b", "go_back", "Back"),
        ("ctrl+c", "app.quit", "Quit"),
    ]

    CSS = """
    #main-shell {
        height: 100%;
        padding: 1 2;
    }

    #intro {
        color: $text-muted;
        margin-bottom: 1;
    }

    #steps {
        height: auto;
        margin-bottom: 1;
        padding: 0 1;
        border: round $surface;
    }

    #health-status {
        min-height: 4;
        margin-bottom: 1;
        padding: 0 1;
        border: round $surface;
    }

    #selection {
        min-height: 4;
        margin-bottom: 1;
        padding: 0 1;
        border: round $surface;
    }

    #step-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #step-help {
        color: $text-muted;
        margin-bottom: 1;
    }

    #catalog-loader {
        height: 3;
        margin-bottom: 1;
    }

    #choices {
        height: 1fr;
        border: round $surface;
    }

    #wizard-status {
        min-height: 3;
        margin-top: 1;
        padding: 0 1;
        border: round $surface;
    }

    #download-progress {
        margin-top: 1;
    }

    #training-panel {
        display: none;
        height: 1fr;
        border: round $surface;
        padding: 1;
        margin-top: 1;
    }

    #training-metrics {
        height: 9;
        margin-bottom: 1;
    }

    .training-metric {
        width: 1fr;
        height: 100%;
        border: round $surface-lighten-1;
        padding: 0 1;
        margin-right: 1;
    }

    .metric-title {
        text-style: bold;
    }

    .metric-summary {
        color: $text-muted;
    }

    .metric-chart {
        height: 1fr;
    }

    #training-log-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #training-log {
        height: 1fr;
        border: round $surface-lighten-1;
    }

    #status-bar {
        dock: bottom;
        height: 3;
        padding: 0 1;
        border: round $accent;
        background: $surface;
    }

    #workspace-status {
        width: 1fr;
        content-align: left middle;
    }

    #auth-hint {
        width: auto;
        color: $text-muted;
        content-align: right middle;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._session = load_roboflow_session()
        self._busy = False
        self._catalog_loading = False
        self._catalog_task: asyncio.Task[None] | None = None
        self._download_task: asyncio.Task[None] | None = None
        self._training_task: asyncio.Task[None] | None = None
        self._gpu_task: asyncio.Task[None] | None = None
        self._step = "projects"
        self.projects: list[ProjectChoice] = []
        self.versions: list[VersionChoice] = []
        self.version_cache: dict[str, list[VersionChoice]] = {}
        self.models: list[ModelChoice] = list(MODEL_CHOICES)
        self.selected_project: ProjectChoice | None = None
        self.selected_version: VersionChoice | None = None
        self.selected_model: ModelChoice | None = None
        self.training_config: TrainingConfigChoice | None = None
        self.download_path: Path | None = None
        self.weights_path: Path | None = None
        self.training_output_path: Path | None = None
        self.training_log_path: Path | None = None
        self.environment_health: EnvironmentHealth | None = None
        self._gpu_utilization_history: deque[float] = deque(maxlen=GPU_HISTORY_LENGTH)
        self._gpu_memory_history: deque[float] = deque(maxlen=GPU_HISTORY_LENGTH)
        self._gpu_supported = shutil.which("nvidia-smi") is not None

    def compose(self) -> ComposeResult:
        with Vertical(id="main-shell"):
            yield Label("Training Ground")
            yield Static(
                "Follow the setup steps to fetch a Roboflow dataset before training starts.",
                id="intro",
            )
            yield Static("", id="steps")
            yield Static("", id="health-status")
            yield Static("", id="selection")
            yield Static("", id="step-title")
            yield Static("", id="step-help")
            yield LoadingIndicator(id="catalog-loader")
            yield OptionList(id="choices")
            with Vertical(id="training-panel"):
                with Horizontal(id="training-metrics"):
                    with Vertical(classes="training-metric"):
                        yield Label("GPU Utilization", classes="metric-title")
                        yield Sparkline(
                            [], id="gpu-utilization-chart", classes="metric-chart"
                        )
                        yield Static(
                            "", id="gpu-utilization-summary", classes="metric-summary"
                        )
                    with Vertical(classes="training-metric"):
                        yield Label("GPU Memory", classes="metric-title")
                        yield Sparkline(
                            [], id="gpu-memory-chart", classes="metric-chart"
                        )
                        yield Static(
                            "", id="gpu-memory-summary", classes="metric-summary"
                        )
                yield Label("Training Logs", id="training-log-title")
                yield RichLog(
                    id="training-log",
                    wrap=True,
                    markup=False,
                    highlight=False,
                    auto_scroll=True,
                    max_lines=TRAINING_LOG_LINES,
                )
            yield Static("", id="wizard-status")
            yield ProgressBar(id="download-progress")
            with Horizontal(id="status-bar"):
                yield Static("", id="workspace-status")
                yield Static("", id="auth-hint")

    async def on_mount(self) -> None:
        self.query_one("#download-progress", ProgressBar).display = False
        self.query_one("#catalog-loader", LoadingIndicator).display = False
        self.query_one("#training-panel", Vertical).display = False
        self.refresh_session(load_roboflow_session())
        self.update_gpu_widgets([])
        self.refresh_wizard()
        if self._session.logged_in:
            self.begin_catalog_load()
        else:
            self.show_login_required()

    def action_toggle_training_details(self) -> None:
        summary = self.render_training_details()
        if summary is None:
            return
        self.app.push_screen(TrainingDetailsScreen(summary))

    def action_authenticate(self) -> None:
        app = self.app
        if isinstance(app, TrainingGroundApp):
            app.open_auth_screen(
                force=self._session.logged_in, required=not self._session.logged_in
            )

    async def action_go_back(self) -> None:
        if self._busy:
            return
        if self._step == "versions":
            await self.stop_gpu_monitor()
            self.selected_project = None
            self.selected_version = None
            self.selected_model = None
            self.training_config = None
            self.download_path = None
            self.weights_path = None
            self.training_output_path = None
            self.clear_training_runtime_view()
            self._step = "projects"
            self.refresh_wizard()
        elif self._step == "models":
            await self.stop_gpu_monitor()
            self.selected_model = None
            self.training_config = None
            self.weights_path = None
            self.training_output_path = None
            self.clear_training_runtime_view()
            self._step = "download"
            self.refresh_wizard()
        elif self._step == "weights":
            self.weights_path = None
            self._step = "models"
            self.refresh_wizard()
        elif self._step == "config":
            await self.stop_gpu_monitor()
            self.training_config = None
            self.training_output_path = None
            self.clear_training_runtime_view()
            self._step = "models"
            self.refresh_wizard()
        elif self._step == "training" and self._training_task is None:
            self.training_output_path = None
            self.open_training_config()
        elif self._step == "download":
            self.download_path = None
            self._step = "versions"
            self.refresh_wizard()

    @on(OptionList.OptionSelected, "#choices")
    async def choose_option(self, event: OptionList.OptionSelected) -> None:
        if self._busy or not self._session.logged_in:
            return

        index = event.option_index

        if self._step == "projects":
            self.selected_project = self.projects[index]
            self.selected_version = None
            self.selected_model = None
            self.training_config = None
            self.download_path = None
            self.weights_path = None
            self.training_output_path = None
            self.clear_training_runtime_view()
            self.show_versions()
        elif self._step == "versions":
            self.selected_version = self.versions[index]
            self.selected_model = None
            self.training_config = None
            self.download_path = None
            self.weights_path = None
            self.training_output_path = None
            self.clear_training_runtime_view()
            self.begin_download()
        elif self._step == "models":
            self.selected_model = self.models[index]
            self.training_config = None
            self.weights_path = None
            self.training_output_path = None
            self.clear_training_runtime_view()
            self.begin_weights_download()

    def refresh_session(self, session: RoboflowSession) -> None:
        self._session = session
        workspace_status = self.query_one("#workspace-status", Static)
        status_bar = self.query_one("#status-bar", Horizontal)

        if session.logged_in:
            workspace_status.update(f"Workspace: {session.identity_label}")
            status_bar.styles.border = ("round", "green")
            status_bar.styles.color = "green"
        else:
            workspace_status.update("Workspace: not authenticated")
            status_bar.styles.border = ("round", "red")
            status_bar.styles.color = "red"

        self.refresh_status_hints()

    def refresh_status_hints(self) -> None:
        try:
            auth_hint = self.query_one("#auth-hint", Static)
        except NoMatches:
            return

        hints = [f"{AUTH_KEY} {'reauth' if self._session.logged_in else 'login'}"]
        if self.render_training_details() is not None:
            hints.append("I details")
        auth_hint.update("  |  ".join(hints))

    def clear_training_runtime_view(self) -> None:
        self.training_log_path = None
        self._gpu_utilization_history.clear()
        self._gpu_memory_history.clear()
        try:
            self.query_one("#training-log", RichLog).clear()
        except NoMatches:
            return
        self.update_gpu_widgets([])

    def show_training_panel(self, visible: bool) -> None:
        try:
            panel = self.query_one("#training-panel", Vertical)
            choices = self.query_one("#choices", OptionList)
            loader = self.query_one("#catalog-loader", LoadingIndicator)
        except NoMatches:
            return

        panel.display = visible
        if visible:
            choices.display = False
            loader.display = False
        elif self._catalog_loading:
            loader.display = True
            choices.display = False
        else:
            loader.display = False
            choices.display = True

    def append_training_log(self, line: str) -> None:
        if not line:
            return
        try:
            log_widget = self.query_one("#training-log", RichLog)
        except NoMatches:
            return
        log_widget.write(line)

    def update_gpu_widgets(self, snapshots: list[GpuSnapshot]) -> None:
        try:
            util_chart = self.query_one("#gpu-utilization-chart", Sparkline)
            util_summary = self.query_one("#gpu-utilization-summary", Static)
            memory_chart = self.query_one("#gpu-memory-chart", Sparkline)
            memory_summary = self.query_one("#gpu-memory-summary", Static)
        except NoMatches:
            return

        if not snapshots:
            if self._gpu_supported:
                if self._gpu_utilization_history:
                    util_summary.update("Waiting for GPU telemetry...")
                    memory_summary.update("Waiting for GPU telemetry...")
                else:
                    util_summary.update("No GPU telemetry reported yet.")
                    memory_summary.update("No GPU telemetry reported yet.")
            else:
                util_summary.update("GPU telemetry unavailable on this system.")
                memory_summary.update("GPU telemetry unavailable on this system.")
            util_chart.data = list(self._gpu_utilization_history) or [0]
            memory_chart.data = list(self._gpu_memory_history) or [0]
            return

        average_utilization = sum(item.utilization for item in snapshots) / len(
            snapshots
        )
        average_memory = sum(item.memory_percent for item in snapshots) / len(snapshots)
        self._gpu_utilization_history.append(average_utilization)
        self._gpu_memory_history.append(average_memory)
        util_chart.data = list(self._gpu_utilization_history)
        memory_chart.data = list(self._gpu_memory_history)

        util_summary.update(
            " | ".join(
                f"GPU {item.index}: {item.utilization:.0f}%" for item in snapshots
            )
        )
        memory_summary.update(
            " | ".join(
                (
                    f"GPU {item.index}: {item.memory_used_mb:.0f}/"
                    f"{item.memory_total_mb:.0f} MB ({item.memory_percent:.0f}%)"
                )
                for item in snapshots
            )
        )

    async def monitor_gpu_metrics(self) -> None:
        if not self._gpu_supported:
            self.update_gpu_widgets([])
            return

        try:
            while self._training_task is not None:
                try:
                    snapshots = await asyncio.to_thread(query_gpu_snapshots)
                except Exception:
                    snapshots = []
                self.update_gpu_widgets(snapshots)
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            raise

    def start_gpu_monitor(self) -> None:
        if self._gpu_task is not None:
            return
        self._gpu_task = asyncio.create_task(self.monitor_gpu_metrics())

    async def stop_gpu_monitor(self) -> None:
        if self._gpu_task is None:
            return
        task = self._gpu_task
        self._gpu_task = None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def handle_training_event(
        self,
        phase: str,
        message: str,
        progress: float,
        total: float | None,
    ) -> None:
        if message:
            self.set_status(message)
            self.append_training_log(message)

    async def restart_wizard(self) -> None:
        await self.stop_gpu_monitor()
        self.selected_project = None
        self.selected_version = None
        self.selected_model = None
        self.training_config = None
        self.download_path = None
        self.weights_path = None
        self.training_output_path = None
        self.clear_training_runtime_view()
        self.projects = []
        self.versions = []
        self.version_cache = {}
        self.environment_health = None
        self._step = "projects"
        self.refresh_wizard()
        if self._session.logged_in:
            self.begin_catalog_load()
        else:
            self.show_login_required()

    def begin_catalog_load(self) -> None:
        if self._catalog_task is not None:
            return
        self._catalog_loading = True
        self.show_catalog_loader(True)
        self.refresh_wizard()
        self._catalog_task = asyncio.create_task(self.load_projects())

    async def load_projects(self) -> None:
        self._step = "projects"
        self.set_busy(
            True,
            "Loading Roboflow projects, dataset versions, and training environment health...",
        )

        try:
            (
                (self.projects, self.version_cache),
                self.environment_health,
            ) = await asyncio.gather(
                asyncio.to_thread(fetch_projects_with_versions),
                asyncio.to_thread(run_environment_health_check),
            )
        except Exception as exc:
            self.projects = []
            self.version_cache = {}
            self.environment_health = None
            self.set_status(f"Could not load projects: {exc}", error=True)
        else:
            if not self.projects:
                self.set_status(
                    "No projects with dataset versions are available in this workspace.",
                    error=True,
                )
            elif (
                self.environment_health is not None
                and not self.environment_health.training_ready
            ):
                self.set_status(
                    "Projects loaded, but the training environment is not ready. Review the health panel before starting training.",
                    error=True,
                )
            else:
                self.set_status(
                    "Select a project. Only projects with at least one dataset version are shown."
                )
        finally:
            self._catalog_loading = False
            self._catalog_task = None
            self.show_catalog_loader(False)
            self.set_busy(False)
            self.refresh_wizard()

    def show_versions(self) -> None:
        if self.selected_project is None:
            return

        self._step = "versions"
        self.versions = self.version_cache.get(self.selected_project.slug, [])
        if not self.versions:
            self.set_status(
                "This project does not currently expose any dataset versions.",
                error=True,
            )
        else:
            self.set_status("Select a dataset version.")
        self.refresh_wizard()

    def begin_download(self) -> None:
        if (
            self.selected_project is None
            or self.selected_version is None
            or self._download_task is not None
        ):
            return

        self._step = "download"
        self.show_progress("Preparing COCO export...", 0, 100)
        self.set_busy(
            True,
            f"Downloading {self.selected_project.slug} v{self.selected_version.number} as {DATASET_FORMAT_LABEL}...",
        )
        self.refresh_wizard()
        self._download_task = asyncio.create_task(self.download_selected_dataset())

    def show_models(self) -> None:
        self._step = "models"
        self.set_status("Select an RF-DETR segmentation model size.")
        self.refresh_wizard()

    def build_training_config(self) -> TrainingConfigChoice | None:
        if self.training_config is not None:
            return self.training_config
        if (
            self.selected_project is None
            or self.selected_version is None
            or self.selected_model is None
        ):
            return None

        return default_training_config(
            self.selected_project,
            self.selected_version,
            self.selected_model,
        )

    def open_training_config(self) -> None:
        config = self.build_training_config()
        if config is None:
            return
        self._step = "config"
        self.refresh_wizard()
        self.app.push_screen(
            TrainingConfigScreen(config),
            callback=self.handle_training_config,
        )

    def handle_training_config(self, config: TrainingConfigChoice | None) -> None:
        if config is None:
            if self.training_config is None:
                self._step = "models"
                self.set_status(
                    "Training configuration cancelled. Select a model size to continue."
                )
            else:
                self._step = "config"
                self.set_status("Training configuration cancelled.")
            self.refresh_wizard()
            return
        self.training_config = config
        self.begin_training()

    def begin_training(self) -> None:
        if (
            self.selected_model is None
            or self.download_path is None
            or self.weights_path is None
            or self.training_config is None
            or self._training_task is not None
        ):
            return
        if self.environment_health is not None and not self.environment_health.training_ready:
            issue = (
                self.environment_health.issues[0]
                if self.environment_health.issues
                else "The training environment is not ready."
            )
            self.set_status(f"Training cannot start: {issue}", error=True)
            return

        self._step = "training"
        self.training_output_path = None
        self.clear_training_runtime_view()
        output_dir = Path(self.training_config.output_dir).expanduser().resolve()
        self.training_log_path = output_dir / "training.log"
        self.hide_progress()
        self.show_training_panel(True)
        self.set_busy(
            True,
            f"Starting training for {self.selected_model.label}...",
        )
        self.append_training_log(
            f"Starting RF-DETR training for {self.selected_model.label}."
        )
        self.refresh_wizard()
        self._training_task = asyncio.create_task(self.run_training_task())
        self.start_gpu_monitor()

    def begin_weights_download(self) -> None:
        if self.selected_model is None or self._download_task is not None:
            return

        self._step = "weights"
        self.show_progress(
            f"Preparing {self.selected_model.label} pretrained weights...",
            0,
            100,
        )
        self.set_busy(
            True, f"Downloading {self.selected_model.label} pretrained weights..."
        )
        self.refresh_wizard()
        self._download_task = asyncio.create_task(self.download_selected_weights())

    async def download_selected_dataset(self) -> None:
        if self.selected_project is None or self.selected_version is None:
            return

        try:
            self.download_path = await asyncio.to_thread(
                download_dataset,
                self.selected_project.slug,
                self.selected_version.number,
                lambda phase, message, progress, total: self.app.call_from_thread(
                    self.update_download_progress,
                    phase,
                    message,
                    progress,
                    total,
                ),
            )
        except Exception as exc:
            self.download_path = None
            self.hide_progress()
            self.set_status(f"Dataset download failed: {exc}", error=True)
        else:
            self.hide_progress()
            self.set_status(f"Dataset downloaded to {self.download_path}")
            self.show_models()
        finally:
            self._download_task = None
            self.set_busy(False)
            self.refresh_wizard()

    async def download_selected_weights(self) -> None:
        if self.selected_model is None:
            return

        try:
            self.weights_path = await asyncio.to_thread(
                download_model_weights,
                self.selected_model,
                lambda phase, message, progress, total: self.app.call_from_thread(
                    self.update_download_progress,
                    phase,
                    message,
                    progress,
                    total,
                ),
            )
        except Exception as exc:
            self.weights_path = None
            self.hide_progress()
            self.set_status(f"Weight download failed: {exc}", error=True)
        else:
            self.hide_progress()
            self.set_status(f"Pretrained weights downloaded to {self.weights_path}")
            self.open_training_config()
        finally:
            self._download_task = None
            self.set_busy(False)
            self.refresh_wizard()

    async def run_training_task(self) -> None:
        if (
            self.selected_model is None
            or self.download_path is None
            or self.weights_path is None
            or self.training_config is None
        ):
            return

        try:
            self.training_output_path = await asyncio.to_thread(
                run_training,
                self.selected_model,
                self.download_path,
                self.weights_path,
                self.training_config,
                lambda phase, message, progress, total: self.app.call_from_thread(
                    self.handle_training_event,
                    phase,
                    message,
                    progress,
                    total,
                ),
            )
        except Exception as exc:
            self.training_output_path = None
            self.set_status(f"Training failed: {exc}", error=True)
            self.append_training_log(f"Training failed: {exc}")
        else:
            self.set_status(
                f"Training completed. Outputs written to {self.training_output_path}"
            )
        finally:
            self._training_task = None
            await self.stop_gpu_monitor()
            snapshots = await asyncio.to_thread(query_gpu_snapshots)
            self.update_gpu_widgets(snapshots)
            self.show_training_panel(self._step == "training")
            self.set_busy(False)
            self.refresh_wizard()

    def refresh_wizard(self) -> None:
        try:
            steps = self.query_one("#steps", Static)
            health_status = self.query_one("#health-status", Static)
            selection = self.query_one("#selection", Static)
            step_title = self.query_one("#step-title", Static)
            step_help = self.query_one("#step-help", Static)
        except NoMatches:
            return

        steps.update(self.render_steps())
        health_status.update(self.render_health_status())
        self.apply_health_styles(health_status)
        health_status.display = self._step == "projects"
        selection.update(self.render_selection())
        selection.display = self._step != "training"
        step_title.update(self.render_step_title())
        step_help.update(self.render_step_help())
        self.refresh_status_hints()
        self.show_training_panel(self._step == "training")
        self.refresh_choices()

    def refresh_choices(self) -> None:
        choices = self.query_one("#choices", OptionList)
        choices.clear_options()
        choices.display = not self._catalog_loading and self._step != "training"

        prompts: list[str]
        if not self._session.logged_in or self._catalog_loading:
            prompts = []
        elif self._step == "projects":
            prompts = [
                f"{project.slug} | {project.project_type} | {project.version_count} versions | {project.images} images"
                for project in self.projects
            ]
        elif self._step == "versions":
            prompts = [
                f"Version {version.number} | {version.images} images | {version.created_at} | {version.splits_summary}"
                for version in self.versions
            ]
        elif self._step == "models":
            prompts = [f"{model.label} | {model.size_key}" for model in self.models]
        else:
            prompts = []

        if prompts:
            choices.add_options(prompts)
            choices.disabled = self._busy
            choices.highlighted = 0
            choices.focus()
        else:
            choices.disabled = True

    def show_login_required(self) -> None:
        self.set_status("Log into Roboflow to begin the dataset setup wizard.")
        self.refresh_wizard()

    def set_busy(self, busy: bool, message: str | None = None) -> None:
        self._busy = busy
        try:
            choices = self.query_one("#choices", OptionList)
        except NoMatches:
            return
        choices.disabled = busy or choices.option_count == 0
        if message is not None:
            self.set_status(message)

    def set_status(self, message: str, *, error: bool = False) -> None:
        try:
            status = self.query_one("#wizard-status", Static)
        except NoMatches:
            return
        status.update(message)
        if error:
            status.styles.border = ("round", "red")
            status.styles.color = "red"
        else:
            status.styles.border = ("round", "cyan")
            status.styles.color = "white"

    def show_catalog_loader(self, visible: bool) -> None:
        try:
            loader = self.query_one("#catalog-loader", LoadingIndicator)
            panel = self.query_one("#training-panel", Vertical)
        except NoMatches:
            return
        loader.display = visible
        panel.display = False
        try:
            choices = self.query_one("#choices", OptionList)
        except NoMatches:
            return
        choices.display = not visible

    def show_progress(self, message: str, progress: float, total: float | None) -> None:
        try:
            bar = self.query_one("#download-progress", ProgressBar)
        except NoMatches:
            return
        bar.display = True
        bar.update(total=total, progress=progress)
        self.set_status(message)

    def update_download_progress(
        self,
        phase: str,
        message: str,
        progress: float,
        total: float | None,
    ) -> None:
        try:
            bar = self.query_one("#download-progress", ProgressBar)
        except NoMatches:
            return
        if not bar.display:
            bar.display = True
        bar.update(total=total, progress=progress)
        self.set_status(message)

    def hide_progress(self) -> None:
        try:
            bar = self.query_one("#download-progress", ProgressBar)
        except NoMatches:
            return
        bar.display = False

    def render_steps(self) -> str:
        steps = [
            ("projects", "Project", self.selected_project is not None),
            ("versions", "Version", self.selected_version is not None),
            ("download", "Dataset", self.download_path is not None),
            ("models", "Model", self.selected_model is not None),
            ("weights", "Weights", self.weights_path is not None),
            ("config", "Config", self.training_config is not None),
            ("training", "Train", self.training_output_path is not None),
        ]

        parts: list[str] = []
        for key, label, complete in steps:
            if complete:
                parts.append(f"✓ {label}")
            elif self._step == key:
                parts.append(f"● {label}")
            elif self.is_step_reached(key):
                parts.append(f"• {label}")
            else:
                parts.append(label)
        return "  >  ".join(parts)

    def is_step_reached(self, key: str) -> bool:
        order = [
            "projects",
            "versions",
            "download",
            "models",
            "weights",
            "config",
            "training",
        ]
        return order.index(self._step) >= order.index(key)

    def render_selection(self) -> str:
        project = self.selected_project.slug if self.selected_project else "-"
        version = str(self.selected_version.number) if self.selected_version else "-"
        model = self.selected_model.label if self.selected_model else "-"
        location = str(self.download_path) if self.download_path else "-"
        weights = str(self.weights_path) if self.weights_path else "-"
        training_output = (
            str(self.training_output_path) if self.training_output_path else "-"
        )
        training_log = str(self.training_log_path) if self.training_log_path else "-"

        if self._step == "training":
            return (
                "Run details are hidden during training.\n"
                "Press I to show the full configuration and paths.\n"
                f"Training Log: {training_log}\n"
                f"Training Output: {training_output}"
            )

        return (
            f"Project: {project}\n"
            f"Version: {version}\n"
            f"Format: {DATASET_FORMAT_LABEL}\n"
            f"Dataset: {location}\n"
            f"Model: {model}\n"
            f"Weights: {weights}\n"
            f"Training Log: {training_log}\n"
            f"Training Output: {training_output}"
        )

    def render_health_status(self) -> str:
        if self._catalog_loading and self.environment_health is None:
            return "Health: checking training environment...\nTorch, RF-DETR, and accelerator support are being verified."
        if self.environment_health is None:
            return "Health: not checked yet"

        health = self.environment_health
        torch_label = (
            f"torch {health.torch_version}"
            if health.torch_installed and health.torch_version
            else "torch missing"
        )
        gpu_label = ", ".join(health.gpu_names) if health.gpu_names else "no compatible GPU detected"
        detail = "ready for training" if health.training_ready else (
            health.issues[0] if health.issues else "environment needs attention"
        )
        return (
            f"Health: {health.state_label} | {health.platform_label} | {health.accelerator} | {torch_label}\n"
            f"Devices: {gpu_label}\n"
            f"RF-DETR: {'installed' if health.rfdetr_installed else 'missing'} | {detail}"
        )

    def apply_health_styles(self, widget: Static) -> None:
        if self.environment_health is None:
            widget.styles.border = ("round", "yellow")
            widget.styles.color = "white"
            return
        if self.environment_health.training_ready:
            widget.styles.border = ("round", "green")
            widget.styles.color = "white"
            return
        widget.styles.border = ("round", "red")
        widget.styles.color = "white"

    def render_training_details(self) -> str | None:
        if self.training_config is None:
            return None

        project = self.selected_project.slug if self.selected_project else "-"
        version = str(self.selected_version.number) if self.selected_version else "-"
        model = self.selected_model.label if self.selected_model else "-"
        dataset = str(self.download_path) if self.download_path else "-"
        weights = str(self.weights_path) if self.weights_path else "-"
        training_log = str(self.training_log_path) if self.training_log_path else "-"
        training_output = (
            str(self.training_output_path) if self.training_output_path else "-"
        )
        early_stopping = "enabled" if self.training_config.early_stopping else "disabled"
        use_ema = "yes" if self.training_config.early_stopping_use_ema else "no"

        return (
            f"Project: {project}\n"
            f"Version: {version}\n"
            f"Format: {DATASET_FORMAT_LABEL}\n"
            f"Dataset: {dataset}\n"
            f"Model: {model}\n"
            f"Weights: {weights}\n"
            f"Image Size: {self.training_config.image_size}\n"
            f"Epochs: {self.training_config.epochs}\n"
            f"Batch Size: {self.training_config.batch_size}\n"
            f"Workers: {self.training_config.num_workers}\n"
            f"Early Stopping: {early_stopping}\n"
            f"Patience: {self.training_config.early_stopping_patience}\n"
            f"Min Delta: {self.training_config.early_stopping_min_delta}\n"
            f"EMA Metric: {use_ema}\n"
            f"Output Dir: {self.training_config.output_dir}\n"
            f"Training Log: {training_log}\n"
            f"Training Output: {training_output}"
        )

    def render_step_title(self) -> str:
        if self._catalog_loading:
            return "Preparing Roboflow Workspace"
        titles = {
            "projects": "Choose A Roboflow Project",
            "versions": "Choose A Dataset Version",
            "download": "Dataset Download",
            "models": "Choose RF-DETR Segmentation Size",
            "weights": "Pretrained Weight Download",
            "config": "Training Configuration",
            "training": "RF-DETR Training Monitor",
        }
        return titles[self._step]

    def render_step_help(self) -> str:
        if not self._session.logged_in:
            return f"Press {AUTH_KEY} to log into Roboflow."
        if self._catalog_loading:
            return "Fetching projects and dataset versions in the background."
        if self._step == "projects":
            return "Use Up/Down to move, Enter to select. Projects without dataset versions are hidden."
        if self._step == "versions":
            return f"Use Up/Down to move, Enter to download as {DATASET_FORMAT_LABEL}. Press B to go back."
        if self._step == "models":
            return "Use Up/Down to move, Enter to fetch pretrained weights. Press B to go back."
        if self._step == "weights" and self.weights_path is not None:
            return (
                "Weights downloaded. The training configuration window will open next."
            )
        if self._step == "config":
            return (
                "Adjust the training settings in the configuration window. "
                "Press B to choose a different model size."
            )
        if self._step == "training" and self._training_task is not None:
            return (
                "Training is active. Live logs are streamed below, and GPU charts "
                "appear when telemetry is available. Press I for run details."
            )
        if self._step == "training" and self.training_output_path is not None:
            return (
                "Training finished. Review the logs below, or press B to adjust the "
                "configuration and train again. Press I for run details."
            )
        if self._step == "weights":
            return "Downloading pretrained weights for the selected RF-DETR segmentation model."
        if self._step == "download":
            return "Downloading the selected COCO Segmentation dataset."
        return "Complete the current step to continue."


class TrainingGroundApp(App[None]):
    TITLE = "Training Ground"
    SUB_TITLE = "Dataset Wizard"

    def __init__(self) -> None:
        super().__init__()
        self.main_screen = MainScreen()

    def compose(self) -> ComposeResult:
        yield Header()

    def on_mount(self) -> None:
        self.push_screen(self.main_screen)
        if not load_roboflow_session().logged_in:
            self.open_auth_screen(force=False, required=True)

    def open_auth_screen(self, *, force: bool, required: bool) -> None:
        self.push_screen(
            AuthScreen(force=force, required=required),
            callback=lambda session: self.call_after_refresh(
                self.handle_auth_result,
                session,
                required,
            ),
        )

    async def handle_auth_result(
        self, session: RoboflowSession | None, required: bool
    ) -> None:
        current_session = session or load_roboflow_session()
        self.main_screen.refresh_session(current_session)
        if required and not current_session.logged_in:
            self.exit()
            return
        await self.main_screen.restart_wizard()


def main() -> None:
    TrainingGroundApp().run()


if __name__ == "__main__":
    main()
