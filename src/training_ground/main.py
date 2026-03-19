from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import os
import time
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
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
    Header,
    Input,
    Label,
    LoadingIndicator,
    OptionList,
    ProgressBar,
    Static,
)

AUTH_URL: Final = "https://app.roboflow.com/auth-cli"
DOWNLOAD_ROOT: Final = Path.cwd() / "datasets"
WEIGHTS_ROOT: Final = Path.cwd() / "weights"
AUTH_KEY: Final = "A"
DATASET_FORMAT: Final = "coco"
DATASET_FORMAT_LABEL: Final = "COCO Segmentation"


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
    filename: str
    url: str
    md5_hash: str


MODEL_CHOICES: Final[tuple[ModelChoice, ...]] = (
    ModelChoice(
        "Nano",
        "rfdetr-seg-nano",
        "rf-detr-seg-nano.pt",
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-n-ft.pth",
        "9995497791d0ff1664a1d9ddee9cfd20",
    ),
    ModelChoice(
        "Small",
        "rfdetr-seg-small",
        "rf-detr-seg-small.pt",
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth",
        "0a2a3006381d0c42853907e700eadd08",
    ),
    ModelChoice(
        "Medium",
        "rfdetr-seg-medium",
        "rf-detr-seg-medium.pt",
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-m-ft.pth",
        "a49af1562c3719227ad43d0ca53b4c7a",
    ),
    ModelChoice(
        "Large",
        "rfdetr-seg-large",
        "rf-detr-seg-large.pt",
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-l-ft.pth",
        "275f7b094909544ed2841c94a677d07e",
    ),
    ModelChoice(
        "XLarge",
        "rfdetr-seg-xlarge",
        "rf-detr-seg-xlarge.pt",
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-xl-ft.pth",
        "3693b35d0eea86ebb3e0444f4a611fba",
    ),
    ModelChoice(
        "2XLarge",
        "rfdetr-seg-2xlarge",
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

    if (
        target.exists()
        and compute_md5(target).lower() == model.md5_hash.lower()
    ):
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


class MainScreen(Screen[None]):
    BINDINGS = [
        ("a", "authenticate", "Auth"),
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
        self._step = "projects"
        self.projects: list[ProjectChoice] = []
        self.versions: list[VersionChoice] = []
        self.version_cache: dict[str, list[VersionChoice]] = {}
        self.models: list[ModelChoice] = list(MODEL_CHOICES)
        self.selected_project: ProjectChoice | None = None
        self.selected_version: VersionChoice | None = None
        self.selected_model: ModelChoice | None = None
        self.download_path: Path | None = None
        self.weights_path: Path | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="main-shell"):
            yield Label("Training Ground")
            yield Static(
                "Follow the setup steps to fetch a Roboflow dataset before training starts.",
                id="intro",
            )
            yield Static("", id="steps")
            yield Static("", id="selection")
            yield Static("", id="step-title")
            yield Static("", id="step-help")
            yield LoadingIndicator(id="catalog-loader")
            yield OptionList(id="choices")
            yield Static("", id="wizard-status")
            yield ProgressBar(id="download-progress")
            with Horizontal(id="status-bar"):
                yield Static("", id="workspace-status")
                yield Static("", id="auth-hint")

    async def on_mount(self) -> None:
        self.query_one("#download-progress", ProgressBar).display = False
        self.query_one("#catalog-loader", LoadingIndicator).display = False
        self.refresh_session(load_roboflow_session())
        self.refresh_wizard()
        if self._session.logged_in:
            self.begin_catalog_load()
        else:
            self.show_login_required()

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
            self.selected_project = None
            self.selected_version = None
            self.selected_model = None
            self.download_path = None
            self.weights_path = None
            self._step = "projects"
            self.refresh_wizard()
        elif self._step == "models":
            self.selected_model = None
            self.weights_path = None
            self._step = "download"
            self.refresh_wizard()
        elif self._step == "weights":
            self.weights_path = None
            self._step = "models"
            self.refresh_wizard()
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
            self.download_path = None
            self.weights_path = None
            self.show_versions()
        elif self._step == "versions":
            self.selected_version = self.versions[index]
            self.selected_model = None
            self.download_path = None
            self.weights_path = None
            self.begin_download()
        elif self._step == "models":
            self.selected_model = self.models[index]
            self.weights_path = None
            self.begin_weights_download()

    def refresh_session(self, session: RoboflowSession) -> None:
        self._session = session
        workspace_status = self.query_one("#workspace-status", Static)
        auth_hint = self.query_one("#auth-hint", Static)
        status_bar = self.query_one("#status-bar", Horizontal)

        if session.logged_in:
            workspace_status.update(f"Workspace: {session.identity_label}")
            auth_hint.update(f"{AUTH_KEY} reauth")
            status_bar.styles.border = ("round", "green")
            status_bar.styles.color = "green"
        else:
            workspace_status.update("Workspace: not authenticated")
            auth_hint.update(f"{AUTH_KEY} login")
            status_bar.styles.border = ("round", "red")
            status_bar.styles.color = "red"

    async def restart_wizard(self) -> None:
        self.selected_project = None
        self.selected_version = None
        self.selected_model = None
        self.download_path = None
        self.weights_path = None
        self.projects = []
        self.versions = []
        self.version_cache = {}
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
        self.set_busy(True, "Loading Roboflow projects and dataset versions...")

        try:
            self.projects, self.version_cache = await asyncio.to_thread(
                fetch_projects_with_versions
            )
        except Exception as exc:
            self.projects = []
            self.version_cache = {}
            self.set_status(f"Could not load projects: {exc}", error=True)
        else:
            if not self.projects:
                self.set_status(
                    "No projects with dataset versions are available in this workspace.",
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
        finally:
            self._download_task = None
            self.set_busy(False)
            self.refresh_wizard()

    def refresh_wizard(self) -> None:
        try:
            self.query_one("#steps", Static).update(self.render_steps())
            self.query_one("#selection", Static).update(self.render_selection())
            self.query_one("#step-title", Static).update(self.render_step_title())
            self.query_one("#step-help", Static).update(self.render_step_help())
        except NoMatches:
            return
        self.refresh_choices()

    def refresh_choices(self) -> None:
        choices = self.query_one("#choices", OptionList)
        choices.clear_options()
        choices.display = not self._catalog_loading

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
        except NoMatches:
            return
        loader.display = visible
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
            ("projects", "1. Project", self.selected_project is not None),
            ("versions", "2. Dataset Version", self.selected_version is not None),
            ("download", "3. Dataset Download", self.download_path is not None),
            ("models", "4. Model Size", self.selected_model is not None),
            ("weights", "5. Weights", self.weights_path is not None),
        ]

        lines: list[str] = []
        for key, label, complete in steps:
            marker = ">"
            if complete:
                marker = "x"
            elif self._step != key and not self.is_step_reached(key):
                marker = " "
            elif self._step != key:
                marker = "-"
            lines.append(f"[{marker}] {label}")
        return "\n".join(lines)

    def is_step_reached(self, key: str) -> bool:
        order = ["projects", "versions", "download", "models", "weights"]
        return order.index(self._step) >= order.index(key)

    def render_selection(self) -> str:
        project = self.selected_project.slug if self.selected_project else "-"
        version = str(self.selected_version.number) if self.selected_version else "-"
        model = self.selected_model.label if self.selected_model else "-"
        location = str(self.download_path) if self.download_path else "-"
        weights = str(self.weights_path) if self.weights_path else "-"
        return (
            f"Project: {project}\n"
            f"Version: {version}\n"
            f"Format: {DATASET_FORMAT_LABEL}\n"
            f"Dataset: {location}\n"
            f"Model: {model}\n"
            f"Weights: {weights}"
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
            return "Weights downloaded. Press B to choose a different model size."
        if self.download_path is not None:
            return "Downloading the selected asset."
        return "Downloading the selected asset."


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
