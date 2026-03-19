from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import roboflow as roboflow_sdk
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Center, Horizontal, Middle, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Header, Input, Label, Static

AUTH_URL: Final = "https://app.roboflow.com/auth-cli"


@dataclass(slots=True)
class RoboflowSession:
    logged_in: bool
    workspace_name: str | None = None
    workspace_slug: str | None = None

    @property
    def identity_label(self) -> str:
        if self.workspace_slug:
            return self.workspace_slug
        return "Unknown workspace"


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
        return RoboflowSession(logged_in=True)

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
        logged_in=True,
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
            else "Authenticate with Roboflow before proceeding to the training UI."
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
        self.query_one(Input).focus()
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
            self.query_one(Input).focus()
            return

        self._login_in_flight = True
        self.set_controls_disabled(True)
        self.set_status("Running roboflow.login()...")

        try:
            session = await asyncio.to_thread(
                login_with_roboflow, token, force=self.force
            )
        except Exception as exc:
            if self.force and not load_roboflow_session().logged_in:
                self.required = True
                self.set_status(
                    f"Login failed and the previous session was cleared: {exc}",
                    error=True,
                )
            else:
                self.set_status(f"Login failed: {exc}", error=True)
        else:
            self.dismiss(session)
        finally:
            self._login_in_flight = False
            self.set_controls_disabled(False)

    def set_controls_disabled(self, disabled: bool) -> None:
        self.query_one("#auth-token", Input).disabled = disabled

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
        ("ctrl+c", "app.quit", "Quit"),
    ]

    CSS = """
    #main-shell {
        height: 100%;
        padding: 1 2;
    }

    #status-bar {
        dock: bottom;
        height: 3;
        padding: 0 1;
        border: round $accent;
        background: $surface;
    }

    #status-content {
        height: 1fr;
        width: 100%;
        align: left middle;
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

    #content {
        height: 1fr;
        align: center middle;
    }

    #welcome {
        text-style: bold;
        margin-bottom: 1;
    }

    #summary {
        color: $text-muted;
        margin-bottom: 1;
    }

    """

    def __init__(self) -> None:
        super().__init__()
        self._session = load_roboflow_session()

    def compose(self) -> ComposeResult:
        with Vertical(id="main-shell"):
            with Center(id="content"):
                with Middle():
                    with Vertical():
                        yield Label("Training Ground", id="welcome")
                        yield Static(
                            "Roboflow is connected. Dataset selection and training flows can be added here next.",
                            id="summary",
                        )
            with Horizontal(id="status-bar"):
                yield Static("", id="workspace-status")
                yield Static("", id="auth-hint")

    def on_mount(self) -> None:
        self.refresh_session(load_roboflow_session())

    def action_authenticate(self) -> None:
        app = self.app
        if isinstance(app, TrainingGroundApp):
            app.open_auth_screen(force=self._session.logged_in, required=False)

    def refresh_session(self, session: RoboflowSession) -> None:
        self._session = session
        workspace_status = self.query_one("#workspace-status", Static)
        auth_hint = self.query_one("#auth-hint", Static)
        status_bar = self.query_one("#status-bar", Horizontal)
        if session.logged_in:
            workspace_status.update(f"Workspace: {session.identity_label}")
            auth_hint.update("A Reauthenticate")
            status_bar.styles.border = ("round", "green")
            status_bar.styles.color = "green"
        else:
            workspace_status.update("Roboflow not authenticated")
            auth_hint.update("A Log in")
            status_bar.styles.border = ("round", "red")
            status_bar.styles.color = "red"


class TrainingGroundApp(App[None]):
    TITLE = "Training Ground"
    SUB_TITLE = "RF-DETR Training"

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
            callback=lambda session: self.handle_auth_result(
                session, required=required
            ),
        )

    def handle_auth_result(
        self, session: RoboflowSession | None, *, required: bool
    ) -> None:
        current_session = session or load_roboflow_session()
        self.main_screen.refresh_session(current_session)
        if required and not current_session.logged_in:
            self.exit()


def main() -> None:
    TrainingGroundApp().run()


if __name__ == "__main__":
    main()
