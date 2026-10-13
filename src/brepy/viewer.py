"""Preview a model in an OCP viewer, rebuilding it when a source file changes."""

import os
import platform
import socket
import threading
import time
import traceback
import webbrowser
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import webview
from ocp_viewer import set_port, show
from ocp_viewer.server import create_server

from brepy.build import build_files, load_model

HOST = "127.0.0.1"
PORT = 3939
WATCH_INTERVAL_SECONDS = 0.5
WATCHED_SUFFIXES = (".py", ".json")


def port_in_use(port: int) -> bool:
    try:
        with socket.create_connection((HOST, port), timeout=0.5):
            return True
    except OSError:
        return False


def start_server(port: int) -> tuple[Any, Any]:
    """Serve the viewer from this process, so brepy can see when a page connects."""
    if port_in_use(port):
        raise SystemExit(f"brepy: port {port} is in use; pass another --port")
    viewer, server = create_server({"host": HOST, "port": port})
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return viewer, server


def watched_files(root: Path) -> dict[Path, float]:
    """Modification times of the source files under `root`, skipping hidden folders."""
    times: dict[Path, float] = {}
    for folder, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for name in names:
            if name.endswith(WATCHED_SUFFIXES):
                path = Path(folder) / name
                try:
                    times[path] = path.stat().st_mtime
                except FileNotFoundError:
                    continue
    return times


@dataclass(frozen=True)
class Session:
    """What to show and where: the model, its build arguments, and the viewer's port."""

    path: Path
    root: Path
    parameters: Sequence[str]
    part: str | None
    port: int

    def push(self) -> None:
        model = load_model(self.path, self.root)
        files = build_files(model, self.parameters, self.part)
        shapes = {label: shape for held in files.values() for label, shape in held.items()}
        set_port(self.port, HOST)
        show(*shapes.values(), names=list(shapes))


def keep_current(viewer: Any, session: Session, watch: bool) -> None:
    """Push the parts to each page that connects and, when watching, after each source
    change. The viewer drops a model that arrives while no page is connected."""
    shown_page: object = None
    shown_files = watched_files(session.root)
    while True:
        time.sleep(WATCH_INTERVAL_SECONDS)
        page = viewer.browser
        files = watched_files(session.root) if watch else shown_files
        if page is None or (page is shown_page and files == shown_files):
            continue
        shown_page, shown_files = page, files
        try:
            session.push()
            print("brepy: sent the parts to the viewer", flush=True)
        except Exception:
            traceback.print_exc()


def open_window(url: str) -> None:
    """A native window on macOS and Windows; the browser on Linux, where pywebview
    needs Qt or GTK."""
    if platform.system() != "Linux":
        try:
            webview.create_window("brepy", url)
            webview.start()
            return
        except webview.errors.WebViewException as error:
            print(f"brepy: no native window ({error})")
    print(f"brepy: opening {url} in the browser", flush=True)
    webbrowser.open(url)
    threading.Event().wait()


def preview(session: Session, watch: bool) -> None:
    viewer, server = start_server(session.port)
    threading.Thread(target=keep_current, args=(viewer, session, watch), daemon=True).start()
    try:
        open_window(f"http://{HOST}:{session.port}/")
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        if viewer.browser is not None:
            viewer.browser.close()
