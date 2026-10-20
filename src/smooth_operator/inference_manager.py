"""Start and supervise the Python 3.12 VLA inference server used by the simulator."""

from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

VLA_ENV_NAME = "smooth-operator-vla"
READY_MARKER = "inference server listening"


def find_vla_python() -> Optional[Path]:
    """Locate the Python 3.12 interpreter that has LeRobot installed."""
    override = os.environ.get("SMOOTH_OPERATOR_VLA_PYTHON")
    candidates = [Path(override)] if override else []
    # Sibling conda environment next to the simulator environment.
    candidates.append(Path(sys.prefix).parent / VLA_ENV_NAME / "bin" / "python")
    candidates.append(Path.cwd() / ".venv312" / "bin" / "python")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class InferenceServerManager:
    """Owns one inference-server subprocess serving a single model spec at a time."""

    def __init__(self, port: int = 5478, log_path: Path = Path("data/inference_server.log")) -> None:
        self.port = port
        self.log_path = log_path
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._spec: Optional[dict[str, Any]] = None
        self._error: Optional[str] = None
        atexit.register(self.stop)

    def status(self) -> dict[str, Any]:
        python = find_vla_python()
        if python is None:
            return {
                "state": "unavailable",
                "detail": f"No Python 3.12 LeRobot environment found (expected conda env '{VLA_ENV_NAME}').",
            }
        process = self._process
        if process is None:
            state = "error" if self._error else "stopped"
        elif process.poll() is not None:
            state = "error"
            fatal = [line for line in self._log_tail(50).splitlines() if line.startswith("FATAL: ")]
            self._error = self._error or (
                fatal[-1][len("FATAL: "):] if fatal else f"Inference server exited with code {process.returncode}"
            )
        elif self._is_ready():
            state = "ready"
        else:
            state = "loading"
        return {
            "state": state,
            "model_id": self._spec["id"] if self._spec else None,
            "checkpoint": self._spec["checkpoint"] if self._spec else None,
            "port": self.port,
            "python": str(python),
            "detail": self._error if state == "error" else self._log_tail(1),
        }

    def start(self, spec: dict[str, Any]) -> None:
        """Start serving `spec`, replacing a server that serves a different model."""
        with self._lock:
            if (
                self._process is not None
                and self._process.poll() is None
                and self._spec == spec
            ):
                return
            self._stop_locked()
            # A server orphaned by a crashed web process would still hold the port.
            subprocess.run(
                ["pkill", "-f", f"smooth_operator.inference_server .*--port {self.port}"],
                check=False,
            )
            time.sleep(0.5)
            python = find_vla_python()
            if python is None:
                raise RuntimeError(self.status()["detail"])
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log = self.log_path.open("w", encoding="utf-8")
            self._process = subprocess.Popen(
                [
                    str(python), "-u", "-m", "smooth_operator.inference_server",
                    "--spec-json", json.dumps(spec),
                    "--port", str(self.port),
                    "--device", "auto",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            log.close()
            self._spec = dict(spec)
            self._error = None

    def ensure_ready(self, spec: dict[str, Any], timeout_s: float = 1800.0) -> int:
        """Block until `spec` is loaded and accepting requests; return the port."""
        self.start(spec)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status = self.status()
            if status["state"] == "ready":
                return self.port
            if status["state"] == "error":
                raise RuntimeError(
                    f"{spec['name']} inference server failed to start: {status['detail']}\n{self._log_tail(15)}"
                )
            time.sleep(1.0)
        raise RuntimeError(f"Timed out waiting for the inference server to load {spec['checkpoint']}")

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        self._spec = None

    def _is_ready(self) -> bool:
        if READY_MARKER not in self._log_tail(400).lower():
            return False
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=1.0):
                return True
        except OSError:
            return False

    def _log_tail(self, lines: int) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(line for line in text.splitlines()[-lines:] if line.strip())
