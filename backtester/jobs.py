"""Start, watch and stop background jobs (backtest batches, data downloads) from the app.

Each job writes its output to results/<name>.log and its process id to results/<name>.pid,
so the app can show progress even after it is closed and reopened.
"""

import os
import subprocess
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
JOB_DIR = ROOT / "results"


def _paths(name):
    return JOB_DIR / f"{name}.pid", JOB_DIR / f"{name}.log"


def start(name: str, args: list[str], env: dict | None = None) -> None:
    JOB_DIR.mkdir(exist_ok=True)
    pid_file, log_file = _paths(name)
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    with open(log_file, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-u", *args],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})},
            creationflags=flags,
        )
    pid_file.write_text(str(proc.pid))


def running(name: str) -> bool:
    pid_file, _ = _paths(name)
    if not pid_file.exists():
        return False
    try:
        p = psutil.Process(int(pid_file.read_text()))
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, ValueError):
        return False


def stop(name: str) -> None:
    pid_file, _ = _paths(name)
    try:
        p = psutil.Process(int(pid_file.read_text()))
        for child in p.children(recursive=True):
            child.kill()
        p.kill()
    except (psutil.Error, ValueError, FileNotFoundError):
        pass


def log_tail(name: str, lines: int = 15) -> str:
    _, log_file = _paths(name)
    if not log_file.exists():
        return ""
    return "\n".join(log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
