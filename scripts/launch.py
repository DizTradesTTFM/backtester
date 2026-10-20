"""Open the app: update first, then start it (or just open the browser if it's already running)."""

import socket
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 8501


def already_running() -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


if __name__ == "__main__":
    if already_running():
        webbrowser.open(f"http://localhost:{PORT}")
        sys.exit(0)
    # run the updater as its own process so the app starts on the freshly installed code
    subprocess.run([sys.executable, "-m", "scripts.update"], cwd=ROOT)
    print("\nStarting the Backtester. Keep this window open (minimize it); closing it closes the app.\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(PORT),
         "--browser.gatherUsageStats", "false", "--server.headless", "false", "--client.toolbarMode", "minimal"],
        cwd=ROOT,
    )
