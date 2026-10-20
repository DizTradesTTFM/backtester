"""Self-update: if GitHub has a newer version, download it and replace the app's code.

Your data/, results/, .venv/ and my_strategies/ folders are never touched.
If anything goes wrong (no internet, GitHub down), the current version just keeps running.
"""

import io
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "DizTradesTTFM/backtester"
ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / ".installed_version"
MANIFEST = ROOT / ".installed_files"
KEEP = {"data", "results", ".venv", ".git", "my_strategies"}
# files from older versions that the app replaced
OBSOLETE = ["dashboard.py", "dashboard.bat", "run_overnight.bat", "make_sample_data.bat", "download_data.bat", "schedule_nightly.bat"]


def _get(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "backtester-updater"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def update() -> bool:
    """Returns True if new code was installed."""
    latest = json.loads(_get(f"https://api.github.com/repos/{REPO}/commits/main"))["sha"]
    current = VERSION_FILE.read_text().strip() if VERSION_FILE.exists() else ""
    if latest == current:
        print("Up to date.")
        return False

    print("Downloading the latest version...")
    z = zipfile.ZipFile(io.BytesIO(_get(f"https://codeload.github.com/{REPO}/zip/{latest}", timeout=120)))
    prefix = z.namelist()[0].split("/")[0] + "/"
    old_reqs = (ROOT / "requirements.txt").read_text() if (ROOT / "requirements.txt").exists() else ""
    old_files = set(MANIFEST.read_text().splitlines()) if MANIFEST.exists() else set(OBSOLETE)

    new_files = set()
    for info in z.infolist():
        rel = info.filename[len(prefix):]
        if not rel or info.is_dir() or rel.split("/")[0] in KEEP:
            continue
        dest = ROOT / rel
        new_files.add(rel)
        data = z.read(info)
        if dest.exists() and dest.read_bytes() == data:
            continue  # unchanged; never rewrite a file that might be running (like a .bat)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    for rel in old_files - new_files:
        if rel.split("/")[0] not in KEEP:
            (ROOT / rel).unlink(missing_ok=True)

    if (ROOT / "requirements.txt").read_text() != old_reqs:
        print("Installing new packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")], check=True)

    MANIFEST.write_text("\n".join(sorted(new_files)))
    VERSION_FILE.write_text(latest)
    print("Updated.")
    return True


if __name__ == "__main__":
    try:
        update()
    except Exception as e:
        print(f"Could not check for updates ({e}). Starting the current version.")
