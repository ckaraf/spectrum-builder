from pathlib import Path
import subprocess
import sys


def main():
    app_path = Path(__file__).with_name("app.py")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(cmd))
