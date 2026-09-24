"""Create an isolated CPU environment and run the cached reproduction suite."""

import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11 or newer is required.")
environment = ROOT / ".venv-reproduce"
python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if not python.exists():
    venv.EnvBuilder(with_pip=True).create(environment)
requirements = ROOT / "requirements.txt"
stamp = environment / "requirements-installed.txt"
if not stamp.exists() or stamp.read_text() != requirements.read_text():
    subprocess.run(
        [str(python), "-m", "pip", "install", "-r", str(requirements)], check=True
    )
    stamp.write_text(requirements.read_text())
raise SystemExit(
    subprocess.run(
        [str(python), str(ROOT / "reproduce.py"), *sys.argv[1:]], cwd=ROOT
    ).returncode
)
