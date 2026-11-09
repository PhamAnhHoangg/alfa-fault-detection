import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ALFA_DATA", PROJECT / "data"))
