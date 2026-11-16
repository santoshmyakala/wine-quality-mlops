import os
from pathlib import Path

os.environ.setdefault("WINE_QUALITY_ROOT", str(Path(__file__).resolve().parents[1]))
