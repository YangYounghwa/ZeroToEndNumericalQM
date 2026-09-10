"""Make the chapter's self-contained solution modules importable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solutions"))
