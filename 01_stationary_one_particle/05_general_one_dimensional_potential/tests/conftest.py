"""Make the chapter solution modules importable during tests."""

import sys
from pathlib import Path

SOLUTIONS_DIR = Path(__file__).resolve().parents[1] / "solutions"
sys.path.insert(0, str(SOLUTIONS_DIR))
