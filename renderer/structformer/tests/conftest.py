"""Shared setup for the StructFormer suite.

Unlike the M4 training tests these need **no session, no teacher targets and no
checkpoint** — they are pure-tensor unit tests, so they run in full on a clean
checkout. Only the ONNX/TensorRT tests skip, and only when those packages (or a
GPU) are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
