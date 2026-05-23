"""Why: The gateway code lives under app/ which is added to sys.path by the
Dockerfile's WORKDIR /app + COPY app/ . layout. Locally the tests live one
directory above app/, so pytest cannot import main, subtitle_utils, etc. without
help. This conftest puts app/ on sys.path before any test imports it.
What: Inserts <repo>/app at position 0 of sys.path at collection time.
Test: Run `pytest tests/test_utils.py -v` from the repo root with no env vars —
imports must succeed without ImportError.
"""

from __future__ import annotations

import os
import sys

_APP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)
