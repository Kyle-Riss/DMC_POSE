"""Repository-wide pytest collection compatibility."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAIL = ROOT / "rail"
if str(RAIL) not in sys.path:
    sys.path.insert(0, str(RAIL))


def pytest_ignore_collect(collection_path, config):
    """Skip only the optional HTTP client test when httpx is unavailable."""
    return (
        collection_path.name == "test_edge_auth_v1.py"
        and importlib.util.find_spec("httpx") is None
    )
