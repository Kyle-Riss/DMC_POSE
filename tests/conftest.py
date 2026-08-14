"""Test-suite feature gates for optional development-only dependencies."""

from importlib.util import find_spec


collect_ignore = []
if find_spec("httpx") is None:
    collect_ignore.append("test_edge_control_v1.py")

