#!/usr/bin/env python3
"""Print the configured web viewer address."""

from __future__ import annotations

import json
from pathlib import Path


CONFIG = Path(__file__).resolve().parents[1] / "config" / "service.json"


def main() -> None:
    with CONFIG.open(encoding="utf-8") as handle:
        config = json.load(handle)
    print(f"http://{config['host']}:{config['port']}/viewer")


if __name__ == "__main__":
    main()
