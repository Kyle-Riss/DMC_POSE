#!/usr/bin/env python3
"""Check the public readiness endpoint without exposing internal state."""

from __future__ import annotations

import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


HEALTH_URL = "http://127.0.0.1:8030/health"


def main() -> int:
    try:
        with urlopen(HEALTH_URL, timeout=3) as response:
            payload = json.load(response)
    except (HTTPError, URLError, OSError, ValueError) as exc:
        print(f"service unavailable: {exc}")
        return 1

    if payload.get("status") == "ok":
        print("service ready")
        return 0
    print("service not ready")
    return 1


if __name__ == "__main__":
    sys.exit(main())
