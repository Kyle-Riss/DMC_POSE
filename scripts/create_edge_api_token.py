#!/usr/bin/env python3
"""Create the local edge API token once without printing it."""
from __future__ import annotations

import argparse
import os
import secrets
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("runtime_data/edge_control/api_token"))
    args = parser.parse_args()
    path = args.path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        os.chmod(path, 0o600)
        print(f"token_exists path={path} mode=0600")
        return 0
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, (secrets.token_urlsafe(48) + "\n").encode())
    finally:
        os.close(descriptor)
    print(f"token_created path={path} mode=0600")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
