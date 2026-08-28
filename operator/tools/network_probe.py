#!/usr/bin/env python3
"""Perform a generic TCP connectivity check for a configured endpoint."""

from __future__ import annotations

import argparse
import socket
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host")
    parser.add_argument("port", type=int)
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()

    try:
        with socket.create_connection((args.host, args.port), args.timeout):
            pass
    except OSError as exc:
        print(f"unreachable: {args.host}:{args.port} ({exc})")
        return 1

    print(f"reachable: {args.host}:{args.port}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
