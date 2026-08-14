#!/usr/bin/env python3
"""Pi-side CLI for authenticated staging or activation of an edge bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge_bundle_client import EdgeBundleClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--install-root", default="/opt/dmc_pose/models")
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    client = EdgeBundleClient.from_token_file(args.server, args.token_file, args.install_root)
    bundle, destination = client.download_and_install(activate=args.activate)
    print(json.dumps({
        "bundle_version": bundle.bundle_version,
        "status": bundle.status,
        "installed": str(destination),
        "activated": bool(args.activate),
    }, indent=2))


if __name__ == "__main__":
    main()
