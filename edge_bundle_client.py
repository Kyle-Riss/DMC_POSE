"""Authenticated, checksum-verified edge bundle downloader."""

from __future__ import annotations

import json
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from edge_bundle_manager import BundleVerificationError, EdgeBundleManager
from edge_contract_v1 import EdgeModelBundle


class EdgeBundleDownloadError(RuntimeError):
    pass


class EdgeBundleClient:
    """Download into isolation, verify, then stage/atomically activate a bundle."""

    def __init__(self, server_url: str, token: str, install_root: str | Path):
        self.server_url = server_url.rstrip("/")
        self.token = token.strip()
        if len(self.token) < 32:
            raise ValueError("edge API token must contain at least 32 characters")
        self.manager = EdgeBundleManager(install_root)

    @classmethod
    def from_token_file(cls, server_url, token_file, install_root):
        token = Path(token_file).expanduser().read_text(encoding="utf-8").strip()
        return cls(server_url, token, install_root)

    def _request(self, path: str) -> urllib.request.Request:
        return urllib.request.Request(
            self.server_url + path,
            headers={"Authorization": f"Bearer {self.token}"},
        )

    def fetch_manifest(self) -> EdgeModelBundle:
        try:
            with urllib.request.urlopen(self._request("/edge/model-manifest"), timeout=10) as response:
                return EdgeModelBundle.model_validate(json.load(response))
        except (urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
            raise EdgeBundleDownloadError(f"manifest download failed: {exc}") from exc

    def download_and_install(self, *, activate: bool = False):
        bundle = self.fetch_manifest()
        with tempfile.TemporaryDirectory(prefix="dmc-edge-download-") as temporary:
            root = Path(temporary)
            for artifact in bundle.artifacts:
                quoted = urllib.parse.quote(artifact.filename, safe="")
                try:
                    with urllib.request.urlopen(
                        self._request(f"/edge/artifacts/{quoted}"), timeout=60
                    ) as response, (root / artifact.filename).open("wb") as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                except (urllib.error.URLError, OSError) as exc:
                    raise EdgeBundleDownloadError(
                        f"artifact download failed: {artifact.filename}: {exc}"
                    ) from exc
            try:
                destination = self.manager.install(bundle, root, activate=activate)
            except BundleVerificationError as exc:
                raise EdgeBundleDownloadError(f"bundle verification failed: {exc}") from exc
        return bundle, destination
