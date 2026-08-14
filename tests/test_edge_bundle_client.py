from pathlib import Path

import pytest

from edge_bundle_client import EdgeBundleClient, EdgeBundleDownloadError
from edge_bundle_manager import sha256_file
from edge_contract_v1 import EdgeModelBundle, ModelArtifact, utc_now


def bundle_for(path: Path, status="shadow"):
    return EdgeModelBundle(
        bundle_version="unit-v1", status=status, created_at=utc_now(), target="rpi5",
        sample_hz=10, temporal_rows=30,
        artifacts=[ModelArtifact(role="fusion_config", filename=path.name,
            sha256=sha256_file(path), bytes=path.stat().st_size, format="json")],
    )


def test_client_requires_secret(tmp_path):
    with pytest.raises(ValueError):
        EdgeBundleClient("http://server", "short", tmp_path)


def test_download_stages_verified_bundle(monkeypatch, tmp_path):
    artifact = tmp_path / "fusion.json"
    artifact.write_text('{"ok":true}')
    bundle = bundle_for(artifact)
    client = EdgeBundleClient("http://server", "a" * 32, tmp_path / "install")
    monkeypatch.setattr(client, "fetch_manifest", lambda: bundle)

    class Response:
        def __init__(self, data): self.data = data
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self, _size=-1):
            data, self.data = self.data, b""
            return data

    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: Response(artifact.read_bytes()))
    got, installed = client.download_and_install(activate=False)
    assert got.bundle_version == "unit-v1"
    assert (installed / "fusion.json").read_bytes() == artifact.read_bytes()
    assert not client.manager.current_link.exists()


def test_tampered_download_is_rejected(monkeypatch, tmp_path):
    artifact = tmp_path / "fusion.json"
    artifact.write_text('{"ok":true}')
    bundle = bundle_for(artifact)
    client = EdgeBundleClient("http://server", "a" * 32, tmp_path / "install")
    monkeypatch.setattr(client, "fetch_manifest", lambda: bundle)

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self, _size=-1):
            if hasattr(self, "done"): return b""
            self.done = True
            return b"tampered"

    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: Response())
    with pytest.raises(EdgeBundleDownloadError, match="verification"):
        client.download_and_install()
