import hashlib
from datetime import datetime, timezone

import pytest

from edge_bundle_manager import BundleVerificationError, EdgeBundleManager, verify_bundle_files
from edge_contract_v1 import EdgeModelBundle, ModelArtifact


def bundle(path, version="bundle-v1", status="shadow"):
    payload = path.read_bytes()
    artifact = ModelArtifact(
        role="fusion_config", filename=path.name,
        sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload), format="json",
    )
    return EdgeModelBundle(
        bundle_version=version, status=status,
        created_at=datetime(2026, 8, 7, tzinfo=timezone.utc), target="rpi5",
        feature_schema="pose_temporal_109_v1", sample_hz=10,
        temporal_rows=30, artifacts=[artifact],
    )


def test_verified_install_and_atomic_activation(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    artifact = source / "fusion.json"
    artifact.write_text('{"threshold":0.5}')
    manager = EdgeBundleManager(tmp_path / "install")
    installed = manager.install(bundle(artifact), source)
    assert installed.is_dir()
    assert manager.current_link.resolve() == installed.resolve()
    verify_bundle_files(bundle(artifact), installed)


def test_corrupt_or_empty_bundle_is_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    artifact = source / "fusion.json"
    artifact.write_text("good")
    manifest = bundle(artifact)
    artifact.write_text("bad")
    with pytest.raises(BundleVerificationError):
        verify_bundle_files(manifest, source)


def test_benchmark_required_cannot_be_activated(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    artifact = source / "fusion.json"
    artifact.write_text("benchmark")
    manager = EdgeBundleManager(tmp_path / "install")
    with pytest.raises(BundleVerificationError):
        manager.install(bundle(artifact, status="benchmark_required"), source)


def test_rollback_switches_to_previous_verified_directory(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    artifact = source / "fusion.json"
    artifact.write_text("v1")
    manager = EdgeBundleManager(tmp_path / "install")
    v1 = manager.install(bundle(artifact, version="v1"), source)
    artifact.write_text("v2")
    v2 = manager.install(bundle(artifact, version="v2"), source)
    assert manager.current_link.resolve() == v2.resolve()
    assert manager.rollback().resolve() == v1.resolve()
    assert manager.current_link.resolve() == v1.resolve()
