"""Checksum-verified, atomic model bundle installer for an edge node."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from edge_contract_v1 import EdgeModelBundle


class BundleVerificationError(ValueError):
    pass


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle_files(bundle: EdgeModelBundle, source_dir: str | Path) -> None:
    root = Path(source_dir).resolve()
    if not bundle.artifacts:
        raise BundleVerificationError("bundle contains no artifacts")
    seen_roles = set()
    for artifact in bundle.artifacts:
        if artifact.role in seen_roles:
            raise BundleVerificationError(f"duplicate artifact role: {artifact.role}")
        seen_roles.add(artifact.role)
        if Path(artifact.filename).name != artifact.filename:
            raise BundleVerificationError("artifact filename must not contain a path")
        path = (root / artifact.filename).resolve()
        if path.parent != root:
            raise BundleVerificationError("artifact escaped source directory")
        if not path.is_file():
            raise BundleVerificationError(f"artifact missing: {artifact.filename}")
        if path.stat().st_size != artifact.bytes:
            raise BundleVerificationError(f"artifact size mismatch: {artifact.filename}")
        if sha256_file(path) != artifact.sha256:
            raise BundleVerificationError(f"artifact checksum mismatch: {artifact.filename}")


class EdgeBundleManager:
    """Stage verified files then atomically swap the current symlink."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.bundles_dir = self.root / "bundles"
        self.current_link = self.root / "current"
        self.previous_link = self.root / "previous"
        self.bundles_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_version(version: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", version):
            raise BundleVerificationError("unsafe bundle version")
        return version

    def install(self, bundle: EdgeModelBundle, source_dir: str | Path, *, activate: bool = True) -> Path:
        version = self._safe_version(bundle.bundle_version)
        if activate and bundle.status == "benchmark_required":
            raise BundleVerificationError("benchmark_required bundle cannot be activated")
        verify_bundle_files(bundle, source_dir)
        destination = self.bundles_dir / version
        if not destination.exists():
            staging = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=self.bundles_dir))
            try:
                for artifact in bundle.artifacts:
                    shutil.copy2(Path(source_dir) / artifact.filename, staging / artifact.filename)
                (staging / "manifest.json").write_text(
                    json.dumps(bundle.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                verify_bundle_files(bundle, staging)
                os.replace(staging, destination)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        else:
            verify_bundle_files(bundle, destination)
        if activate:
            self._activate(destination)
        return destination

    def _atomic_link(self, link: Path, target: Path) -> None:
        temporary = self.root / f".{link.name}.tmp"
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        temporary.symlink_to(target)
        os.replace(temporary, link)

    def _activate(self, destination: Path) -> None:
        if self.current_link.is_symlink():
            old_target = self.current_link.resolve()
            if old_target.exists() and old_target != destination.resolve():
                self._atomic_link(self.previous_link, old_target)
        self._atomic_link(self.current_link, destination.resolve())

    def rollback(self) -> Path:
        if not self.previous_link.is_symlink():
            raise BundleVerificationError("no previous bundle available")
        target = self.previous_link.resolve()
        if not target.is_dir():
            raise BundleVerificationError("previous bundle is unavailable")
        current = self.current_link.resolve() if self.current_link.is_symlink() else None
        self._atomic_link(self.current_link, target)
        if current and current.is_dir():
            self._atomic_link(self.previous_link, current)
        return target

