#!/usr/bin/env python3
"""Build selected central shadow modules as CPython ABI-matched extensions."""
from __future__ import annotations

import argparse
import json
import shutil
import sysconfig
import tempfile
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Distribution, Extension

PROJECT = Path(__file__).resolve().parents[1]
MODULES = (
    "server_all_cameras",
    "live_temporal",
    "temporal_model",
    "temporal_features",
    "temporal_sequence",
    "swin3d_verifier",
    "video_verifier_runtime",
)


def build(out_dir: Path) -> dict:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if suffix != ".cpython-311-x86_64-linux-gnu.so":
        raise ValueError(f"central runtime ABI mismatch: {suffix}")
    sources = [PROJECT / f"{name}.py" for name in MODULES]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dmc-shadow-cython-") as temporary:
        build_temp = Path(temporary)
        extensions = [
            Extension(name, [str(source)], extra_compile_args=["-O2"])
            for name, source in zip(MODULES, sources)
        ]
        compiled = cythonize(
            extensions,
            compiler_directives={
                "language_level": 3,
                "binding": True,
                "embedsignature": True,
            },
            build_dir=str(build_temp / "cython"),
            quiet=True,
        )
        distribution = Distribution({"name": "dmc-central-shadow", "ext_modules": compiled})
        command = distribution.get_command_obj("build_ext")
        command.build_lib = str(build_temp / "lib")
        command.build_temp = str(build_temp / "objects")
        command.force = True
        command.ensure_finalized()
        command.run()
        artifacts = []
        for module in MODULES:
            matches = list((build_temp / "lib").glob(module + "*" + suffix))
            if len(matches) != 1:
                raise ValueError(f"expected one extension for {module}, got {matches}")
            target = out_dir / matches[0].name
            shutil.copy2(matches[0], target)
            artifacts.append({"module": module, "path": str(target.resolve()), "bytes": target.stat().st_size})
    report = {
        "schema_version": "dmc_central_shadow_extensions_v1",
        "python_abi_suffix": suffix,
        "modules": artifacts,
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT / "build/protected_shadow_gru_v2",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.out_dir.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
