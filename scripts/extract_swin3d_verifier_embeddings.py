#!/usr/bin/env python3
"""Extract frozen Swin3D-B RGB embeddings from a reviewed clip manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import Swin3D_B_Weights, swin3d_b


def sample_indices(start: int, end: int, count: int = 16) -> list[int]:
    if end < start or count <= 0:
        raise ValueError("invalid clip sampling request")
    return np.rint(np.linspace(start, end, count)).astype(int).tolist()


def decode_clip(path: Path, start: int, end: int, count: int = 16) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        for index in sample_indices(start, end, count):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"failed decoding frame {index}: {path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    array = np.stack(frames)
    return torch.from_numpy(array).permute(0, 3, 1, 2)


def flush(model, tensors, rows, outputs, *, device: torch.device) -> None:
    if not tensors:
        return
    batch = torch.stack(tensors).to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        encoded = model(batch)
    outputs.extend(encoded.float().cpu().numpy())
    tensors.clear()


def extract(manifest_path: Path, out_dir: Path, *, batch_size: int = 4, device_name: str = "cuda") -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
    transform = weights.transforms()
    model = swin3d_b(weights=weights)
    embedding_size = model.head.in_features
    model.head = nn.Identity()
    device = torch.device(device_name)
    model.eval().to(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "val", "test"):
        tensors = []
        encoded = []
        labels = []
        metadata = []
        errors = []
        rows = [row for row in manifest["clips"] if row["split"] == split]
        for row in rows:
            variants = (False, True) if split == "train" else (False,)
            for flipped in variants:
                try:
                    clip = decode_clip(Path(row["video_path"]), row["start_frame"], row["end_frame"], row["sample_frames"])
                    if flipped:
                        clip = torch.flip(clip, dims=(-1,))
                    tensors.append(transform(clip))
                    labels.append(int(row["label"]))
                    metadata.append({**row, "horizontal_flip": flipped})
                    if len(tensors) >= batch_size:
                        flush(model, tensors, rows, encoded, device=device)
                except Exception as exc:
                    errors.append({"clip_id": row["clip_id"], "horizontal_flip": flipped, "error": f"{type(exc).__name__}: {exc}"})
        flush(model, tensors, rows, encoded, device=device)
        if len(encoded) != len(labels):
            raise AssertionError("embedding/label length mismatch")
        x = np.asarray(encoded, dtype=np.float32).reshape(-1, embedding_size)
        y = np.asarray(labels, dtype=np.int64)
        np.savez_compressed(out_dir / f"{split}.npz", x=x, y=y)
        (out_dir / f"{split}_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        summary[split] = {"embeddings": len(y), "non_fall": int((y == 0).sum()), "fall": int((y == 1).sum()), "errors": errors}
    report = {
        "schema_version": "dmc_swin3d_b_embeddings_v1",
        "manifest": str(manifest_path.resolve()),
        "weights": weights.name,
        "embedding_size": embedding_size,
        "sample_frames": 16,
        "input_size": 224,
        "train_horizontal_flip": True,
        "device": str(device),
        "promotion_eligible": False,
        "summary": summary,
    }
    (out_dir / "embeddings_index.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=project / "external_datasets/manifests/swin3d_verifier_staged_v1.json")
    parser.add_argument("--out-dir", type=Path, default=project / "external_datasets/features/swin3d_b_verifier/staged_v1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = extract(args.manifest.resolve(), args.out_dir.resolve(), batch_size=args.batch_size, device_name=args.device)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"embeddings_index: {(args.out_dir / 'embeddings_index.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
