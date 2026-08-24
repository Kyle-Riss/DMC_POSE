from pathlib import Path

from scripts.prepare_usb_sim_falldown_manifest import build_manifest


def test_usb_manifest_never_promotes_video_label_to_temporal_interval(tmp_path, monkeypatch):
    camera = tmp_path / "c1_sim"
    camera.mkdir()
    video = camera / "c1_sim0101.mp4"
    video.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "scripts.prepare_usb_sim_falldown_manifest.video_metadata",
        lambda _: {"readable": True, "fps": 20.0, "frame_count": 240, "duration_sec": 12.0, "width": 640, "height": 360},
    )
    manifest = build_manifest(Path(tmp_path))
    item = manifest["items"][0]
    assert item["binary_fall_label"] == 1
    assert item["intervals"] == []
    assert item["training_eligible"] is False
    assert item["temporal_tcn_eligible"] is False
    assert item["diagnostic_eligible"] is True
    assert manifest["warnings"] == []
