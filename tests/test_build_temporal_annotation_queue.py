from scripts.build_temporal_annotation_queue import build_queue


def test_multiview_files_share_recording_group_and_stay_training_blocked():
    items = []
    for camera in ("c1_sim", "c2_sim", "c3_sim"):
        items.append({
            "video_id": f"usb_{camera}",
            "video_path": f"/data/{camera}/{camera[:2]}_sim0101.mp4",
            "camera_id": camera,
            "subject_id": None,
            "readable": True,
            "fps": 20.0,
            "frame_count": 240,
            "duration_sec": 12.0,
            "width": 640,
            "height": 360,
        })
    rows = build_queue({"dataset": "usb_sim", "items": items})

    assert len(rows) == 3
    assert {row["recording_id"] for row in rows} == {"sim0101"}
    assert {row["split_group"] for row in rows} == {"usb_sim:sim0101"}
    assert all(row["annotation_status"] == "unreviewed" for row in rows)
    assert all(row["training_eligible"] == "false" for row in rows)
    assert all("subject_identity_unknown" in row["training_blockers"] for row in rows)


def test_unreadable_video_is_not_queued():
    rows = build_queue({
        "dataset": "usb_sim",
        "items": [{"video_id": "bad", "video_path": "/bad.mp4", "readable": False}],
    })
    assert rows == []
