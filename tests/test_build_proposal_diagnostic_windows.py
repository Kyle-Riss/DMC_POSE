import pandas as pd

from scripts.build_proposal_diagnostic_windows import apply_proposal_targets, recording_splits


def test_recording_split_keeps_group_atomic():
    mapping = recording_splits(["r3", "r1", "r2", "r1", "r4", "r5"])
    assert set(mapping) == {"r1", "r2", "r3", "r4", "r5"}
    assert set(mapping.values()) == {"train", "val", "test"}


def test_proposal_target_uses_source_frame_index():
    frame = pd.DataFrame({"frame_idx": [0, 10, 20, 30], "target": ["non_fall"] * 4, "active_labels": [""] * 4, "split": ["diagnostic"] * 4})
    output = apply_proposal_targets(frame, 10, 20, "train")
    assert output["target"].tolist() == ["non_fall", "fall", "fall", "non_fall"]
    assert set(output["split"]) == {"train"}
