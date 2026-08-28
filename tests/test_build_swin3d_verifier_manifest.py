from scripts.build_swin3d_verifier_manifest import bounded_window, video_path


def test_video_path_maps_review_ids_to_original_rgb_file(tmp_path):
    assert video_path(tmp_path, "c2_sim", "sim0207") == tmp_path / "c2_0207.mp4"


def test_bounded_window_stays_inside_video():
    assert bounded_window(5, 240) == (0, 79)
    assert bounded_window(235, 240) == (160, 239)
    assert bounded_window(100, 240) == (60, 139)
