import numpy as np

from scripts.build_gru_smoke_fixture import build_fixture, make_split


def test_fixture_has_frozen_gru_shape_and_both_classes(tmp_path):
    index = build_fixture(tmp_path)
    train = np.load(tmp_path / "train.npz")
    assert train["x"].shape == (96, 80, 109)
    assert set(train["y"].tolist()) == {0, 1}
    assert index["sample_hz"] == 20.0
    assert index["promotion_eligible"] is False
    assert index["synthetic_smoke_fixture"] is True


def test_fixture_is_deterministic():
    first_x, first_y = make_split(4, seed=7)
    second_x, second_y = make_split(4, seed=7)
    assert np.array_equal(first_x, second_x)
    assert np.array_equal(first_y, second_y)
