import pytest

from scripts.extract_swin3d_verifier_embeddings import sample_indices


def test_sample_indices_include_clip_boundaries():
    values = sample_indices(10, 89, 16)
    assert len(values) == 16
    assert values[0] == 10
    assert values[-1] == 89
    assert values == sorted(values)


def test_sample_indices_reject_invalid_range():
    with pytest.raises(ValueError):
        sample_indices(20, 10, 16)
