import numpy as np
import pytest

from external_temporal_contracts import (
    FALLVISION_FALL,
    WARDY_M04,
    ExternalContractError,
    fallvision_midhip_pixel_features,
    validate_feature_window,
    validate_observed_cadence,
    wardy_from_dmc_109,
)


def test_exact_wardy_window_is_float32_contiguous():
    value = np.zeros((20, 80), dtype=np.float64)
    result = validate_feature_window(value, WARDY_M04)
    assert result.shape == (20, 80)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous


@pytest.mark.parametrize("shape", [(20, 79), (19, 80), (1, 20, 80)])
def test_wrong_external_window_shape_fails_closed(shape):
    with pytest.raises(ExternalContractError):
        validate_feature_window(np.zeros(shape, dtype=np.float32), WARDY_M04)


def test_nan_and_integer_windows_fail_closed():
    with pytest.raises(ExternalContractError):
        validate_feature_window(np.zeros((20, 80), dtype=np.int32), WARDY_M04)
    value = np.zeros((20, 80), dtype=np.float32)
    value[0, 0] = np.nan
    with pytest.raises(ExternalContractError):
        validate_feature_window(value, WARDY_M04)


def test_wardy_accepts_real_observed_10hz_timestamps():
    result = validate_observed_cadence(np.arange(20) / 10.0, WARDY_M04)
    assert result[-1] == pytest.approx(1.9)


def test_wardy_rejects_duplicate_and_gap():
    duplicated = np.arange(20) / 10.0
    duplicated[7] = duplicated[6]
    with pytest.raises(ExternalContractError):
        validate_observed_cadence(duplicated, WARDY_M04)

    gap = np.arange(20) / 10.0
    gap[10:] += 0.2
    with pytest.raises(ExternalContractError):
        validate_observed_cadence(gap, WARDY_M04)


def test_fallvision_cadence_remains_quarantined():
    with pytest.raises(ExternalContractError, match="unresolved"):
        validate_observed_cadence(np.arange(60) / 30.0, FALLVISION_FALL)


def test_fallvision_midhip_pixel_transform_matches_source_contract():
    points = np.zeros((17, 2), dtype=np.float32)
    points[11] = [10.0, 20.0]
    points[12] = [14.0, 24.0]
    points[5] = [15.0, 30.0]
    result = fallvision_midhip_pixel_features(points)
    assert result.shape == (24,)
    assert result[:2].tolist() == pytest.approx([3.0, 8.0])
    assert result[12:16].tolist() == pytest.approx([-2.0, -2.0, 2.0, 2.0])


def test_direct_109d_to_wardy_conversion_is_forbidden():
    with pytest.raises(ExternalContractError, match="forbidden"):
        wardy_from_dmc_109(np.zeros((20, 109), dtype=np.float32))
