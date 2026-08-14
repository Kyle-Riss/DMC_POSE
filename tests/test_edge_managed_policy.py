from edge_managed_policy import edge_runtime_policy, parse_camera_ids


def test_parse_camera_ids_ignores_whitespace_and_empty_values():
    assert parse_camera_ids(" bed_161,bed_162, ,bed_161 ") == {"bed_161", "bed_162"}


def test_healthy_edge_suppresses_watcher_and_reduces_empty_probe():
    result = edge_runtime_policy(
        managed=True, connected=True, disconnected_for_sec=0,
        failover_grace_sec=3, normal_empty_probe_hz=0.75,
        managed_empty_probe_hz=0.05,
    )
    assert result == {
        "suppress_local_watcher": True,
        "fallback_active": False,
        "empty_probe_hz": 0.05,
    }


def test_disconnect_grace_does_not_thrash_watcher():
    result = edge_runtime_policy(
        managed=True, connected=False, disconnected_for_sec=2.9,
        failover_grace_sec=3, normal_empty_probe_hz=0.75,
        managed_empty_probe_hz=0.05,
    )
    assert result["suppress_local_watcher"] is True
    assert result["fallback_active"] is False


def test_disconnect_after_grace_restores_central_fallback():
    result = edge_runtime_policy(
        managed=True, connected=False, disconnected_for_sec=3.0,
        failover_grace_sec=3, normal_empty_probe_hz=0.75,
        managed_empty_probe_hz=0.05,
    )
    assert result["suppress_local_watcher"] is False
    assert result["fallback_active"] is True
    assert result["empty_probe_hz"] == 0.75


def test_stale_edge_result_restores_central_fallback_after_grace():
    result = edge_runtime_policy(
        managed=True, connected=True, result_fresh=False,
        disconnected_for_sec=3.0, failover_grace_sec=3,
        normal_empty_probe_hz=0.75, managed_empty_probe_hz=0.05,
    )
    assert result["suppress_local_watcher"] is False
    assert result["fallback_active"] is True
    assert result["empty_probe_hz"] == 0.75


def test_unmanaged_camera_keeps_existing_central_policy():
    result = edge_runtime_policy(
        managed=False, connected=False, disconnected_for_sec=999,
        failover_grace_sec=3, normal_empty_probe_hz=0.75,
        managed_empty_probe_hz=0.05,
    )
    assert result["suppress_local_watcher"] is False
    assert result["fallback_active"] is False
    assert result["empty_probe_hz"] == 0.75
