"""Pure policy helpers for central cameras managed by a healthy edge node."""
from __future__ import annotations


def parse_camera_ids(value: str) -> frozenset[str]:
    """Parse a comma-separated camera allow-list without accepting blanks."""
    return frozenset(item.strip() for item in str(value).split(",") if item.strip())


def edge_runtime_policy(
    *,
    managed: bool,
    connected: bool,
    disconnected_for_sec: float,
    failover_grace_sec: float,
    normal_empty_probe_hz: float,
    managed_empty_probe_hz: float,
    result_fresh: bool = True,
) -> dict:
    """Decide local watcher suppression, failover, and EMPTY probe cadence."""
    if not managed:
        return {
            "suppress_local_watcher": False,
            "fallback_active": False,
            "empty_probe_hz": float(normal_empty_probe_hz),
        }
    signal_healthy = bool(connected) and bool(result_fresh)
    failover = (
        not signal_healthy
        and float(disconnected_for_sec) >= max(0.0, float(failover_grace_sec))
    )
    return {
        "suppress_local_watcher": not failover,
        "fallback_active": failover,
        "empty_probe_hz": float(
            normal_empty_probe_hz if failover else managed_empty_probe_hz
        ),
    }
