import edge_node_agent
from edge_node_agent import EdgeNodeAgent


def config(tmp_path):
    return {
        "node_id": "rpi-bed-161",
        "camera_id": "bed_161",
        "server_url": "http://127.0.0.1:8020",
        "api_token": "x" * 32,
        "spool_path": str(tmp_path / "outbox.sqlite3"),
        "capabilities": {"camera_capture": True, "rtsp_publish": True},
    }


def test_heartbeat_reports_real_spool_and_no_model_claim(tmp_path):
    agent = EdgeNodeAgent(config(tmp_path))
    payload = agent.heartbeat()
    assert payload.node_id == "rpi-bed-161"
    assert payload.model_bundle_version is None
    assert payload.runtime_mode == "DEGRADED"
    assert payload.spool_depth == 0


def test_offline_cycle_keeps_heartbeat_in_durable_spool(tmp_path):
    agent = EdgeNodeAgent(config(tmp_path))
    agent._post = lambda endpoint, payload: False
    agent.writer.start()
    try:
        result = agent.cycle()
    finally:
        agent.writer.stop()
    assert result["failed"] == 1
    assert result["pending"] == 1


def test_runtime_state_only_accepts_public_fields(tmp_path):
    agent = EdgeNodeAgent(config(tmp_path))
    agent.update_runtime(
        capture_connected=True, capture_fps=20, watcher_fps=5,
        runtime_mode="EMPTY", roi_state="READY", roi_version=2,
    )
    heartbeat = agent.heartbeat()
    assert heartbeat.capture_connected
    assert heartbeat.roi_version == 2


def test_capture_health_uses_port_and_process_names(tmp_path, monkeypatch):
    value = config(tmp_path)
    value["capture_health"] = {
        "tcp_host": "127.0.0.1",
        "tcp_port": 8554,
        "required_processes": ["mediamtx", "rpicam-vid", "ffmpeg"],
    }
    monkeypatch.setattr(edge_node_agent, "tcp_port_open", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        edge_node_agent,
        "process_names",
        lambda: {"mediamtx", "rpicam-vid", "ffmpeg"},
    )
    agent = EdgeNodeAgent(value)
    assert agent.heartbeat().capture_connected is True

    monkeypatch.setattr(edge_node_agent, "process_names", lambda: {"mediamtx"})
    assert agent.heartbeat().capture_connected is False


def test_heartbeat_sequence_survives_agent_restart(tmp_path):
    first = EdgeNodeAgent(config(tmp_path))
    assert first.heartbeat().sequence == 0

    restarted = EdgeNodeAgent(config(tmp_path))
    assert restarted.heartbeat().sequence == 1


def test_motion_watcher_updates_public_runtime_state(tmp_path):
    class FakeWatcher:
        def status(self):
            return {"watcher_fps": 5.0, "burst_active": True}

    agent = EdgeNodeAgent(config(tmp_path))
    agent.motion_watcher = FakeWatcher()
    heartbeat = agent.heartbeat()
    assert heartbeat.watcher_fps == 5.0
    assert heartbeat.runtime_mode == "BURST"


def test_site_runtime_facts_are_added_to_heartbeat(tmp_path):
    class FakeSiteRuntime:
        def refresh(self):
            return {
                "motion_ratio": 0.2,
                "motion_active": True,
                "scene_state": "STABLE",
                "scene_change_score": 0.03,
                "roi_state": "READY",
                "roi_version": 4,
                "roi_source": "fixed_profile",
                "ring_buffer_ready": True,
                "ring_buffer_segments": 12,
                "ring_buffer_bytes": 4096,
                "ring_buffer_coverage_sec": 24.0,
            }

    agent = EdgeNodeAgent(config(tmp_path))
    agent.site_runtime = FakeSiteRuntime()
    heartbeat = agent.heartbeat()
    assert heartbeat.scene_state == "STABLE"
    assert heartbeat.roi_state == "READY"
    assert heartbeat.ring_buffer_ready is True
    assert heartbeat.motion_ratio == 0.2


def test_site_telemetry_is_stripped_for_legacy_server_by_default(tmp_path):
    agent = EdgeNodeAgent(config(tmp_path))
    payload = agent.heartbeat().model_dump(mode="json")
    wire = agent.wire_payload("/edge/heartbeat", payload)
    assert "scene_state" not in wire
    assert "ring_buffer_ready" not in wire
    assert wire["roi_state"] == "UNAVAILABLE"


def test_site_telemetry_can_be_enabled_after_server_upgrade(tmp_path):
    value = config(tmp_path)
    value["send_site_telemetry"] = True
    agent = EdgeNodeAgent(value)
    payload = agent.heartbeat().model_dump(mode="json")
    wire = agent.wire_payload("/edge/heartbeat", payload)
    assert wire["scene_state"] == "UNCALIBRATED"
    assert wire["ring_buffer_ready"] is False


def test_pose_shadow_result_uses_durable_authenticated_contract(tmp_path):
    agent = EdgeNodeAgent(config(tmp_path))
    agent.writer.start()
    try:
        payload = agent.queue_pose_shadow_result({
            "event_unix": 1786331029.0,
            "detection_count": 1,
            "best_person_score": 0.84,
            "best_visible_keypoints": 14,
            "snapshot_and_pose_ms": 186.0,
        })
        agent.writer.queue.join()
        pending = agent.outbox.due(limit=8)
    finally:
        agent.writer.stop()
    assert payload.frame_seq == 0
    assert payload.fusion_phase == "INSUFFICIENT"
    assert payload.temporal_ready is False
    assert "pose_shadow" in payload.evidence
    assert pending[0].endpoint == "/edge/results"


def test_pose_result_sequence_survives_agent_restart(tmp_path):
    first = EdgeNodeAgent(config(tmp_path))
    first.writer.start()
    try:
        first.queue_pose_shadow_result({
            "event_unix": 1786331029.0, "detection_count": 0,
            "best_person_score": 0.0, "best_visible_keypoints": 0,
            "snapshot_and_pose_ms": 180.0,
        })
    finally:
        first.writer.stop()
    restarted = EdgeNodeAgent(config(tmp_path))
    assert restarted.pose_result_sequence == 1
