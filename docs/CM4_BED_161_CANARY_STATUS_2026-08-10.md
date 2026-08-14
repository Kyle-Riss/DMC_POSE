# CM4 bed_161 canary status — 2026-08-10

## Outcome

- Device: Raspberry Pi Compute Module 4 Rev 1.1, aarch64, 4 cores, 4 GB RAM
- Camera pipeline preserved: `mediamtx`, `rpicam-vid`, and `ffmpeg` remain active; TCP 8554 is listening
- Edge heartbeat: enabled as `dmc-pose-edge-agent.service` on `192.168.0.161`
- Central control API: enabled as user service `dmc-pose-edge-control.service` on `192.168.0.108:8020`
- Authentication: SSH public key and bearer-token file; token values are not stored in repository configuration
- Latest verified state: `capture_connected=true`, durable outbox pending/retrying both zero

## Deployment boundary

The canary now runs a low-cost motion watcher and a local-only Pose shadow:

- `model_bundle_version=null` remains intentional: the bundle is not promoted
- motion watcher: active at 5 Hz using the existing local RTSP relay
- retained watcher frame: 320×180 RGB; motion comparison: 160×90 grayscale
- Pose shadow: one YOLO11n Pose inference per new motion burst, 90° clockwise rotation
- automatic bed ROI: disabled on Pi; the central server owns automatic ROI today
- temporal/fusion inference and upstream event upload: disabled

The Pose result is written only under the Pi state directory. The node still advertises `pose_inference=false`, because shadow validation is not the same as production capability.

## Installed paths

- Pi agent: `/home/moredigm/dmc_pose_edge_canary`
- Pi token: `/home/moredigm/.config/dmc_pose/edge_api_token` (mode 600)
- Pi outbox: `/home/moredigm/.local/state/dmc_pose/outbox.sqlite3`
- Pi sequence state: `/home/moredigm/.local/state/dmc_pose/heartbeat_sequence`
- Pi Pose shadow status: `/home/moredigm/.local/state/dmc_pose/pose_shadow_status.json`
- Pi Pose shadow events: `/home/moredigm/.local/state/dmc_pose/pose_shadow_events.jsonl`
- Pre-cleanup outbox backup: `/home/moredigm/.local/state/dmc_pose/outbox.pre_sequence_fix.sqlite3`
- Central node config: `config/edge_node_bed_161_cm4_canary.json`

## Verification commands

```bash
ssh -i /home/dmc/.ssh/id_ed25519_AI-server-ubuntu moredigm@192.168.0.161 \
  'sudo systemctl status dmc-pose-edge-agent.service --no-pager'

systemctl --user status dmc-pose-edge-control.service --no-pager
```

## Live motion-to-Pose gate

Physical validation on camera `bed_161` passed on 2026-08-10:

| Run | Person score | Visible keypoints | Pose time | Trigger-to-result |
|---|---:|---:|---:|---:|
| Cold/model-load run | 0.805 | 17/17 | 297 ms | 630 ms |
| Warm run | 0.844 | 14/17 | 184 ms | 186 ms |

An empty-bed frame produced zero detections at threshold `0.25`, both before and after the required 90° rotation. After the live runs the service remained active with `NRestarts=0`, outbox depth zero, temperature 55.5°C, and throttling `0x0`.

The earlier one-shot ffmpeg design took 3.22 seconds because a fresh RTSP reader waited for stream startup/keyframe. It was rejected. Pose now reuses the RGB frame already held by the 5 Hz motion watcher, removing the second RTSP connection from the hot path.

A subsequent 120-second empty-room soak produced zero new Pose wakes. The service remained active with zero restarts; agent CPU was 3.3%, watcher ffmpeg CPU 10.0%, temperature 55.5°C, and throttling `0x0`.

Authenticated result upload was then enabled in shadow mode. Four monotonic results (`frame_seq` 0–3) reached central `:8020/edge/results`, were persisted in `runtime_data/edge_control/edge_control.jsonl`, and left Pi outbox depth at zero. Results explicitly carry `temporal_ready=false`, `fusion_phase=INSUFFICIENT`, and shadow evidence, so they cannot raise an alert.

## Central scheduler wake bridge

The authenticated edge-result bridge was physically validated on 2026-08-10. The Pi result is used only as a fresh person-presence/wake hint; it is never used as temporal or fusion evidence.

| Time (KST) | Observed transition |
|---|---|
| 13:05:56 | Central changed `EMPTY -> OCCUPIED`, `person_count=1` |
| 13:05:57 | Local motion changed the central runtime to `BURST`; central Pose reported `sitting_edge` |
| 13:05:59 | Pi result advanced `frame_seq=14 -> 15`, `person_present=true`, confidence/quality `0.8413` |
| 13:05:59 | `edge_signal_wake_active=true`; central scheduler ran at priority `P0` |
| 13:06:03 | The four-second edge-result TTL expired and edge wake automatically disabled |
| 13:06:17 | Pi emitted `frame_seq=16`, `person_present=false`; it did not wake the scheduler |
| 13:06:29 | After exit, central returned to `EMPTY`, priority `P3`, fusion risk `0` |

This validates the intended boundary: the CM4 cheaply watches motion and reports a person hint, while the central YOLO11m Pose, bed geometry, TCN, and fusion remain authoritative. A fresh edge hint raises analysis cadence only. It does not raise risk by itself.

## Next gate

Keep Pose shadow-only and run a longer occupied/empty false-wake soak. Roll the same watcher/result contract to the remaining Pi devices after SSH access is established. Do not feed YOLO11n keypoints into TCN: the current temporal training/live contract uses YOLO11m Pose, so the 109-D feature distribution is not proven compatible.

## CPU artifact benchmark

The `cm4-onnx-candidate-v1` bundle was staged with target `rpi4` and status `benchmark_required`. Checksums matched before loading. Results use synthetic tensors, ONNX Runtime CPUExecutionProvider, and exclude decode/preprocessing/postprocessing.

| Artifact | Mean | P95 | Approx. throughput |
|---|---:|---:|---:|
| Bed segmentation ONNX, 320×320 | 209.12 ms | 239.12 ms | 4.78 FPS |
| YOLO11n pose ONNX, 320×320 | 173.88 ms | 202.38 ms | 5.75 FPS |
| TCN ONNX, 30×109 | 0.79 ms | 1.33 ms | 1264.65 windows/s |
| Six-class posture TFLite, 1×34 | 0.027 ms | 0.027 ms | 36701.90 samples/s |

Temperature rose from 55.0°C to 65.7°C during the 30-iteration ONNX run, with `get_throttled=0x0` before and after. TFLite requires NumPy 1.26.4 with the installed `tflite-runtime==2.14.0` ABI.

Raw reports:

- `runs/edge_benchmarks/cm4_bed_161/cm4_onnx_benchmark_2026-08-10.json`
- `runs/edge_benchmarks/cm4_bed_161/cm4_tflite_benchmark_2026-08-10.json`

Decision: keep bed segmentation at bootstrap/slow refresh cadence, keep pose asleep while empty, and use motion-triggered burst inference. TCN and posture costs are negligible relative to pose and segmentation. Pose is active only as a local shadow; production inference remains disabled.


### JPEG preprocessing included

Using an existing 640×360 `bed_161` JPEG, decode + resize + normalize averaged 12.92 ms. With both ONNX sessions resident, bed segmentation averaged 276.41 ms, pose averaged 260.40 ms, and sequential bed + pose on one frame averaged 554.36 ms (1.80 FPS), excluding YOLO postprocessing. Temperature reached 68.6°C with no throttling.

This confirms the production scheduler must never run bed segmentation and pose continuously together on the CM4. Bed ROI should be cached and refreshed slowly; pose should run only after a cheap motion/presence watcher wakes the pipeline. Raw report: `runs/edge_benchmarks/cm4_bed_161/cm4_preprocess_onnx_benchmark_2026-08-10.json`.

## Central edge-managed power policy

The central server now treats `bed_161` as an edge-managed camera. This changes compute cadence without changing the live viewer path:

```text
Pi RTSP relay ------------------------> central capture/viewer (~20 FPS)
Pi motion + YOLO11n person hint ------> authenticated edge result
fresh person hint --------------------> central burst/P1 scheduling
central YOLO11m Pose -----------------> 109-D observed-only TCN input
central bed ROI + Pose + TCN ---------> authoritative fusion result
```

While the edge heartbeat/result channel is healthy, the central low-resolution motion watcher is stopped and the empty-room YOLO11m probe is reduced from `0.75 Hz` to `0.05 Hz`. A 45-second empty-room measurement produced two Pose calls, or `0.0444 Hz`: a **94.07% reduction** from the legacy empty cadence. The RTSP capture remained approximately `20 FPS`, so the viewer remains responsive.

The policy is fail-open for detection. Stopping the secure edge-control service for longer than the configured three-second grace period produced:

| State | Central watcher | Empty Pose probe | Result |
|---|---:|---:|---|
| Edge healthy | 0 FPS / stopped | 0.05 Hz | edge-managed savings active |
| Edge unavailable | 19.34 FPS / running | 0.75 Hz | central fallback active |
| Edge recovered | 0 FPS / stopped | 0.05 Hz | savings restored automatically |

No restart of the camera pipeline was needed during failover or recovery.

## Physical edge-only wake validation

With the central watcher still stopped, a person entered and sat on the bed. At 14:24:38 KST, edge result `frame_seq=17` reported confidence `0.825462`. The central server entered `BURST`, detected one person with its own YOLO11m model, classified `sitting_edge`, set `in_bed=YES`, and scheduled priority `P1`. When the edge result expired, central occupied tracking continued from central evidence rather than trusting stale edge state.

After the person exited, the system returned to `EMPTY` at 14:25:24 KST with `person_count=0`, priority `P3`, and fusion risk `0`. The central watcher remained stopped throughout. This verifies the intended automatic chain: cheap edge wake, authoritative central confirmation, and automatic return to the low-power empty cadence.

After the final code restart, `bed_161` reported `capture_connected=true`, capture `19.97 FPS`, watcher `0 FPS`, edge connected, fallback disabled, effective empty probe `0.05 Hz`, and runtime `EMPTY`. Stale edge person state is now exposed safely as `edge_signal_person_present=false`; the raw historical value is retained separately as `edge_signal_last_person_present` for diagnostics.

Raw acceptance artifact: `runs/edge_benchmarks/cm4_bed_161/central_edge_managed_bed_161_2026-08-10.json`.

Decision: `bed_161` passes the single-camera edge-managed canary. Keep TCN in shadow and keep YOLO11n output limited to wake/person-presence hints. Rollout to other beds requires Pi SSH/service access and the same per-camera physical acceptance test.

## Heartbeat outbox compaction fix

A later 120-second heartbeat soak found a control-plane defect that did not affect RTSP capture or live inference: all 24 samples were connected and the Pi watcher averaged `5.002 FPS`, but `spool_depth` remained at 22. Inspection showed that these were old heartbeat snapshots rejected by the central monotonic sequence gate after newer heartbeats had already arrived. The durable sender treated the replaceable snapshots like events and retried them indefinitely.

`EdgeOutbox.enqueue()` now retains only the latest `/edge/heartbeat` snapshot. Identical messages remain idempotent. `/edge/results` and event messages are not compacted and retain durable retry behavior. Twenty targeted outbox, agent, signal, and edge-managed tests passed before deployment.

The change was backed up and deployed to `bed_161`; the deployed SHA-256 is `09fdf97f35bb8c994d6fd04e684cc8aa16b557a90746c8bc05348200caa408ae`. After restart, pending/retrying changed from `22/22` to `0/0`. A 30-second post-fix soak collected six unique heartbeat sequences with zero errors, capture connected for every sample, mean watcher `5.008 FPS`, and maximum spool depth `0`.

Soak artifacts:

- `runs/edge_benchmarks/cm4_bed_161/bed_161_edge_soak_120s_2026-08-10.json` — defect discovery
- `runs/edge_benchmarks/cm4_bed_161/bed_161_edge_soak_post_outbox_fix_30s_2026-08-10.json` — post-fix acceptance

## Post-fix physical enter/exit cycle

A 120-second physical cycle after the outbox deployment collected 240 central/edge samples with zero polling errors. All automated transport and state checks passed:

- edge result sequence advanced `19 → 24`
- fresh edge person result observed
- central YOLO11m observed one person
- runtime traversed `EMPTY → OCCUPIED/BURST → EMPTY`
- `sitting_edge` and `in_bed=YES` were observed
- TCN obtained a real observed-only ready window
- central watcher remained suppressed at 0 FPS
- edge fallback never activated
- maximum Pi spool depth remained zero

This was a normal enter/sit/exit scenario, not a fall. The feature audit exposed why TCN remains shadow-only: TCN probability peaked at `0.9993` and produced 56 candidate samples during normal sitting. Structural fusion nevertheless stayed within `NO_PERSON`, `INSUFFICIENT`, `WARMING`, and `SAFE`; it never entered `CANDIDATE`, `VERIFY`, or `SHADOW_ALERT`, and frame fall score remained `0`. The observed maximum fusion risk of `0.5166` is retained as a calibration signal, not treated as an alert.

Artifacts:

- `runs/edge_benchmarks/cm4_bed_161/bed_161_physical_cycle_post_outbox_fix_2026-08-10.json`
- `runs/edge_benchmarks/cm4_bed_161/bed_161_physical_cycle_feature_audit_2026-08-10.json`

Decision: Edge transport, wake scheduling, central confirmation, TCN readiness, fusion suppression, EMPTY return, and durable delivery pass on `bed_161`. TCN standalone promotion remains blocked.
