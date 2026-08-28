# CM4 camera appliance v1

The CM4 remains an H.264 camera appliance. It does not own `FALL` authority.
Heavy bed segmentation, YOLO11m Pose, GRU and the video verifier remain on the
central RTX server.

## Runtime flow

```text
Camera -> H.264 hardware encode -> MediaMTX :8554 -> central server
                     |
                     +-> existing low-resolution watcher (5 FPS, one local decode)
                           |-> motion ratio / BURST hint
                           |-> scene relocation guard
                           |-> fixed ROI validity
                           +-> encoded ring health
                                      |
                                      +-> authenticated heartbeat/outbox -> :8020
```

The site runtime reuses the watcher's retained 320x180 RGB snapshot. It does
not open a second permanent RTSP connection. The encoded ring monitor only
inventories MediaMTX/ffmpeg segments and never decodes or deletes them.

## Installation

1. Put a normalized ROI profile at the path configured by
   `site_runtime_config.roi_profile_path`. Use
   `config/edge_roi_profile.example.json` as the contract.
2. With the camera fixed and the room in its accepted setup state, calibrate
   the scene once:

```bash
python scripts/calibrate_edge_scene.py \
  --config config/edge_node_bed_161_camera_appliance.json
```

3. Configure MediaMTX or the existing ffmpeg publisher to create short encoded
   segments under `site_runtime_config.encoded_ring.directory`. Segment
   creation must use stream copy; raw RGB recording or re-encoding is outside
   this design.
4. Install and start `dmc-pose-camera-appliance.service`.

## Safety behavior

- `scene_state=CHANGED` immediately makes `roi_state=DEGRADED`.
- Motion is a scheduler hint and never a fall decision.
- A missing/stale ring is telemetry only; it does not invent evidence.
- Network failure is handled by the existing durable SQLite outbox.
- Event-triggered YOLO11n Pose shadow may remain optional, but its authority is
  `INSUFFICIENT/telemetry`; central inference owns the final event.
- Pose/ONNX modules are lazy-loaded and are not shipped in the appliance-only
  handoff. Add them as a separate optional bundle only after a CM4 benchmark.
- Recalibration is explicit. The agent never silently accepts a moved camera as
  the new normal scene.

## Telemetry fields

The existing heartbeat additionally carries motion ratio, scene state/change
score, ROI source/version, and ring segment count/bytes/coverage. All new
fields have safe defaults so configurations without `site_runtime_config`
continue to operate.

`send_site_telemetry` defaults to `false` in the Pi configuration. In that
compatibility mode the agent strips only the new detailed fields before HTTP
delivery; existing `runtime_mode`, `watcher_fps`, `roi_state` and `roi_version`
remain available. Enable it only after the central strict contract has been
updated to the matching `edge_contract_v1.py`.
