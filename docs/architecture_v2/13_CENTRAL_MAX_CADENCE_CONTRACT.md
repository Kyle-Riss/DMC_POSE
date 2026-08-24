# Central max-cadence inference contract

Status: target architecture; no live cutover has been performed.

## Fixed boundary

```mermaid
flowchart TD
    C[CM4 camera<br/>H.264 640x360 at 20 FPS] -->|one RTSP/TCP session| H[Central camera hub]
    H --> D[one decoder per camera]
    D --> L[latest decoded frame]
    L --> V[viewer overlay]
    L --> R[rolling evidence buffer]
    L --> S[deadline scheduler]
    S -->|dynamic micro-batch<br/>up to source cadence| P[shared YOLO11m Pose]
    P --> T[multi-person tracking]
    T --> K[(camera_id, track_id) state]
    K --> F109[DMC 109D feature]
    F109 -->|observed-only 10 Hz| OLD[legacy DMC TCN shadow]
    F109 -->|observed-only 20 Hz<br/>80 rows / 4 s| NEW[single DMC GRU v1<br/>production candidate]
    K -->|dedicated 80D adapter<br/>10 Hz| W[Wardy offline baseline]
    K -. blocked: source contract mismatch .-> FV[FallVision quarantine]
    OLD --> U[fusion candidate]
    NEW --> U
    U -->|candidate| R
    R --> HV[heavy video VERIFY]
    HV --> O[BED_EXIT / FALL / BED_EXIT_FALL]
    O --> A[SHADOW_ALERT]
```

## Cadence means real time, not row count

Central Pose consumes the newest real frame at up to the camera's native 20
FPS. It does not manufacture duplicate observations when overloaded. Each
temporal model receives a timestamp-aware route matching its training contract:

| Route | Input | Time contract | Use |
|---|---:|---:|---|
| DMC TCN v1 | `30 x 109` | observed-only 10 Hz, 3 s | legacy shadow baseline |
| Wardy M-04 | `20 x 80` | 10 Hz, 2 s | offline baseline only |
| FallVision fall model | `60 x 24` | unresolved | quarantined |
| DMC Pose GRU v1 | `80 x 109` | observed-only 20 Hz, 4 s | single production candidate; shadow-only |

Running Pose at 20 FPS does not authorize feeding 20 FPS rows to a model trained
at 10 Hz. Downsampling one route must not reduce the shared Pose rate used by
tracking, kinematics, or another temporal route.

The selected candidate is a two-layer, unidirectional GRU with hidden width 128.
It is causal and uses the existing `pose_temporal_109_v1` feature schema. Missing
people or missed frames reset history and never create a copied/zero time row.
An individual missing joint remains zero-valued only inside a real observation
and is explicitly identified by its visibility feature. Threshold and temporal
persistence are deliberately unset until validation data can select them.

## Resource ownership

```text
RTSP connection       exactly 1 per camera
decoder               exactly 1 per camera
latest RGB frame      exactly 1 current slot per camera
model weights         exactly 1 shared instance per model
temporal state        one per (camera_id, track_id, model_route)
viewer/model/recorder consumers never reopen RTSP
```

The scheduler is deadline-aware and latest-only. Under overload it drops stale
work and never grows an unbounded frame queue. Candidate and VERIFY work has
higher priority than routine overlay or archival work.

## Evidence and state

The camera hub owns one latest RGB frame, a bounded sampled/JPEG tensor ring for
immediate model input, and optionally a compressed H.264 packet ring for event
export. It does not retain ten seconds of decoded RGB for every camera by
default.

At candidate time `t`, VERIFY immediately consumes `[t-4 s, t]`, then may update
at `t+0.5 s` and `t+1.0 s`. It does not wait for the full evidence window before
its first decision.

Every `(camera_id, track_id, model_route)` reports `WARMING`, `READY`,
`GAP_GRACE`, `RESET`, or `EXPIRED`. Track switch, non-monotonic timestamp, or an
excessive route-specific gap invalidates that route. Missing people never create
zero/copied person rows. Contract mismatch fails closed.

## Deployment boundary

Development and evaluation live in `/home/dmc/AI/DMC_POSE_source`. The existing
operator surface in `/home/dmc/AI/DMC_POSE` remains the management entry point.
Cutover later attaches a versioned central runtime behind that surface; this
contract does not modify or stop current services.

The CM4 target after cutover is capture, hardware H.264 encode, and RTSP publish
only. Existing Pi AI and Edge remain active until central replay, six-camera load,
and soak gates pass, then are removed in a separate reversible cutover.

## Promotion gates

1. One RTSP/decode owner per camera is proven from sockets and process trees.
2. Six-camera achieved Pose Hz, queue age p95, inference p95, VRAM, and decode
   age p95 meet the measured deadline at source cadence.
3. The GPU driver/NVML mismatch is resolved and NVDEC use is measured.
4. Subject/session-safe hospital holdout evaluation reports event recall,
   precision, false events/hour, and onset/impact latency.
5. Bed fall, slow slide, normal bed exit, caregiver interaction, and occlusion
   are reported separately.
6. Central shadow runs beside the current path before Pi AI and Edge removal.
7. External output remains `SHADOW_ALERT` until explicit authority review.

## Initial temporal compute measurement

The synthetic six-track benchmark on the central RTX 5080 is recorded in
[`central_temporal_compute_benchmark_20260824.json`](../central_temporal_compute_benchmark_20260824.json)
and visualized in
[`central_temporal_compute_benchmark_20260824.svg`](../central_temporal_compute_benchmark_20260824.svg).
It measures model forward cost only, not accuracy or end-to-end camera latency.

The first 20 Hz USB extraction and Pose micro-batch measurements are documented
in [`CENTRAL_20HZ_DATA_READINESS_2026-08-24.md`](../CENTRAL_20HZ_DATA_READINESS_2026-08-24.md).
