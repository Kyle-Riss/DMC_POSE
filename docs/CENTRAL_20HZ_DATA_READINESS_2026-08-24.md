# Central 20 Hz data and compute readiness

This is a pipeline-readiness measurement, not a fall-accuracy report.

## Data inventory

The attached USB contains 78 simulated-fall videos: 26 each from `c1_sim`,
`c2_sim`, and `c3_sim`. Every video is readable at 640x360 and 20 FPS. Total
duration is 947.8 seconds.

The generated diagnostic manifest deliberately preserves only the source's
video-level fall label. It does not invent fall onset/end intervals and marks
every item `training_eligible=false` and `temporal_tcn_eligible=false`.

## 20 Hz extraction result

All 78 videos were processed with the deployed YOLO11m Pose hash and six-class
Keras hash under `observed_only_20hz_v1`.

| Slice | Pose observations / probes | Coverage | No-primary rate | Gap resets / 1000 probes |
|---|---:|---:|---:|---:|
| Overall | 17,187 / 18,680 | 92.01% | 7.99% | 13.60 |
| c1_sim | 5,760 / 6,220 | 92.60% | 7.40% | 14.31 |
| c2_sim | 5,588 / 6,220 | 89.84% | 10.16% | 15.60 |
| c3_sim | 5,839 / 6,240 | 93.57% | 6.43% | 10.90 |

Mean visible joints per accepted observation were 15.28 of 17. There were no
duplicate or non-monotonic timestamp skips. The c2 camera slice needs the first
visual failure review because it has the lowest coverage and highest gap rate.

## Central compute result

The sequential offline extractor processed 947.8 seconds of source in 522.28
seconds, or 1.815x real time. That path performs one Python prediction and one
Keras prediction per frame and is not the target scheduler.

A separate YOLO11m Pose micro-batch benchmark used six different 640x360 hospital
frames. Batch 6 achieved 351.86 frames/s mean with 21.20 ms p95 wall latency,
including Ultralytics preprocessing and postprocessing but excluding decode.
The 6-camera native target is 120 frames/s, leaving 2.93x measured Pose-only
headroom.

## Remaining accuracy gate

Precision, recall, false events/hour, and onset latency remain unavailable
because this USB set has no temporal fall boundaries and no matched normal-ADL
control set. The next valid accuracy step is:

1. annotate fall onset, impact, and recovery for the 78 videos;
2. identify subject/session groups before splitting;
3. connect a matched normal ADL and normal bed-exit set;
4. freeze train/validation/test groups;
5. build 20 Hz windows and train TCN/GRU/BiLSTM/Transformer on the same split;
6. report event metrics and camera-specific failure slices.

Artifacts:

- [`usb_sim_falldown_pose_quality_20260824.json`](usb_sim_falldown_pose_quality_20260824.json)
- [`central_pose_microbatch_benchmark_20260824.json`](central_pose_microbatch_benchmark_20260824.json)
- [`central_20hz_readiness_20260824.svg`](central_20hz_readiness_20260824.svg)
