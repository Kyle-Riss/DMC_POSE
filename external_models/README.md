# External temporal baselines

This directory is a local, read-only vendor area. Third-party repositories and
weights are not copied into DMC runtime code and are not committed by this
repository. DMC-owned adapters live outside this directory.

## Locally pinned artifacts (2026-08-24)

| Baseline | Local directory | Pinned revision | Runtime status |
|---|---|---|---|
| FallVision posture-aware fall detection | `posture-aware-fall-detection/` | `68b177a31300eecc76fdf5a51b7eac2c4f117263` | quarantined pending contract reconstruction |
| Wardy M-04 GRU | `wardy-m4-fall/` | `df40ae0310d2ccd38fb39a0a7311a7df420fe4b7` | offline baseline only |

The Wardy `model.onnx` was resolved from Git LFS and verified against the
SHA-256 in its upstream `manifest.json`:

```text
5ba44fbd8dde53457da0ad063098fa8194454b2d8f7599a8acc808e919bff63f
```

Do not install either repository's requirements into the `pose-cuda`
environment. Do not load third-party pickle scalers in the central service.

## Admission rules

An external model enters an offline comparison only after these are explicit
and tested: tensor shape/dtype, joint order, missing-joint policy, coordinate
normalization, cadence/window duration, tracking/gap/padding policy, output
semantics, threshold, license boundary, and artifact hashes.

Promotion to central shadow additionally requires a DMC-owned adapter,
subject/session-safe holdout results, false events/hour, event recall, latency,
and a fail-closed contract check. No external model has alert authority.

## Known findings

### FallVision

- Shipped fall weights accept `60 x 24`; posture weights accept `30 x 20`.
- The checked-in fall training source declares `30 x 20`, conflicting with the
  shipped fall weights and README.
- The README says BiLSTM, but the inspected HDF5 graph and training source use
  two ordinary unidirectional LSTM layers.
- Coordinates are translated around the mid-hip in pixel units and transformed
  by a pickled `StandardScaler`; the source has no body-scale normalization.
- The inference default is 30 rows, so it cannot drive the shipped 60-row fall
  weights as written.

FallVision stays quarantined until training/FPS provenance is reconstructed or
the network is retrained under a DMC-owned contract.

### Wardy

- Input: float tensor `batch x 20 x 80`.
- Declared cadence: 10 Hz; window: 2 seconds; threshold: 0.5.
- The final 12 motion/geometry features are not a slice of DMC's 109D vector.
- A separately implemented and verified 80D builder is required. Direct
  truncation or index copying from DMC 109D is forbidden.
- Its URFD-derived license boundary is compatible with non-commercial research,
  so this artifact is retained as an offline external baseline.
