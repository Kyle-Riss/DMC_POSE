# GRU real-feature diagnostic training

Status date: 2026-08-24

This run verifies the training and export machinery on real DMC 109D Pose
features. It is not a fall-performance result and cannot be promoted.

## Input

```text
78 USB staged-fall camera views
  -> deployed YOLO11m Pose + posture model
  -> observed-only 20 Hz, 109D rows
  -> uncalibrated motion boundary proposals
  -> 80-row causal windows, stride 5 rows
```

The proposal CSV remains separate from the human annotation CSV. No proposal
was persisted as ground truth.

Window inventory:

| Diagnostic split | Windows | Non-fall | Fall | Recordings |
|---|---:|---:|---:|---:|
| train | 928 | 109 | 819 | 16 |
| validation | 383 | 35 | 348 | 5 |
| test | 103 | 0 | 103 | 5 |

The split locks the three camera views by recording ID, but subject identity is
unknown. The test split also has no negative window. Metrics from this run do
not estimate precision, false-alert rate, or generalization.

## Machinery result

```text
architecture       gru_v1
input              80 x 109 @ 20 Hz
parameters         190,977
device             CUDA / RTX 5080
epochs completed   9
run purpose        smoke
promotion eligible false
```

The checkpoint trained, saved, reloaded, and exported successfully. The ONNX
artifact embeds train-split normalization and sigmoid probability conversion.
PyTorch versus ONNX maximum absolute error was `5.96e-08` over a three-sample
dynamic-batch parity check.

## Hard gate

The reported validation values are diagnostic only because labels are
uncalibrated proposals and people may cross diagnostic splits. Test ROC AUC is
explicitly `null` because only the fall class is present.

Production-candidate training still requires:

1. manual onset/impact/stable/end review;
2. subject and session identity;
3. subject-disjoint frozen splits;
4. native-20-Hz hospital hard negatives;
5. matched non-fall test exposure and valid bed-hours.

Artifacts:

- `external_datasets/windows/diagnostic/usb_proposal_gru_80x109_20hz_v1/window_index.json`
- `runs/temporal_gru/diagnostic_usb_proposals_20260824/report.json`
- `runs/temporal_gru/diagnostic_usb_proposals_20260824/model_diagnostic.onnx.json`
