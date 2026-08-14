# DMC_POSE temporal fall-detection strategy

## 1. Decision

The deployed six-class Keras model remains a **single-frame posture model** with
input shape `(34,)`. It is not a temporal model and must never receive a
flattened multi-frame buffer.

The temporal component is a separate event model:

```text
video / RTSP at 10 Hz
  -> YOLO COCO-17 pose (XY + confidence)
  -> shared offline/live feature adapter
  -> 3-second temporal window (30 samples)
  -> TCN fall-event probability
  -> bed/rail/rule score fusion
  -> persistence + cooldown
  -> alert
```

Initial TCN target is binary `fall` versus `non_fall`. The existing six posture
probabilities are input features, not the TCN target. Hospital-specific states
such as safe bed exit, unsafe bed exit, and post-fall immobility are added only
after the own-data label set is ready.

## 2. Why the inputs are separated

Public fall datasets do not consistently contain bed segmentation, rail state,
or calibrated floor ROI. Therefore:

- **Core temporal features:** normalized pose, confidence, visibility, velocity,
  torso angle/change, hip vertical motion, six-class posture probabilities.
- **Hospital context:** bed attachment, overflow, body-in-bed ratio, rail state,
  floor ROI and camera-specific geometry.

The TCN is pretrained with core features. Hospital context is fused by the rule
and scoring layer, then optionally used during own-data fine-tuning. Missing
context must be represented by an explicit availability mask, never by a value
that can be confused with a real zero.

## 3. Data contract

### Manifest (`temporal_manifest_v1`)

Each video entry contains:

- stable `video_id`, dataset and subject ID;
- subject-disjoint `split`;
- absolute source video path;
- FPS, frame count, dimensions and duration;
- one or more `{label, start_sec, end_sec}` intervals;
- source description and provenance.

### 10 Hz frame table

Required columns:

```text
video_id, dataset, subject_id, split,
frame_idx, timestamp_sec, sample_hz,
person_detected,
kpt_0_x, kpt_0_y, kpt_0_conf, ... kpt_16_conf,
pose_prob_0, ... pose_prob_5,
feature_schema_version
```

Coordinates used by the temporal model must be normalized in the shared feature
adapter. Raw pixel coordinates are retained for debugging only.

### Window table

Default window is 3.0 seconds / 30 samples with stride 0.5 seconds / 5 samples.
A window is positive when it overlaps a fall interval according to a documented
policy. Boundary windows are retained with overlap metadata; they are not
silently assigned to the negative class.

## 4. Split policy and leakage prevention

- GMDCSA-24: subjects 1-2 train, subject 3 validation, subject 4 test.
- No frames or windows from one video may cross splits.
- Later datasets are also split by subject, never randomly by frame.
- Final hospital evaluation uses camera/day/person groups that were not used for
  threshold tuning.
- Public pretraining and own-data fine-tuning results are reported separately.

With only four GMDCSA subjects, the fixed split is for pipeline validation. A
leave-one-subject-out report is required before treating its metric as a model
quality claim.

## 5. Execution phases and gates

### Phase 0 — stabilize and inventory

- Correct live buffer data shapes and preserve the 34-D frame classifier.
- Verify model paths and server launch paths use `/home/dmc/AI/DMC_POSE`.
- Build manifests for downloaded data.
- Treat an unreadable archive as incomplete; never extract or train from it.

Gate: Python compile passes, buffer shape tests pass, and manifest has no missing
videos or unparsed labels.

### Phase 1 — extraction v2

- Update extraction to accept a manifest rather than requiring copied videos.
- Read true video FPS; create deterministic 10 Hz sample timestamps.
- Store XY and confidence for the selected person.
- Store all six frame-class probabilities, not only argmax.
- Record extractor/model hashes and failure reasons.

Gate: one fall and one ADL video reproduce identical features on repeated runs;
row spacing is 0.1 seconds within tolerance.

### Phase 2 — shared feature adapter

- Implement body-centred, scale-normalized pose features.
- Add confidence/visibility masks and timestamp-aware velocities.
- Use the exact adapter in batch training and live inference.
- Test translation/scale invariance and missing-keypoint behavior.

Gate: offline replay and live adapter outputs match for the same frames.

### Phase 3 — baseline before TCN

Evaluate in this order:

1. existing rules only;
2. logistic/gradient-boosted window baseline;
3. small causal TCN;
4. GRU only if TCN does not meet latency/accuracy needs.

This order checks whether added model complexity provides real value.

Gate metrics are event sensitivity, false alerts/hour, event precision,
detection latency, and person-missing rate. Frame accuracy alone is not a gate.

### Phase 4 — FallVision validation

- Validate archive integrity before extraction.
- Build its manifest and map only supported intervals.
- Re-run the same pose extractor; do not mix vendor landmarks into training.
- Use it for external generalization, then decide whether to include it in
  training.

Gate: cross-dataset result is reported separately from in-dataset validation.

### Phase 5 — own hospital data

Required labels:

```text
normal_in_bed, normal_exit, unsafe_exit,
fall_transition, post_fall_immobile, hard_negative, unknown
```

Label event start/end plus person/camera/day and quality flags. `unknown` regions
are excluded from loss. Hard negatives must include blanket motion, sitting at
the edge, kneeling, picking objects, caregiver interaction, occlusion and empty
bed movement.

Gate: held-out day/camera test meets the agreed sensitivity and false-alert/hour
target. Thresholds are calibrated on validation only.

### Phase 6 — live shadow deployment

- Load the TCN as a separate artifact with schema/version validation.
- Keep per-camera 30-sample buffers.
- Run causal inference at 10 Hz without future frames.
- Publish probability and reason fields without sending alerts first.
- Compare shadow predictions with operator-reviewed events.
- Enable alerts only after rollback and health checks are verified.

## 6. Current local findings (2026-07-31)

- GMDCSA-24: 160 videos and 8 annotation CSV files are present.
- FallVision file is about 30 GB but Python reports `BadZipFile`; it is currently
  considered incomplete/corrupt until integrity is restored.
- No `pose-action*.mp4` was found under `/home/dmc`; only
  `/home/dmc/AI/pose/extracted_frames` and `/home/dmc/extracted_frames` were
  found. The original own-video source must be located or re-copied before
  temporal ground-truth preparation.
- The repository already has extraction/enrichment/rule scripts, but several
  defaults still point to `/home/dmc/pose-sixclass` or `/home/dmc/Dataset` and
  must be migrated before batch execution.


## 7. First baseline result (2026-07-31)

GMDCSA-24 was fully processed with the subject-disjoint policy:

- 160 videos, 12,959 sampled rows, 109 temporal input features;
- 887 train, 458 validation and 384 test causal windows;
- person detection rate: 97.97% to 98.89% by split;
- logistic test: ROC-AUC 0.8427, precision 0.3365, recall 0.8987;
- causal TCN test: ROC-AUC 0.9007, precision 0.5197, recall 0.8354;
- causal TCN event test: 14/15 falls detected, event precision 0.6364,
  median latency 1.55 seconds, 8 false events in 37 short clips.

The TCN is a useful research baseline but **not deployment-ready**. It remains an
offline/shadow artifact until external data and long-duration hospital hard
negatives reduce false events.

The local extracted-frame data contains 153 sequences and 28,556 labelled rows
across 12 stable bed postures. All rows have `is_transition=false`, so it cannot
supply positive fall-transition supervision. It is reserved for long-duration
negative evaluation once original videos or a valid resampling policy exists.

IBM Granite TTM is not selected as the primary model because its released
pretraining and tutorials target multivariate forecasting, while this pipeline
requires short-horizon causal event classification. Classification-oriented
TSFM models may be benchmarked later, but only against the same subject splits
and event metrics.
