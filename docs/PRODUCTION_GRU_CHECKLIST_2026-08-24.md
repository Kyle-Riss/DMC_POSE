# DMC Pose single production GRU checklist

Status date: 2026-08-24

The target is one central fall model, not a research model tournament. Historical
TCN and external weights remain diagnostic evidence only and are not parallel
production routes.

## Frozen candidate

```text
six CM4 H.264/RTSP streams
  -> one central decode per camera
  -> shared YOLO11m Pose at native 20 Hz
  -> tracking
  -> observed-only 80 x 109 sequence per (camera_id, track_id)
  -> shared two-layer unidirectional GRU
  -> kinematic + bed relation + posture fusion
  -> SHADOW_ALERT
```

`observed-only` means no missing person/frame is manufactured. Missing joints in
a real Pose observation are zero-valued together with an explicit visibility
mask; this is not a synthetic temporal row.

## Execution checklist

| Step | State | Evidence / blocker |
|---|---|---|
| Freeze 20 Hz 109D input | DONE | central contract, extraction and cadence tests |
| Freeze one GRU architecture | DONE | `gru_v1`, 2 x 128, causal, `80 x 109` |
| Exercise GRU training path | SMOKE PASS | deterministic synthetic fixture; CUDA train/save/reload only; `promotion_eligible=false` and no accuracy claim |
| Exercise real-feature GRU path | DIAGNOSTIC PASS | 1,414 proposal-labelled USB windows; CUDA train + ONNX parity pass; subject unknown and test negatives zero, so no performance claim |
| Build manually reviewed shadow corpus | DONE / SHADOW ONLY | 1,039 windows at `80 x 109 @ 20 Hz`; 16 reviewed fall recordings and 4 explicit bed-exit hard negatives; multiview recording lock; subject identity unknown |
| Verify six-camera compute budget | PARTIAL PASS | Pose and temporal forward benchmarks pass; decode/NVDEC end-to-end remains |
| Audit external weights | DONE | FallVision contract mismatch; Wardy 80D source builder unavailable |
| Build external safe adapters | DONE | load/shape/scaler tests; diagnostic only |
| Audit egi03 Run-5 BiLSTM | DONE / REJECTED FOR WARM-START | verified 5 folds are `30 x 15 @ 15 Hz` bidirectional LSTM; DMC is causal `80 x 109 @ 20 Hz` GRU; license file absent |
| Integrate RealBiomFall 100 | DONE / REJECTED FOR POSE-GRU | archives and labels verified; 16.04% Pose coverage and zero valid 80-row windows; preserve for RGB verifier research only |
| Audit legacy AI_runner event clips | DONE / REPLAY ONLY | 3 normal exits + 3 simulated falls are about 9 Hz, not native 20 Hz; no frame duplication allowed |
| Acquire TsetFall | EXTERNAL GATE | form and decoding key required; no dataset or license assumption made |
| Prepare staged-clip review queue | DONE | 78 views grouped into 26 multiview recordings; no labels guessed |
| Audit staged media decode counts | DONE | all 78 views sequentially decoded; 45 container-count mismatches reconciled without changing 6 existing reviewed rows; 934.0 decoded seconds |
| Generate staged review proposals | DONE / HUMAN REVIEW REQUIRED | motion-only fixed heuristic for all 78 views; failures 0; proposals are not persisted as ground truth until explicit approval |
| Add multiview proposal consensus | DONE / HUMAN REVIEW REQUIRED | 13/26 recordings consistent under spread gates; 13/26 flagged for adjudication; one-click completion disabled for disagreements |
| Enforce reviewed manifest gate | DONE | complete positives require impact and identity for promotion manifests; excluded/adjudication dispositions remain explicit |
| Complete temporal labels | DONE | all 26 physical events (78/78 views) reviewed; 11 complete falls, 5 impact-occluded/gradual falls, 10 excluded no-fall/unusable recordings |
| Collect hospital hard negatives | PARTIAL | four explicitly reviewed normal bed-exit recordings included; blanket, caregiver, occlusion and pickup remain |
| Collect hospital staged positives | PARTIAL | 16 reviewed bed-fall/slide recordings included; broader people/sessions and collapse variants remain |
| Build leakage-safe Train/Val/Test | PROMOTION BLOCKED | recording-disjoint diagnostic split exists; authoritative person/session identity is absent, so subject-disjoint evidence does not |
| Train GRU candidate | DONE / TELEMETRY ONLY | `gru_v1`, 190,977 parameters, CUDA, reviewed staged shadow v2; `promotion_eligible=false` |
| Select threshold/persistence | DIAGNOSTIC ONLY | validation threshold 0.341508; persistence remains non-authoritative and Fusion consumption is disabled |
| Frozen Test event evaluation | FAILED PROMOTION GATE | window test AUC 0.2555, 108/135 non-fall windows positive at diagnostic threshold; no performance claim |
| ONNX export and parity test | DONE | embedded normalization+sigmoid; dynamic batch `80 x 109`; max PyTorch/ONNX error `5.96e-08` |
| Central six-camera shadow integration | DEPLOYED / TELEMETRY ONLY | six streams online, one RTSP connection per camera, model load verified, Fusion disabled |
| Live temporal readiness | FAILED / OPTIMIZATION STAGED | first physical floor-transition test produced zero temporal predictions because shared `Keras.predict()` limited Pose to roughly 9-15 Hz; direct inference replacement has exact output parity and awaits live cadence verification |
| Shadow valid-bed-hour soak | STARTED / NOT VALID FOR PROMOTION | first physical test exposed person/track loss during rapid floor transition; output remains telemetry only |
| Production ALERT promotion | NOT AUTHORIZED | requires explicit safety/authority review |

## Immediate next runtime milestone

Deploy the direct Keras inference optimization, verify at least 17 Hz live Pose
cadence and an `80/80` ready window, then repeat the controlled floor-transition
test while recording raw probability, candidate state, gap resets, and track
resets. The current checkpoint is useful for observing the central contract but
its diagnostic split does not demonstrate fall discrimination.

Manual boundary semantics are frozen in
`docs/TEMPORAL_LABEL_PROTOCOL_V1.md`. Sequential decode reconciliation is
recorded in `docs/usb_sim_falldown_decode_reconciliation_20260824.json`;
container header frame counts are not treated as ground truth.

RealBiomFall and egi03 do not bypass this gate. See
`docs/EXTERNAL_RESOURCE_AUDIT_2026-08-24.md` for the measured rejection reasons.

Build the non-overwriting queue and launch the existing local review UI:

```bash
python scripts/build_temporal_annotation_queue.py
python scripts/serve_fallvision_annotation.py \
  --annotations external_datasets/annotations/usb_sim_falldown_temporal_v1.csv \
  --port 8010
```

The UI name is historical; the CSV remains the USB staged dataset and is not
relabelled as FallVision. Re-running the queue builder refuses to overwrite
human work unless `--force` is explicitly supplied.

The identity template is generated separately because filenames do not prove
participant identity:

```bash
python scripts/build_reviewed_staged_manifest.py --write-identity-template
```

After explicit identity/split completion, the same script without that flag
creates a promotion-eligible input manifest. It fails closed if any boundary or
identity is missing, or if one subject crosses Train/Validation/Test. The
separate reviewed shadow builder deliberately cannot remove that promotion gate.
