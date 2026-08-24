# External fall resources audit

Status date: 2026-08-24

This audit decides whether an external resource may enter the DMC central
production GRU path. It does not treat a repository's reported metrics as DMC
fall performance.

## Decision table

| Resource | Acquired / verified | DMC production-GRU use | Decision |
|---|---|---|---|
| egi03 `ai-fall-detection-system` Run-5 | release archive SHA-256 verified; 5 folds safely loaded | incompatible input and cadence | diagnostic quarantine only |
| RealBiomFall 100 v1 | both official MD5 values verified; 100 videos/labels readable | zero valid 4-second observed-only windows | RGB verifier research only |
| TsetFall | metadata repository inspected | dataset requires form and decoding key | not acquired; no training claim |
| Wardy M-04 | external adapter contract audited earlier | URFD-derived license and 80D builder constraints | diagnostic only |

## egi03 checkpoint contract

Local source commit:

```text
2f305fa7d2183181aa1cc87413e46a5e09c79a8a
```

Verified release archive:

```text
run5_stride2_aug.zip
SHA-256 1f6e654683896d1bfde3a9411e2f74de49508301790a7fa48e3e8333febf5281
```

All five checkpoints were loaded with PyTorch `weights_only=True`. Their actual
state dictionaries require 15 features and include reverse-direction recurrent
weights. The input projection is `[512, 15]`, and the classifier projection is
`[2, 256]`; this is a two-class bidirectional LSTM with hidden size 128. The
normalization arrays are `[1, 1, 15]`.

The external runtime contract is approximately:

```text
30 rows x 15 biomechanical features @ 15 Hz
  -> bidirectional LSTM
  -> two-class output
```

The frozen DMC candidate is:

```text
80 rows x 109 observed Pose features @ 20 Hz
  -> causal unidirectional GRU
  -> fall probability
```

Weights cannot be warm-started across these contracts. The checkout also has no
`LICENSE` file even though the README badge says MIT, so product reuse remains
license-unresolved. Its documentation says mean pooling while the checked source
uses the final recurrent state; source and checkpoint shapes are treated as the
stronger evidence.

Official source: <https://github.com/egi03/ai-fall-detection-system>

## RealBiomFall integration result

Official archives:

```text
labels-100.zip
MD5 65a7c3b8346e9cff8497b5bdd5c80372

video_clips-trimmed_cropped_padded_resized-100.zip
MD5 1168284537004b7937ec552015461719
```

The manifest loader is fail-closed and uses a restricted unpickler. It allows
only the two NumPy constructors observed in the verified semantic annotation;
unexpected pickle globals are rejected.

Dataset audit:

- 100/100 videos readable, total 551.5 seconds;
- fall onset present for 100, lowest-position/impact present for 97;
- 11 clips contain at least four seconds before annotated onset;
- only six source videos produced all 100 clips;
- four source videos occur in both official training and testing subsets.

The official 66/34 split is therefore not accepted as DMC promotion evidence.
The generated manifest marks every item as train augmentation only and
`promotion_metric_eligible=false`.

Deployed DMC Pose/Posture extraction at 20 Hz produced:

| Measurement | Result |
|---|---:|
| Pose rows / probes | 1,769 / 11,026 |
| Observation coverage | 16.04% |
| No-primary rate | 72.59% |
| Empty Pose CSVs | 10 / 100 |
| Mean visible joints | 13.82 / 17 |
| Valid causal `80 x 109` windows | **0** |

The result is not repaired by zero-fill, pose copying, or relaxing the frozen
gap contract. RealBiomFall is excluded from production-GRU training and retained
only as a possible RGB verifier dataset. It contains no matched normal-ADL
control and cannot establish precision or false-alert rate by itself.

Official source: <https://zenodo.org/records/11620083>

Artifacts:

- `external_datasets/manifests/realbiomfall_100_train_augmentation_v1.json`
- `docs/realbiomfall_pose_quality_20260824.json`
- `external_datasets/windows/pose_gru_109_observed_only_20hz/realbiomfall_100_v1_4s/window_index.json`

## Consequence

No external checkpoint or dataset audited here removes the present data gate.
The valid next production milestone remains reviewed hospital-domain staged
falls plus matched hard negatives under the frozen 20 Hz observed-only contract.
