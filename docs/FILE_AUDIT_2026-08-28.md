# DMC POSE file audit — 2026-08-28

## Scope and recovery

Audited paths:

- `/home/dmc/AI/DMC_POSE_source`
- `/home/dmc/AI/DMC_POSE`

The source tree had 1,803 files and 3,562,315,227 bytes before cleanup. The
management tree had 17 files. Full pre-change hashes and a targeted recovery
archive are stored outside both trees:

```text
/home/dmc/AI/.DMC_POSE_cleanup_audit_20260828T081204Z/
  source-before.sha256       1,803 entries
  management-before.sha256      17 entries
  recovery-before-cleanup.tar.gz
  source-after.sha256        1,589 entries
  management-after.sha256       10 entries
```

Recovery archive SHA-256:

```text
7edd8cbcb35510212714d3c0f7b912a7e93bb776cb7dda9e53905e93b2e88131
```

The source root has no Git metadata, so this external recovery set is required.

## Integrity results

| Check | Result |
|---|---|
| Python AST parse | 288 files, 0 errors |
| JSON parse | 123 files, 0 errors |
| shell syntax | 0 errors |
| pytest | 319 passed, 1 skipped |
| broken symlinks | 0 |
| Markdown links before cleanup | 6 broken, all in old management README |
| CM4 ZIP integrity | 4/4 valid |
| CM4 credential flag | 4/4 false |
| 20Hz shadow plan | pass |
| 10Hz shadow plan | pass |

Zero-byte files were package markers and `weights/.gitkeep`; none were treated
as garbage.

## Frozen critical hashes

```text
20Hz GRU model.pt
55673cc2a2770187554855b65d14ca4d2a522f2e20f735e7166f23d7412070ea

10Hz GRU model.pt
2b01d247271f184263ea2fbd017a81cc3a4382649f13c8b914662473006b5bb4

10Hz GRU model_shadow.onnx
4d4c59a968c4aa61bea7394b4d465bbc6d60c9d7d6b09a5c63f302a7f2d788b0

Swin3D-B pretrained weight
7c6ae6fa165f481a9c71156644a7c0e61bb393e470ca3671b8d24a30d365ffc6
```

Protected build and deploy plans also passed. This cleanup does not change
model, ONNX, protected binary, dataset, run, or runtime-data content.

## Size and retention decisions

| Area | Approx. size | Decision |
|---|---:|---|
| `runs/` | 2.5 GB | keep; checkpoints, reports, training evidence |
| `external_models/` | 544 MB | keep; external baselines and Swin3D |
| `external_datasets/` | 220 MB | keep; manifests, windows, evaluation data |
| `runtime_data/` | 40 MB | keep; review ledgers and evidence |
| `yolo11m-pose.pt` | 41 MB | keep |
| `build/` | 3.3 MB | keep; protected deploy inputs |
| generated caches | about 2.7 MB | 215 files deleted after final regression |
| superseded CM4 handoffs | about 0.1 MB | 9 files deleted; latest set retained |

No Python module was deleted. Static inbound-reference counts cannot prove that
manual CLI entrypoints are unused.

## Model report warning

Artifacts are loadable and correctly versioned, but diagnostic performance is
not an alert promotion result.

| Candidate | Test recall | Test precision | Test ROC-AUC | Status |
|---|---:|---:|---:|---|
| 20Hz GRU | 0.8311 | 0.5325 | 0.2555 | telemetry only |
| 10Hz small GRU | 0.9071 | 0.5497 | 0.2958 | telemetry only |

Both reports warn about unknown subject identity and correlated multiview
windows. Fusion remains disabled.

## Security and permission findings

- No private-key marker or credential-bearing project RTSP configuration was
  found. Credentialed RTSP strings exist only in tests.
- API token paths are configured, but token contents are not stored in the
  audited source configs.
- Three third-party repositories retain nested `.git` directories for
  provenance.
- Thirty-five third-party model files are mode `0666`. They were not changed
  because external repositories are preserved unmodified.
- Review images and ledgers under `runtime_data` are generally mode `0664`.
  That is broader than recommended for patient/bed evidence. Tightening access
  to an explicit service group or `0600/0640` is a separate security change and
  was not mixed into structural cleanup.

## Documentation findings

The old root README and many Phase documents mixed three different eras:

1. standalone source server on `:8000` with legacy 10Hz TCN;
2. Pi/Edge-first experimental designs;
3. protected systemd deployment on `:8030` with central GRU shadow.

No history document was deleted. The canonical current boundary is now
`docs/CURRENT_ARCHITECTURE.md`; `docs/README.md` explicitly labels older
documents as history.

## Final cleanup result

All acceptance gates passed after quarantine and again after deletion:

1. 319 passed, 1 skipped;
2. both shadow plans passed with Fusion disabled;
3. help, viewer, health, status and six-camera doctor checks passed;
4. 78 Markdown files had zero broken local links and balanced fences;
5. latest CM4 ZIP, sidecar and archive test passed;
6. all frozen critical hashes remained unchanged.

Exactly 215 generated cache files and 9 superseded handoff files were removed.
No source module, model, protected build, dataset, run, runtime evidence, or latest
handoff was deleted. Post-cleanup manifests contain 1,589 source files and 10
management files.
