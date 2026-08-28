# DMC POSE repository handoff

## Two-directory contract

A complete handoff distinguishes source from operation:

| Directory | Purpose |
|---|---|
| `DMC_POSE_source` | source, tests, docs, training and evaluation |
| `DMC_POSE` | operator commands and protected shadow deployment tools |

Do not rename or merge them without updating absolute paths used by the deploy
and fine-tune tools.

## Developer acceptance

```bash
cd /home/dmc/AI/DMC_POSE_source
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider -q

cd /home/dmc/AI/DMC_POSE
./run.sh help
./run.sh status
./run.sh shadow-plan
./run.sh shadow-plan-10hz
```

Baseline at this handoff: `319 passed, 1 skipped`.

## Operator endpoints

```text
Viewer        http://<server-ip>:8030/viewer
Gateway API   http://<server-ip>:8030/api/cameras
Gateway health http://<server-ip>:8030/health
Edge health   http://<server-ip>:8020/health/live
Core internal /run/company-core/core.sock
```

Historical source-server documents use `:8000`. That is a standalone
development route, not the protected systemd operator endpoint.

## Current model policy

- Central RTX server owns heavy inference and final state aggregation.
- CM4 nodes publish H.264 RTSP and may provide lightweight diagnostic telemetry.
- Default deployed temporal contract is `80 x 109 @ 20Hz` GRU shadow.
- A `40 x 109 @ 10Hz` small-GRU alternative is staged separately.
- Fusion and real alert authority remain disabled.
- Swin3D-B is offline/staged verification research and site fine-tuning; its
  presence does not mean it is the active live fall authority.

## Data and artifacts

A source-only archive should exclude local secrets, patient/camera imagery,
datasets, runtime evidence, checkpoints, protected binaries, and service
backups. A runnable internal handoff must transfer those artifacts separately
with an explicit manifest and SHA-256 list.

Never assume the source can be restored from Git: this snapshot has no root
`.git` directory. The latest structural audit recovery set is recorded in
[FILE_AUDIT_2026-08-28.md](FILE_AUDIT_2026-08-28.md).

## CM4 appliance handoff

Build with:

```bash
cd /home/dmc/AI/DMC_POSE_source
python scripts/build_cm4_camera_handoff.py
```

The generated package sets `credentials_included=false` and excludes API/SSH/
RTSP credentials, images, video, and optional Pose weights. Secret provisioning
and Pi installation remain separate steps.
