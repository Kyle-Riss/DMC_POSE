# DMC POSE local operations

The operator entry point is `/home/dmc/AI/DMC_POSE`.

```bash
cd /home/dmc/AI/DMC_POSE
./run.sh status
./run.sh viewer
```

Start the service only when it is not ready:

```bash
./run.sh start
```

For a non-mutating diagnosis:

```bash
./run.sh health
./run.sh doctor
./run.sh shadow-status
```

The Viewer uses `http://<server-ip>:8030/viewer`. Do not use the historical
standalone-source `:8000` instructions for the protected service.

A shadow deployment is not a normal restart. First run the matching read-only
plan, review the model contract and files, and only then use the deploy command.
Both available shadow routes remain telemetry-only and keep Fusion disabled.

Do not edit `/opt/.company-core` directly or delete `runs`, `build`, weights,
datasets, or runtime evidence.
