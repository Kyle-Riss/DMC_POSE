# DMC POSE documentation index

문서는 **현재 기준**, **운영·검증 근거**, **역사 문서**로 구분합니다. 서로 충돌할
때는 아래 현재 기준 문서를 우선합니다.

## 처음 읽는 순서

1. [CURRENT_ARCHITECTURE.md](CURRENT_ARCHITECTURE.md) — 현재 배포 경계와 flowchart
2. [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) — 개발·테스트·배포 인수인계
3. [PRODUCTION_GRU_CHECKLIST_2026-08-24.md](PRODUCTION_GRU_CHECKLIST_2026-08-24.md) — 완료/차단 상태
4. [architecture_v2/13_CENTRAL_MAX_CADENCE_CONTRACT.md](architecture_v2/13_CENTRAL_MAX_CADENCE_CONTRACT.md) — 시간축·소유권 계약
5. [FILE_AUDIT_2026-08-28.md](FILE_AUDIT_2026-08-28.md) — 정리 전후 보존 근거

## 현재 운영·개발 문서

- [CM4_CAMERA_APPLIANCE_V1.md](CM4_CAMERA_APPLIANCE_V1.md)
- [SITE_FINETUNE_QUICKSTART.md](SITE_FINETUNE_QUICKSTART.md)
- [TEMPORAL_LABEL_PROTOCOL_V1.md](TEMPORAL_LABEL_PROTOCOL_V1.md)
- [EXTERNAL_RESOURCE_AUDIT_2026-08-24.md](EXTERNAL_RESOURCE_AUDIT_2026-08-24.md)
- [CENTRAL_20HZ_DATA_READINESS_2026-08-24.md](CENTRAL_20HZ_DATA_READINESS_2026-08-24.md)
- [GRU_DIAGNOSTIC_TRAINING_2026-08-24.md](GRU_DIAGNOSTIC_TRAINING_2026-08-24.md)

## 성능 근거

PNG/SVG dashboard는 보기 좋은 요약일 뿐, promotion 근거를 대신하지 않습니다.

- `gru_shadow_training_performance_20260828.*`
- `gru_model_comparison_20260828.*`
- `gru_10hz_20hz_comparison_20260828.*`
- `deployed_model_performance_20260824.svg`
- `central_temporal_compute_benchmark_20260824.*`

정확한 수치는 해당 run의 `report.json`과 데이터 경고를 함께 읽습니다.

## Architecture V2

`architecture_v2/`는 중앙 latest-frame, scheduler, tracking, fusion으로 진화한
설계 기록입니다. 13번 max-cadence 계약은 현재 기준이고, 00–12번 문서에는 당시
10Hz TCN·Pi-first·Edge-first 설계가 남아 있습니다.

## 역사 문서

다음 문서는 삭제하지 않고 설계 이력으로 보존합니다. 경로·포트·모델 계약을 현재
운영 명령으로 그대로 사용하지 않습니다.

- `PHASE_*.md`, `IMPLEMENTATION_PLAN.md`, `NEXT_EXECUTION_PLAN_2026-08-07.md`
- `SMART_CAMERA_PIPELINE_PLAN.md`, `EDGE_FIRST_ARCHITECTURE_V1.md`
- `MULTI_CAMERA_GUIDE.md`, `RPI_RTSP_TCN_SHADOW.md`
- `diagrams/`의 기존 `:8000` / `run_all_cameras.sh` 흐름
- `runtime_artifact.json`: 2026-08-07 TCN snapshot이며 현재 배포 설명이 아님
- 루트 `PROJECT_README.md`, `FASTAPI_PLAN.md`: 초기 독립 실행 문서

현재 운영은 `/home/dmc/AI/DMC_POSE/run.sh`와 Viewer `:8030`을 사용합니다.
소스 독립 개발 서버의 `:8000` 문서와 혼동하지 마십시오.
