# DMC POSE 문서 인덱스

현재 설계 기준선은 `architecture_v2/`입니다. 기존 문서는 배경과 이전 구현 기록으로 보존하며, 새 구현 판단은 Architecture V2와 `runtime_artifact.json`을 우선합니다.

## 처음 읽는 순서

1. `BLACKBOX_RUNTIME_GUIDE.md` — 비개발자용 런타임 설명
2. `architecture_v2/00_REQUIREMENTS.md` — 목표와 절대 제약
3. `architecture_v2/01_RUNTIME_ARCHITECTURE.md` — 전체 런타임 구조
4. `architecture_v2/02_CAMERA_STATE_MACHINE.md` — 절전과 wake-up 상태 전이
5. `architecture_v2/03_INFERENCE_AND_FUSION.md` — 하이브리드 모델 계약
6. `architecture_v2/04_OPERATIONS_AND_ACCEPTANCE.md` — 운영과 합격 기준
7. `architecture_v2/05_IMPLEMENTATION_ROADMAP.md` — Phase별 구현 순서
8. `runtime_artifact.json` — 현재 실행본과 정책 스냅샷

## 프로젝트 기여·이력서 기록

- `PROJECT_CONTRIBUTION_AND_RESUME_RECORD_2026-08-10.md` — 실제 구현 범위, 검증 수치, 이력서 권장 문장과 과장 방지 경계
- `AUTOMATIC_TEMPORAL_EVENT_CAPTURE_2026-08-10.md` — 자동 시작·종료 트리거, 109D 세션 아티팩트, Mermaid 데이터 흐름
- `REVIEWED_HYBRID_TCN_EXPERIMENT_2026-08-10.md` — FallVision 수동 양성+자체 정상 통합, 동결 평가, 승격 보류 근거
- `reviewed_hybrid_tcn_dashboard_2026-08-10.svg` — 동결 test recall과 false events/hour 시각 비교
- `FALL_DETECTION_OPERATIONAL_READINESS_2026-08-14.md` — hybrid v6 실사용 준비도, 실제 기록 재생 결과, Go/No-Go 표와 승격 기준
- `FALL_EVENT_LOG_POSE_EVIDENCE_2026-08-14.md` — bed_161 통제 낙상 로그와 영상 유래 109D 스켈레톤을 병합한 재현 가능한 단일 사건 근거

## 현재 구현 상태

| Phase | 내용 | 상태 |
|---|---|---|
| 1 | 영상 캡처와 Viewer/AI 경로 분리 | 완료 |
| 2 | 자동 침대 ROI | 완료 |
| 3 | 20Hz 경량 watcher와 10Hz 시계열 입력 | 완료 |
| 4 | 중앙 latest-only 우선순위 스케줄러 | 완료 |
| 5 | 다중 사람 추적과 track별 TCN 버퍼 | 완료 |
| 6 | 침대·pose·motion·TCN 하이브리드 fusion | 완료 |
| 7 | TCN 성숙도 감사 | 완료, shadow-only |
| 8 | 영상 미저장 feature recorder | 완료 |
| 9 | 리뷰 ledger와 정책별 calibration | 완료 |
| 10 | 운영 장시간 검증과 배포 경계 확정 | 다음 단계 |

각 Phase의 근거는 `architecture_v2/PHASE_*.md`에서 확인합니다.

## 다이어그램

- `architecture_v2/*.mmd`: 현재 Architecture V2와 Phase별 Mermaid
- `diagrams/*.mmd`: 시스템·시작·추론·장애 흐름
- `canvas/`: 별도 Canvas 렌더링 자료

## 이전 문서

`IMPLEMENTATION_PLAN.md`, `PHASE_2.md`, `PHASE_TIMESERIES.md`, `PARALLEL_GPU_PIPELINE.md`, `MULTI_CAMERA_GUIDE.md`, `SMART_CAMERA_PIPELINE_PLAN.md`, `TIMESERIES_STRATEGY.md`는 참고 자료입니다. 현재 코드와 충돌하면 Architecture V2를 따릅니다.

## 인수인계

전송 패키지 범위와 복원 순서는 `REPOSITORY_HANDOFF.md`를 참조합니다. 카메라 인증정보, 실제 영상, 데이터셋, 모델 가중치는 소스 ZIP에 포함하지 않습니다.
