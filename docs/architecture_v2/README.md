# DMC POSE Architecture V2

설계 문서 읽는 순서:

1. `00_REQUIREMENTS.md` — 목표와 절대 제약
2. `01_RUNTIME_ARCHITECTURE.md` — 프로세스와 데이터 흐름
3. `02_CAMERA_STATE_MACHINE.md` — 카메라별 상태와 모델 주기
4. `03_INFERENCE_AND_FUSION.md` — 모델 계약과 낙상 융합
5. `04_OPERATIONS_AND_ACCEPTANCE.md` — API, 지표, 합격 기준
6. `05_IMPLEMENTATION_ROADMAP.md` — 현재 코드 차이와 단계적 구현
7. `PHASE_02_AUTO_BED_ROI.md` — 수동 작업 없는 침대 자동 검출과 운영 검사
8. `PHASE_03_CHEAP_WATCHER.md` — 20FPS 경량 감시, 절전, 10Hz TCN 입력
9. `PHASE_03_VALIDATION.md` — 6대 실카메라 측정과 재검증 명령
10. `PHASE_04_CENTRAL_SCHEDULER.md` — latest mailbox, 우선순위, deadline
11. `PHASE_04_VALIDATION.md` — Phase 4 자동/실카메라 검증
12. `PHASE_05_PRIMARY_TRACKING.md` — multi-person primary와 track별 TCN
13. `PHASE_05_VALIDATION.md` — identity/ownership 검증
14. `PHASE_06_HYBRID_FUSION.md` — 공간·pose·motion·TCN shadow 결합
15. `PHASE_07_MODEL_AUDIT.md` — TCN 성숙도와 shadow-only 근거
16. `PHASE_08_FEATURE_RECORDING.md` — 영상 없이 운영 feature 기록
17. `13_CENTRAL_MAX_CADENCE_CONTRACT.md` — 중앙 20Hz 최대 cadence, 단일 decode, 모델별 시간축과 외부 baseline 격리 계약
18. `PHASE_09_REVIEW_AND_CALIBRATION.md` — 리뷰 ledger와 정책별 평가

Mermaid:

- `00_context.mmd`
- `01_runtime_flow.mmd`
- `02_camera_states.mmd`
- `03_inference_sequence.mmd`
- `04_auto_bed_roi_flow.mmd`
- `05_phase3_runtime_flow.mmd`
- `06_phase4_scheduler_flow.mmd`
- `07_phase5_tracking_flow.mmd`
- `08_phase6_hybrid_fusion_flow.mmd`
- `09_phase8_feature_recording_flow.mmd`
- `10_phase9_review_calibration_flow.mmd`

이 디렉토리는 Architecture V2의 설계 기준선이다. 기존 `docs/` 문서는 현재 구현 설명 또는 이전 설계 자료로 취급한다.
