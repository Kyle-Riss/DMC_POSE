# Phase 9 — Review ledger and operational calibration

## 발견된 문제

2026-07-31 feature-only 기록 약 1.0 bed-hour에서 기존 `hybrid_v1`이 10개의 `SHADOW_ALERT` 후보를 만들었다.

- bed_162: 6
- bed_174: 1
- bed_175: 3
- 공통점: 모든 후보 구간의 legacy fall score가 0
- 주요 조합: `tcn_persistent + rapid_motion`

후보는 영상 검토 전이므로 자동으로 오탐 처리하지 않는다. 다만 현재 TCN은 GMDCSA-24의 4명 데이터 기반 shadow 모델이고 motion과 TCN은 같은 움직임에 반응하는 상관 신호이므로, 이 두 신호만으로 alert를 확정하는 구조는 운영에 부적합하다.

## Fusion policy v2

정책 버전은 `hybrid_v2_structural_confirm`이다.

- `TCN persistent + rapid motion`은 `CANDIDATE/VERIFY`까지만 허용
- `outside-bed lying` 또는 `kinematic risk >= 0.35`가 같은 primary track에서 확인되어야 `SHADOW_ALERT`
- 구조적 확인 후에는 기존 alert hold를 유지
- 실제 외부 알림은 계속 비활성화된 shadow-only
- feature 레코드마다 `fusion_policy_version`을 저장하여 정책 전후 결과를 섞지 않음
- summary는 `policy_bed_hours`를 별도로 계산하고 운영 평가 기본값은 v2만 선택

## 두 개의 검토 장부

`runtime_data/shadow_review.csv`

- 시스템 후보의 안정적인 `candidate_id`
- 가능한 label: `pending`, `true_fall`, `false_alarm`, `staged_fall`, `uncertain`
- 스크립트를 다시 실행해도 기존 label/reviewer/note를 보존

`runtime_data/actual_events.csv`

- 실제 및 시험 낙상을 독립적으로 기록
- 가능한 event type: `actual_fall`, `staged_fall`
- 검출 후보가 있으면 `matched_candidate_id`에 연결

후보 장부만으로는 놓친 낙상을 알 수 없다. 실제 사건 장부를 함께 유지해야 sensitivity를 계산할 수 있다.

## 실행

```bash
cd /home/dmc/AI/DMC_POSE
./run_shadow_report.sh
```

생성물:

- `runtime_data/shadow_summary.json`
- `runtime_data/shadow_review.csv`
- `runtime_data/actual_events.csv`
- `runtime_data/operational_report.json`

서버를 재시작한 뒤 `GET /calibration/status`에서 최신 운영 리포트의 readiness와 전체 지표를 읽을 수 있다.

## 시작용 엔지니어링 게이트

아래 수치는 의료·임상 표준이 아니라 현장 시험을 시작하기 위한 프로젝트 기준이다.

- 최소 누적: 168 bed-hours
- false alarms/bed-hour: 0.01 이하
- 실제 낙상 sensitivity: 0.90 이상
- pending/uncertain 후보가 있으면 false-alarm gate는 `NOT_READY`
- 실제 낙상 장부가 비어 있으면 detection gate는 `NOT_MEASURED`

최종 readiness는 false-alarm gate와 detection gate가 모두 PASS일 때만 PASS다.

## v2 초기 관찰

재시작 후 v2만 분리한 최초 관찰은 다음과 같다.

- 누적 0.416797 bed-hours
- v2 `SHADOW_ALERT` 후보 0
- 구버전 후보 10개는 v2 지표에서 제외하고 검토 장부에는 그대로 보존
- false-alarm gate: 누적 시간이 부족하여 `NOT_READY`
- detection gate: 실제/시험 낙상 장부가 비어 있어 `NOT_MEASURED`

후보 0건은 정확도 통과를 의미하지 않는다. 정상 운용 시간과 계획된 시험 낙상 결과를 계속 누적해야 한다.
