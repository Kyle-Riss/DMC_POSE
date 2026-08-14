# Phase 5 검증 기록

## 자동 테스트

전체 44개 테스트 통과. Tracker 전용 테스트는 다음을 검증한다.

- detection 배열 순서가 바뀌어도 공간적으로 이어진 사람의 ID 유지
- sustained bed overlap이 높은 사람을 primary로 선택
- 기존 primary 연속성이 작은 challenger 점수 차이로 탈취되지 않음
- primary TTL 만료 후 새 ID를 선택하고 기존 history와 구분
- 관측되지 않은 frame에서 이전 skeleton을 새 관측처럼 재사용하지 않음

## 실서버 확인

2026-07-31 실카메라 검증에서 실제 사람이 보이는 5개 카메라와 빈 카메라
1개를 동시에 관측했다.

- 모든 primary의 `primary_track_id == tcn_track_id` 확인
- 실제 primary의 TCN buffer가 30 sample까지 도달
- 한 카메라에서 2명이 동시에 검출되어도 primary 한 명만 TCN에 입력
- 초기 2초 track TTL에서는 짧은 포즈 누락 뒤 ID 재생성이 관측되어
  track TTL을 5초로 상향
- 단, TCN gap reset은 1.5초로 유지하여 5초 동안 관측이 끊긴 과거
  skeleton이 시계열에 이어 붙지 않도록 분리

최종 설정 재시작 후 빈 장면 6대에서 30초간 다음 조건을 재확인했다.

- capture와 Watcher 약 20FPS
- `person_count=0`, `track_count=0`
- `primary_track_id=null`, `tcn_track_id=null`
- EMPTY Pose 약 0.75FPS
- scheduler drop 0
- scheduler queue latency 0ms, pending 0
- track 생성/만료/switch 증가 없음

이 결과는 “사람이 없을 때 비싼 모델을 거의 끄는 동작”과 “사람이 있을 때
한 사람의 ID에만 TCN history를 귀속하는 동작”을 각각 검증한 것이다.

반복 검사:

```bash
/home/dmc/anaconda3/envs/pose-cuda/bin/python \
  scripts/check_phase5_tracking.py --seconds 5
```

## 아직 필요한 현장 합격 시험

환자 역할 1명과 의료진 역할 1명이 함께 보이는 시퀀스에서 다음을 기록한다.

- 사람이 서로 지나갈 때 primary ID 유지
- 의료진이 침대 옆에 접근할 때 불필요한 switch 없음
- 환자가 침대에서 바닥으로 이동할 때 continuity가 bed-overlap 감소보다 우선함
- 완전 가림 후에는 기존 history를 억지로 이어 붙이지 않음

이 현장 시험 전에는 primary tracker와 TCN candidate를 외부 경보 확정값으로 승격하지 않는다.
