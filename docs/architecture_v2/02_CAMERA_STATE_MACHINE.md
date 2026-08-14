# Architecture V2 — 카메라 상태머신

상태: 설계 초안

## 1. 상태 목록

### `OFFLINE`

RTSP에서 유효 프레임을 받지 못한다.

- 영상: 연결 끊김 표시
- AI: 실행 안 함
- 동작: 해당 카메라만 지수 backoff 재연결
- 전환: 연속 유효 프레임 확보 → `BOOTSTRAP`

### `BOOTSTRAP`

프레임 품질과 침대 ROI를 준비한다.

- 영상: 즉시 표시
- bed segmentation: ROI 안정화에 필요한 횟수만 실행
- person probe: 1회 이상 실행
- Cheap Watcher: background 초기화
- 전환:
  - ROI 준비 + 사람 없음 → `EMPTY`
  - ROI 준비 + 사람 있음 → `OCCUPIED_CALM`
  - ROI를 자동 생성하지 못함 → cached/manual ROI로 계속
  - 프레임 끊김 → `OFFLINE`

### `EMPTY`

최근 확인 결과 사람이 없다.

- 영상: 정상 속도
- Cheap Watcher: 10~20 Hz
- Pose person probe: 0.5~1 Hz
- 6-class: 중단
- TCN: not-ready
- bed segmentation: scene-change 또는 낮은 빈도의 refresh만
- 전환:
  - probe에서 사람 확인 → `OCCUPIED_CALM`
  - interaction zone rapid motion → `BURST`
  - scene change → `BOOTSTRAP`
  - 프레임 끊김 → `OFFLINE`

`EMPTY` 확정은 한 번의 no-person 결과로 하지 않는다. 초기 기준은 다음 조건을 모두 만족하는 것이다.

- 마지막 person 관측 이후 3초 이상
- 연속 person probe 3회 no-person
- 지속적인 큰 모션 없음

### `OCCUPIED_CALM`

사람이 존재하지만 급격한 움직임이 없다.

- Cheap Watcher: 10~20 Hz
- Pose v1: 실제 관측 10 Hz
- 6-class: Pose 결과마다 또는 5~10 Hz
- TCN v1: 10 Hz, 30 sample rolling window
- 전환:
  - rapid motion 또는 큰 스켈레톤 변화 → `BURST`
  - 사람 소실 TTL 만료 → `EMPTY`
  - scene change → `BOOTSTRAP`

최종 v2 TCN 재학습 후에는 calm Pose를 2~5 Hz로 낮출 수 있다.

### `BURST`

빠른 사건을 놓치지 않기 위한 고속 관측 상태다.

- 진입 즉시 해당 카메라 요청 우선순위 P1
- Pose: 최소 10 Hz, 자원이 허용되면 20 Hz
- 현재 TCN: Pose 결과 중 실제 시간 10 Hz sample만 입력
- 스켈레톤 전역 하강·회전·속도·가속도 계산
- 최소 유지 시간: 초기값 1.5초
- 전환:
  - 낙상 조건 일부 충족 → `VERIFY`
  - 움직임 안정 + 후보 없음 → `OCCUPIED_CALM`
  - 사람 없음이 확인됨 → grace 후 `EMPTY`

### `VERIFY`

낙상 후보를 짧게 집중 검증한다.

- 해당 카메라 요청 우선순위 P0
- Pose: 10~20 Hz
- TCN: ready 여부와 확률 지속성 확인
- 침대 안/가장자리/밖 관계 확인
- 최소 1초, 최대 3초의 검증 window
- 전환:
  - 확인 기준 충족 → `ALERT`
  - 안전 자세/회복 확인 → `RECOVERY`
  - 정보 부족 → `RECOVERY`와 low-confidence event 기록

TCN이 아직 warm-up되지 않은 경우에도 다음과 같은 강한 기구학적 증거가 있으면 `VERIFY`에는 진입할 수 있다.

- 빠른 전역 하강
- 큰 몸통 회전
- 하강 직후 낮은 위치의 누운 자세

### `ALERT`

확인된 낙상 이벤트를 sticky하게 유지한다.

- 이벤트 ID 발급
- 후보 근거와 model version 기록
- viewer와 event stream에 전달
- Pose: 사건 직후 일정 시간 고속 유지
- 전환:
  - 정해진 hold 시간 후 → `RECOVERY`
  - 운영자 확인 여부는 이벤트 상태에 별도 기록

TCN shadow 검증 단계에서는 실제 외부 알람 대신 `SHADOW_ALERT`로만 기록한다.

### `RECOVERY`

오탐 취소, 사람의 회복, 사건 후 상태를 추적한다.

- Pose: 5~10 Hz
- 이전 이벤트 상태 유지
- 전환:
  - 사람이 안정됨 → `OCCUPIED_CALM`
  - 사람 없음 TTL 만료 → `EMPTY`
  - 다시 급격한 움직임 → `BURST`

## 2. 사람 존재 TTL

Pose가 한 프레임 실패했다고 사람이 사라진 것으로 처리하지 않는다.

```text
person_present =
    recent_pose_person within presence_ttl
    OR active_verified_track
```

초기 `presence_ttl`은 3초로 두고 리플레이 평가 후 조정한다.

## 3. 모션 트리거

모션은 세 종류로 분리한다.

| 종류 | 의미 | 처리 |
|---|---|---|
| local motion | 침대 interaction zone의 움직임 | burst 후보 |
| global scene change | 화면 대부분이 동시에 변화 | ROI 재검증 |
| corrupt/noisy frame | decode 깨짐 또는 순간 잡음 | 프레임 폐기/연속성 확인 |

한 프레임의 motion spike만으로 `BURST`에 진입하지 않는다. 단, 매우 큰 변화는 즉시 진입할 수 있다. 일반 변화는 짧은 연속 조건을 사용한다.

## 4. 상태 전환 기록

모든 전환은 다음 형태로 기록한다.

```text
StateTransition
    camera_id
    from_state
    to_state
    reason
    frame_seq
    capture_ts
    transition_ts
    metrics
```

이를 통해 “왜 Pose가 켜졌는가”, “왜 알람 후보가 됐는가”를 재현할 수 있다.

## 5. 상태별 모델 주기

| 상태 | Cheap Watcher | Bed Seg | Pose | 6-class | TCN |
|---|---:|---:|---:|---:|---:|
| OFFLINE | 0 | 0 | 0 | 0 | 0 |
| BOOTSTRAP | 10 Hz | 최대 2~3 Hz 임시 | 1회 이상 | 필요 시 | reset |
| EMPTY | 10~20 Hz | scene-change/refresh | 0.5~1 Hz | 0 | off |
| OCCUPIED_CALM v1 | 10~20 Hz | cached | 10 Hz | 5~10 Hz | 10 Hz |
| BURST | 20 Hz | cached | 10~20 Hz | 10 Hz | 입력 10 Hz |
| VERIFY | 20 Hz | 필요 시 | 10~20 Hz | 10 Hz | 입력 10 Hz |
| ALERT | 20 Hz | cached | 10~20 Hz | 10 Hz | 입력 10 Hz |
| RECOVERY | 10~20 Hz | cached | 5~10 Hz | 5 Hz | 조건부 |

주기는 목표값이며, 실제 실행률과 frame age를 상태 API에서 측정해야 한다.

