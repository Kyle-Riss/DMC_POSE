# Architecture V2 — 운영 인터페이스와 합격 기준

상태: 설계 초안

## 1. 외부 인터페이스

### Viewer

```text
GET /viewer
```

- 영상 plane과 AI data plane을 브라우저에서 합성한다.
- 카메라별 frame age, AI result age, 분석 상태를 표시한다.
- 영상이 정상이고 AI가 중단된 경우도 서로 다르게 표시한다.

### Status snapshot

```text
GET /api/v2/status
GET /api/v2/status/{camera_id}
```

- 현재 카메라 상태 전체를 반환한다.
- polling client와 진단에 사용한다.
- 응답 HTTP 200은 실시간 정상의 충분조건이 아니다.

### Event stream

```text
GET /api/v2/events/stream
```

- 초기 구현은 SSE를 권장한다.
- 낙상 후보, 상태 전환, 연결 장애를 push한다.
- 양방향 명령이 필요해질 때 WebSocket을 검토한다.

### Event history

```text
GET /api/v2/events
GET /api/v2/events/{event_id}
```

- 이벤트 근거와 model/ROI version을 조회한다.

### Health

```text
GET /health/live
GET /health/ready
```

- `live`: API process가 살아 있음
- `ready`: 공유 모델과 최소 구성요소가 준비됨
- 개별 카메라 연결 상태는 readiness와 분리한다.

### Metrics

```text
GET /metrics
```

Prometheus 형식을 권장한다.

## 2. 필수 카메라 지표

```text
capture_connected
capture_fps
capture_frame_age_ms
decode_error_total
reconnect_total

video_delivery_fps
video_delivery_latency_ms

analysis_state
state_age_sec
roi_ready
roi_version
roi_age_sec
roi_source

motion_hz
motion_score
rapid_motion_total
scene_change_total

person_present
person_observation_age_ms
primary_track_id
track_switch_total

pose_requested_hz
pose_completed_hz
pose_result_age_ms
pose_deadline_miss_total

tcn_ready
tcn_samples
tcn_probability
tcn_gap_reset_total

event_candidate_total
shadow_alert_total
```

공유 자원 지표:

```text
gpu_utilization
gpu_memory
inference_requests_by_priority
inference_deadline_miss_by_priority
dropped_stale_request_total
cpu_utilization
process_rss
```

## 3. 기술 합격 기준

### 영상 plane

- 여섯 카메라 동시 viewer에서 카메라당 표시 FPS 15 이상
- 정상 LAN에서 영상 지연 p95 1초 이하
- AI를 강제로 느리게 해도 영상 FPS가 목표 범위 유지
- 한 RTSP를 차단해도 나머지 다섯 영상 무중단

### Capture

- backlog 길이는 항상 0 또는 latest slot 1개
- inference에 전달되는 frame age p95 250ms 이하를 목표
- decode 오류 프레임이 모션/추론 입력으로 전달되지 않음
- 재연결 후 자동으로 `BOOTSTRAP` 수행

### 상태 스케줄러

- `EMPTY`에서 Pose가 설정한 probe 범위를 넘겨 상시 실행되지 않음
- rapid motion부터 `BURST` 요청 생성까지 200ms 이내를 목표
- `VERIFY` 요청이 낮은 우선순위 작업보다 먼저 실행됨
- GPU 과부하 시 stale frame 요청이 쌓이지 않고 폐기됨

### 침대 ROI

- stable ROI가 확보되면 매 프레임 segmentation이 중단됨
- ROI cache를 재시작 후 복원 가능
- scene change 시 기존 ROI가 무효화되고 재검증됨
- 자동 검출 실패 시 cached/manual fallback 동작

### 시간축

- Pose 결과에 원본 `frame_seq`와 capture timestamp 존재
- TCN v1 sample 간격이 실제 시간 100ms 기준
- gap과 track switch가 temporal buffer에 기록 또는 reset됨
- 20 FPS 입력을 v1 TCN에 20 Hz로 직접 넣지 않음

## 4. ML 합격 기준

최종 수치는 현장 요구와 validation replay 결과로 승인한다. 적어도 다음을 event 단위로 측정해야 한다.

- fall event recall
- event precision
- false alerts per camera-hour
- detection latency
- TCN warm-up 전 사건 recall
- 침대 가장자리·천천히 눕기·물건 줍기 오탐률
- 조명 변화·H.264 오류 오탐률
- 의료진 동시 등장 시 track 혼선률

frame accuracy만으로 운영 모델을 승인하지 않는다.

평가 분리는 최소한 다음 누수를 막아야 한다.

- 동일 인물 train/test 중복
- 동일 영상의 인접 window train/test 중복
- 동일 카메라 장면에 대한 과적합

## 5. 장애 시험

1. 각 RTSP를 한 대씩 30초 차단 후 복구
2. H.264 packet 손상 또는 frame drop 주입
3. GPU 추론을 의도적으로 지연
4. bed segmentation 결과 없음
5. TCN model load 실패
6. 브라우저 여러 개 동시 접속
7. 중앙 API client 연결/해제 반복
8. 카메라 화면 이동 및 가림

각 시험에서 영상 plane, AI plane, 이벤트 plane 중 어느 것이 영향을 받았는지 별도로 기록한다.

## 6. 운영 안전장치

- TCN은 shadow flag로 독립 활성화
- fusion alert는 shadow/production mode 분리
- model version과 threshold version 기록
- ROI 변경 시 version 증가
- 모든 상태 전환 reason 기록
- 외부 알림 연동 전 리플레이와 현장 shadow 기간 필수

