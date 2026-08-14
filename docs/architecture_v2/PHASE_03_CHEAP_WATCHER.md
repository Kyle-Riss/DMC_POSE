# Phase 3 — 독립 모션 감시와 10Hz 시계열

상태: 구현 및 6대 실카메라 1차 검증 완료

## 목적

영상 모니터링 속도와 AI 추론 부하를 분리한다.

- RPi는 카메라 캡처와 H.264 RTSP 송출만 담당한다.
- 중앙 서버의 캡처와 뷰어는 약 20FPS를 유지한다.
- 신경망을 쓰지 않는 감시기가 매 새 프레임을 확인한다.
- 빈 침대에서는 YOLO Pose를 약 0.75Hz probe로 제한한다.
- 빠른 움직임이 연속 두 프레임에서 확인되면 즉시 `BURST`를 연다.

## 런타임 경로

```text
RTSP -> latest frame slot
          |-> /video viewer
          |-> 160x90 grayscale MotionWatcher
                     |-> EMPTY: pose probe 0.75Hz
                     `-> BURST: AI loop sleep interrupt + pose hot path
```

감시기는 자동 침대 ROI가 준비되기 전에는 전체 화면을 보고, 준비된
후에는 침대 bbox를 20% 확장한 영역을 본다. 전체 segmentation mask를
매 프레임 복사하지 않고 bbox만 잠금 조회한다.

## 기본 파라미터

| 환경 변수 | 기본값 | 의미 |
|---|---:|---|
| `POSE_MOTION_SMALL_WIDTH` | 160 | 감시 영상 폭 |
| `POSE_MOTION_SMALL_HEIGHT` | 90 | 감시 영상 높이 |
| `POSE_MOTION_RATIO_THRESHOLD` | 0.018 | 변화 픽셀 비율 하한 |
| `POSE_MOTION_CONSECUTIVE_HITS` | 2 | BURST에 필요한 연속 hit |
| `POSE_MOTION_BURST_HOLD_SEC` | 3.0 | BURST 유지 시간 |
| `POSE_EMPTY_PROBE_HZ` | 0.75 | 빈 침대 Pose 확인 주기 |

전체 화면에 가까운 변화 비율 70% 초과는 조명 전환이나 손상 프레임으로
보고 단독 BURST 근거에서 제외한다.

## 10Hz TCN 입력

> 이 절의 최종 기준은 Phase 10 고정 카메라 계약이다. 초기 구현의
> missing-row/1.5초 gap 방식은 폐기됐다.

라이브 Pose 결과는 도착 횟수가 아니라 capture timestamp를 기준으로
10Hz 시간축에 배치한다.

- 동일 primary의 실제 Pose 관측 30행이 준비돼야 한다.
- 관측 간격 70ms 미만은 중복으로 건너뛴다.
- 관측 간격 70~150ms만 같은 window에 추가한다.
- 150ms를 초과하면 해당 primary의 window를 reset한 뒤 새 관측부터 시작한다.
- 사람 전체 미검출 시 zero skeleton 또는 missing row를 삽입하지 않는다.
- 이전 skeleton을 복사하지 않는다.
- TCN은 계속 shadow이며 실제 경보를 변경하지 않는다.

빈 상태에서는 Pose가 0.75Hz이므로 TCN window를 유지하지 않는다.
TCN은 사람 검출 또는 BURST 이후의 연속 관측 구간에서 준비된다.

## API 관측값

`/status`에 다음 값이 추가되었다.

```text
runtime_mode
watcher_fps
watcher_frame_seq
watcher_processed_total
motion_ratio
motion_hit_streak
motion_trigger_total
burst_active
burst_remaining_ms
pose_inference_total
pose_inference_fps
```

## 제한

현재 사람 이력은 아직 카메라 단위다. 의료진과 환자가 동시에 보일 때
스켈레톤이 섞이지 않도록 하는 multi-person tracking과 primary patient
선택은 후속 Phase에서 구현한다.
