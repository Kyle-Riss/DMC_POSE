# DMC POSE 고정 카메라 모듈 계약

상태: **Phase 10 구현 기준선**  
대상: `bed_161`, `bed_162`, `bed_174`, `bed_175`, `bed_178`, `bed_179`  
결정일: 2026-07-31  
운영 원칙: 외부 경보는 독립 리플레이·운영 검증 전까지 `shadow-only`

이 문서는 기존 설계의 선택지를 제거하고, 여섯 고정 카메라에 실제로 구현할
모듈 경계, 프레임 주기, 입력·출력 계약을 고정한다.

## 1. 이번에 고정하는 결론

1. RPi는 H.264 RTSP 송출만 담당한다. 무거운 AI 가중치를 올리지 않는다.
2. 모니터 영상 경로와 AI 경로를 분리한다.
3. 중앙 서버에서 각 가중치는 한 번만 로드하고 여섯 카메라가 공유한다.
4. 각 카메라는 backlog queue가 아니라 `(camera, task)`별 latest mailbox를 쓴다.
5. 침대 ROI는 자동 생성한다. 수동 좌표 fallback은 운영 계약에서 제거한다.
6. 사람이 없는 동안에는 20 Hz Cheap Watcher와 1 Hz Pose probe만 유지한다.
7. 사람이 확인되면 현재 TCN v1 계약 때문에 실제 Pose 관측을 10 Hz로 유지한다.
8. 낙상 후보는 Motion, Tracking/Pose, Kinematic, Bed Relation, TCN을 구분해
   계산한다. 어느 한 모듈도 단독으로 낙상을 확정하지 않는다.
9. 이벤트는 `BED_EXIT`, `FALL`, `BED_EXIT_FALL`로 구분한다.
10. TCN v1에는 zero-filled missing row나 이전 skeleton을 입력하지 않는다.

## 2. 2026-07-31 실카메라 프레임 감사

각 RTSP에서 640x360 프레임 4장을 약 0.35초 간격으로 읽고, 현재 운영
가중치와 threshold를 적용했다.

| 카메라 | Bed conf | Bed mask 면적 | 약 1초 global motion 최대 | Pose person |
|---|---:|---:|---:|---:|
| bed_161 | 0.9314 | 49.52% | 0.000% | 0 |
| bed_162 | 0.9335 | 60.10% | 0.063% | 0 |
| bed_174 | 0.8757 | 46.04% | 0.083% | 0 |
| bed_175 | 0.9171 | 42.38% | 0.049% | 0 |
| bed_178 | 0.9112 | 49.77% | 0.000% | 0 |
| bed_179 | 0.2317 | 54.80% | 0.000% | 0 |

판단:

- 여섯 스트림은 모두 프레임 수신에 성공했다.
- 샘플 구간은 사람이 없는 정적 장면이라 `EMPTY` baseline으로만 유효하다.
- `bed_179`는 반복 IoU가 높아도 confidence가 너무 낮다. 기존 cache의
  `ready=true`를 신뢰하면 안 된다.
- `bed_174`는 bbox가 왼쪽 영상 경계에 닿는다. 경계 접촉만으로 오검출이라
  단정하지 않되, mask 면적과 confidence를 함께 검사해야 한다.
- 이 짧은 샘플은 사람 진입, 정상 침대 이탈, 낙상 민감도를 증명하지 않는다.

## 3. 프로세스 경계

### 3.1 RPi Camera Node

책임:

- 카메라 캡처
- H.264 하드웨어 인코딩
- RTSP 송출
- 재연결과 송출 상태

출력:

```text
rtsp://<camera-ip>:8554/stream
```

금지:

- bed segmentation
- Pose
- 6-class
- TCN
- 최종 사건 판정

### 3.2 Video Plane

브라우저 모니터링용 영상 경로다. AI가 중단돼도 계속 동작해야 한다.

목표 경로:

```text
RPi RTSP → Media gateway → WebRTC/LL-HLS → Browser
```

현재 `/video/{camera_id}` MJPEG는 진단 fallback으로만 유지한다. AI Python
프로세스가 viewer 영상의 유일한 공급자가 되어서는 안 된다.

### 3.3 AI Capture Adapter

카메라별 독립 worker다.

출력 `FrameEnvelope`:

```text
camera_id
frame_seq
source_pts           optional; RTSP PTS를 얻을 수 있을 때만
decode_mono_ts       중앙 서버에서 decode가 끝난 monotonic time
received_wall_ts
frame_bgr
decode_ok
width
height
```

규칙:

- OpenCV decode 시각을 카메라 촬영 시각이라고 부르지 않는다.
- `source_pts`가 없으면 시간축 품질을 `decode_time_only`로 표시한다.
- 최신 프레임 하나만 원자적으로 교체한다.
- 한 카메라의 read/reconnect가 다른 카메라를 막지 않는다.

### 3.4 AI Runtime

중앙 서버에서 다음 가중치를 각각 한 번 로드한다.

```text
BedSegRuntime     bed_seg/runs/bed_seg/weights/best.pt
PoseRuntime       yolo11m-pose.pt
PostureRuntime    my_model_six_check.keras
TemporalRuntime   runs/temporal_tcn/gmdcsa24_tcn/model.pt
```

카메라마다 모델 객체를 새로 만들지 않는다. 별도 Triton 같은 네트워크 모델
서버는 지금 추가하지 않는다. 우선 중앙 scheduler와 micro-batch runtime으로
가중치 중복, 직렬 backlog, 프로세스 복잡도를 줄인다.

## 4. 프레임 단위 처리 모듈

### M1. Frame Quality Gate

입력: `FrameEnvelope`

출력:

```text
VALID
CORRUPT
FORMAT_CHANGED
STALE
```

판정:

- decode 실패 또는 빈 frame: `CORRUPT`
- 해상도 변경: `FORMAT_CHANGED`
- latest frame이 상태별 deadline보다 오래됨: `STALE`
- 실패한 프레임은 motion, Pose, TCN에 넣지 않는다.

### M2. Auto Bed ROI Manager

BOOTSTRAP에서 1 Hz로 후보 5개를 수집한다. 최소 3개가 합의해야 한다.

acceptance gate:

```text
median confidence >= 0.60
bbox area ratio in [0.20, 0.90]
mask area ratio in [0.20, 0.75]
median pairwise bbox IoU >= 0.75
valid mask exists
```

프레임 경계 접촉은 단독 reject 사유가 아니다. 세 개 이상의 경계에 동시에
붙거나 면적 sanity를 위반할 때 reject한다.

cache identity:

```text
camera_id
frame_width
frame_height
bed_model_sha256
scene_fingerprint
schema_version
```

다음 조건이면 cache를 폐기한다.

- 모델 hash 변경
- 해상도 변경
- 3회 지속된 scene change
- refresh 결과가 기존 ROI와 불일치
- cache sanity gate 실패

ROI를 만들지 못하면 `roi_state=DEGRADED`다. 수동 좌표를 자동으로 대입하지 않는다.
이 상태에서도 일반 fall shadow 분석은 가능하지만 `BED_EXIT` 및
`BED_EXIT_FALL`은 `insufficient_roi`로 판정 보류한다.

현재 기준으로 `bed_179`는 새 gate를 통과하기 전까지 `roi_state=DEGRADED`
대상이다.

디스크 cache를 복원한 경우 새 검출처럼 표시하지 않는다.

```text
roi.source = auto_cache
roi.restored_from_cache = true
```

이후 live refresh가 성공하면 `roi.source=auto_refresh`,
`roi.restored_from_cache=false`로 바꾼다.

### M3. Automatic Bed Zones

stable bed mask에서 자동 파생한다.

```text
BED_CORE
    mask를 bed bbox 짧은 변의 8%만큼 erosion

BED_EDGE
    bed_mask - BED_CORE

OUTSIDE_NEAR
    bed_mask를 짧은 변의 25%만큼 dilation한 영역 - bed_mask

OUTSIDE_FAR
    화면 내부에서 위 세 영역을 제외한 영역
```

zone은 detector가 아니라 context다. 화면 perspective가 다르므로 pixel 거리로
실제 지면 거리를 주장하지 않는다. 리플레이에서 카메라별 zone failure가
확인되면 비율만 camera calibration 값으로 저장하되 수동 polygon은 사용하지
않는다.

### M4. Cheap Motion Watcher

입력:

- overlay 없는 raw latest frame
- 160x90 grayscale
- 자동 interaction zone

주기: capture가 유효한 동안 20 Hz

기본 pixel threshold: 22

분류 체계:

```text
global_ratio < 0.40
    local motion 평가

0.40 <= global_ratio < 0.70
    LARGE_CHANGE
    즉시 BURST, ROI 갱신은 동결

global_ratio >= 0.70
    SCENE_CHANGE_CANDIDATE
    첫 관측은 BURST
    3회 연속이면 ROI 무효화 후 BOOTSTRAP
```

local rapid motion:

```text
interaction-zone changed ratio >= 0.018, 2회 연속
OR interaction-zone changed ratio >= 0.08, 1회
```

이 계약은 기존 70~75% 미분류 구간을 제거한다. H.264 손상처럼 frame quality가
실패한 경우에는 motion으로 승격하지 않고 프레임을 폐기한다.

### M5. Occupancy Gate

EMPTY에서는 1 Hz Pose probe를 실행한다. local motion 또는 large change가
발생하면 다음 주기를 기다리지 않고 즉시 Pose를 요청한다.

사람 있음:

```text
person confidence >= 0.50
AND visible keypoints(conf >= 0.25) >= 7
```

한 번의 유효 person 검출로 즉시 `OCCUPIED_CALM`에 진입한다.

사람 없음:

```text
primary를 마지막으로 본 뒤 >= 3초
AND 1 Hz probe 3회 연속 no-person
AND 지속 local motion 없음
```

그 후에만 `EMPTY`로 전환한다.

### M6. Pose and Multi-person Tracking

모든 person을 검출하고 track별로 관리한다.

`PersonObservation`:

```text
camera_id
track_id
frame_seq
pose_ts
17 x (x, y, confidence)
person_bbox
bed_overlap
quality
```

primary 선택 순서:

1. 기존 primary와 공간적으로 연속인 유효 track
2. 최근 2초 동안 bed overlap이 가장 지속적인 track
3. Pose quality와 track continuity

규칙:

- 의료진이 들어와도 detection 배열의 첫 사람을 primary로 쓰지 않는다.
- primary가 관측되지 않은 frame에는 이전 skeleton을 재사용하지 않는다.
- primary 변경이 확정되는 즉시 kinematic history와 TCN buffer를 reset한다.
- switch candidate가 사라지면 switch counter를 0으로 reset한다.
- last-known skeleton은 UI의 stale 상태에만 쓸 수 있고 모델 입력에는 금지한다.

### M7. Pose Cadence and Temporal Sampler

상태별 Pose 목표:

| 상태 | Pose | TCN 입력 |
|---|---:|---:|
| OFFLINE | 0 | reset |
| BOOTSTRAP | 1회 이상 | reset |
| EMPTY | 1 Hz probe | off |
| OCCUPIED_CALM | 실제 관측 10 Hz | 10 Hz |
| BURST | 실제 관측 10 Hz | 10 Hz |
| VERIFY | 실제 관측 10 Hz | 10 Hz |
| SHADOW_ALERT | 실제 관측 10 Hz | 10 Hz |
| RECOVERY | 10 Hz 5초 후 5 Hz | reset 또는 기록 전용 |

TCN v1 sampler:

```text
sample interval target = 100 ms
timestamp source priority:
    1. valid source_pts
    2. decode_mono_ts

dt < 70 ms = duplicate observation, skip
dt 70..150 ms = append observed sample
dt > 150 ms = reset before accepting a new sequence
window = 30 consecutive observed samples
track change = reset
missing Pose = no row inserted
previous Pose = never copied
```

따라서 `tcn_ready=true`는 동일 primary의 실제 관측 30개가 시간 계약을 만족할
때만 가능하다. 151~250 ms도 예외 없이 reset 대상이다. 기존 1.5초 gap과
zero missing row 방식은 TCN v1 운영 입력으로 사용하지 않는다.

### M8. Kinematic Observer

TCN과 독립적으로 실제 관측 pair에서 계산한다.

```text
pelvis/head/shoulder center vertical velocity
bbox center vertical velocity
torso angle velocity
bbox aspect-ratio change
short-window acceleration
body-to-bed direction
post-motion low/lying state
```

좌표는 frame 크기와 bed bbox 짧은 변으로 정규화한다. `kinematic_risk >= 0.35`
같은 값은 shadow tuning용 임시값이며 운영 확정 threshold가 아니다.

### M9. Posture 6-class

현재 순간 자세를 설명한다.

```text
front_lying
prone_back
side_near
side_far
sitting_center
sitting_edge
```

단일 frame의 lying이나 sitting만으로 낙상을 확정하지 않는다.

### M10. TCN v1 Shadow

입력:

- 동일 primary
- 실제 10 Hz
- 연속 30 sample
- 기존 109-feature 계약

출력:

```text
tcn_ready
fall_probability
threshold
persistence
input_quality
```

역할:

- 낙상 transition의 보조 evidence
- 리플레이와 현장 shadow 통계

금지:

- 단독 외부 경보
- ROI나 6-class 결과를 TCN 확률처럼 해석
- missing-aware v2를 학습하기 전 zero/missing row 사용

### M11. Hybrid Event Fusion

출력 사건:

```text
BED_EXIT
FALL
BED_EXIT_FALL
INSUFFICIENT_EVIDENCE
```

`BED_EXIT`:

```text
IN_BED → BED_EDGE → OUTSIDE_NEAR
AND uncontrolled descent confirmation 없음
```

`BED_EXIT`은 일반 상태·이벤트이며 `SHADOW_ALERT`를 생성하지 않는다.

`FALL` shadow candidate 경로 A — 빠른 낙상:

```text
rapid/large motion
AND structural confirmation(
    global downward motion
    AND rotation 또는 shape change
    AND post-motion low/lying 또는 recovery failure
)
```

`FALL` shadow candidate 경로 B — 느린 미끄러짐/하강:

```text
TCN persistent
AND kinematic downward 또는 rotation evidence
AND post-event low/lying 또는 recovery failure
```

Motion Watcher는 경로 A의 깨우기 신호이며 모든 FALL의 필수조건이 아니다.
경로 B에서도 TCN 단독 경보는 금지한다.

`BED_EXIT_FALL`:

```text
bed-exit transition
AND FALL structural confirmation
AND 동일 primary, 동일 event window
```

ROI가 degraded이면 `BED_EXIT` 계열은 보류하고 일반 `FALL` shadow evidence만
기록한다.

`SHADOW_ALERT` 생성 대상은 `FALL`과 `BED_EXIT_FALL`뿐이다.

### M12. Central Scheduler

mailbox key:

```text
(model_name, camera_id)
```

priority:

```text
P0 VERIFY / SHADOW_ALERT
P1 BURST
P2 OCCUPIED_CALM Pose
P3 BOOTSTRAP bed segmentation
P4 EMPTY person probe / ROI refresh
```

규칙:

- mailbox마다 대기 요청은 최신 한 개뿐이다.
- 새 요청이 오면 실행 전인 이전 요청은 `superseded_drop`한다.
- deadline을 넘긴 요청은 `stale_drop`한다.
- `empty_probe_drop`, `p0_stale_drop`, `superseded_drop`을 별도 metric으로 센다.
- P0/P1을 우선하되 P4가 영구 starvation되지 않도록 urgent quota를 둔다.
- BedSeg도 P3 BOOTSTRAP/P4 refresh로 scheduler를 통과하며, camera별
  single/latest runner에서 실행한다.
- Pose는 최대 20 ms collection window에서 같은 priority의 최신 프레임을
  micro-batch한다. batch 최대값 3은 초기값이며 2/4/6대 benchmark로 확정한다.

deadline 초안:

| Priority | Deadline |
|---|---:|
| P0 | 150 ms |
| P1 | 200 ms |
| P2 | 300 ms |
| P3 | 1,000 ms |
| P4 | 1,500 ms |

## 5. 카메라 상태머신 고정안

```text
OFFLINE
  → BOOTSTRAP
  → EMPTY 또는 OCCUPIED_CALM
  → BURST
  → VERIFY
  → SHADOW_ALERT
  → RECOVERY
```

ROI 품질은 analysis state와 독립된 직교 상태다.

```text
analysis_state =
    OFFLINE | BOOTSTRAP | EMPTY | OCCUPIED_CALM |
    BURST | VERIFY | SHADOW_ALERT | RECOVERY

roi_state =
    NOT_READY | READY | DEGRADED
```

`roi_state=DEGRADED`여도 analysis state는 `OCCUPIED_CALM`, `BURST`,
`VERIFY` 등이 될 수 있다. 이 경우 bed relation만 `UNKNOWN`으로 제한한다.

`ALERT`는 아직 활성 상태가 아니다. 현재 외부 출력은 반드시
`SHADOW_ALERT`이며 viewer에서도 운영 확정 경보와 다르게 표시한다.

## 6. API 계약

정식 상태 API는 `/api/v2/status`로 고정한다. `/status`는 호환 alias다.

카메라별 필수 필드:

```text
camera_id
stream_state
analysis_state
frame_seq
source_pts
source_pts_available
decode_age_ms
ai_input_age_ms
roi.ready
roi.quality
roi.source
roi.restored_from_cache
roi.version
roi.model_sha256
occupancy.present
occupancy.primary_track_id
occupancy.age_ms
motion.classification
motion.global_ratio
motion.local_ratio
pose.actual_hz
pose.age_ms
temporal.ready
temporal.sample_count
temporal.input_quality
temporal.fall_probability
fusion.event_type
fusion.shadow_only
scheduler.priority
scheduler.queue_age_ms
last_error
```

`roi.source` enum:

```text
auto_consensus
auto_refresh
auto_cache
auto_not_ready
auto_degraded
```

`manual`은 운영 enum에서 제거한다. 디스크에서 복원된 자동 cache는 identity와
sanity 검사를 통과한 뒤 `auto_cache`, `restored_from_cache=true`로 표시한다.

## 7. 검증 계약

### 즉시 코드 수정 후 unit/integration

- low-confidence cache가 ready가 되지 않는지
- 모델 hash와 해상도 변경 시 cache가 무효화되는지
- 40~70% large change가 미분류되지 않는지
- corrupt frame이 motion burst를 만들지 않는지
- switch candidate 소실 시 counter reset
- primary 소실 시 skeleton 미재사용
- 250 ms TCN gap과 track change에서 buffer reset
- `/api/v2/status` 필드와 enum

### 리플레이 검증

각 카메라 관점에서 다음을 포함한다.

- 사람 진입
- 조명 변화
- 이불 움직임
- 정상 침대 이탈
- 침대 가장자리 앉기
- 의료진과 환자 동시 등장
- 빠르게 주저앉기
- 매트 위 staged fall
- RTSP/H.264 손상

측정:

```text
motion wake latency
first-pose latency
primary ID switch
P0/P1 deadline miss
BED_EXIT/FALL/BED_EXIT_FALL confusion
TCN ready ratio와 reset reason
```

### 장기 shadow 검증

- 정책과 모델 version이 고정된 valid bed-hours만 계산한다.
- capture, ROI, scheduler, recorder가 모두 정상인 시간만 valid다.
- false alarm 0건으로 0.01/hour 이하를 95% 신뢰 수준에서 주장하려면
  약 300 valid bed-hours가 필요하다.
- sensitivity는 staged fall과 actual fall을 분리하고 사건 수와 신뢰구간을
  함께 기록한다.
- 승인된 clip, NVR timestamp 또는 운영자 사건 기록 없이 자동으로
  true/false label을 붙이지 않는다.

## 8. Phase 10 구현 순서

1. `FrameEnvelope`와 timestamp 명칭 통일
2. ROI cache identity/sanity gate 및 `ROI_DEGRADED`
3. Motion 분류 체계 통일
4. primary switch/reset 및 TCN v1 no-missing 계약
5. scheduler metric과 micro-batch
6. `/api/v2/status` 계약 통일
7. 여섯 카메라 replay harness
8. video plane과 AI process 장애 분리
9. shadow soak 재개

이 순서 전에는 장기 Phase 10 시간만 늘리지 않는다. 잘못된 입력 계약으로
수집한 bed-hours는 운영 calibration 근거로 사용할 수 없기 때문이다.
