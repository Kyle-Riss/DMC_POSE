# Architecture V2 — 추론 및 하이브리드 융합

상태: 설계 초안

## 1. 처리 단위

TCN history는 카메라가 아니라 사람 track 단위로 관리한다.

병상에는 환자 외에 보호자나 의료진이 들어올 수 있다. Pose 결과의 첫 번째 사람만 사용하면 다른 사람의 스켈레톤이 환자 history에 섞일 수 있다.

```text
Camera
  └─ PersonTrack 1
       ├─ relation_to_bed
       ├─ latest_pose
       ├─ kinematic_history
       └─ temporal_buffer
  └─ PersonTrack 2
       └─ ...
```

초기 primary patient 선택 기준:

1. 침대 mask와 지속적으로 가장 많이 겹치는 track
2. 이전 primary track과 공간적으로 이어지는 track
3. track continuity와 confidence

track ID가 바뀌거나 대상 사람이 바뀌면 TCN buffer를 그대로 이어 붙이지 않는다.

## 2. 단계별 입력과 출력

### Bed Segmentation

입력:

- 최신 유효 RGB frame

출력:

```text
bed_mask
bed_bbox
bed_confidence
roi_version
```

사용처:

- interaction zone 생성
- 사람과 침대의 공간 관계
- 침대 안 안전 자세와 침대 밖 낙상 구분

### Cheap Watcher

입력:

- 연속 최신 frame의 downscaled grayscale
- interaction zone

출력:

```text
motion_score
motion_area_ratio
motion_center_xy
dominant_direction
rapid_motion
scene_change
frame_quality
```

Cheap Watcher 값은 원본 RGB가 아니라 decode가 정상인 raw frame에서 계산한다. UI 글자나 AI overlay가 포함된 프레임을 다시 입력하면 안 된다.

### Pose and Tracking

입력:

- 최신 RGB frame

출력:

```text
track_id
17 keypoints x/y/confidence
person_bbox
pose_capture_ts
pose_frame_seq
```

가능하면 모든 사람을 추출하고 primary patient track을 별도로 선택한다.

### 6-class

입력:

- 현재 primary track의 17개 keypoint

출력:

```text
front_lying
prone_back
side_near
side_far
sitting_center
sitting_edge
```

이 모델은 현재 자세를 설명할 뿐, 자세로 낙상 과정을 직접 증명하지 않는다.

### TCN v1

입력:

- 실제 시간 10 Hz
- 30개 sample
- 기존 checkpoint와 동일한 109개 feature

출력:

```text
tcn_ready
tcn_fall_probability
tcn_threshold
tcn_persistence_count
```

운영 위치:

- shadow evidence
- 최종 알람을 단독으로 발생시키지 않음

## 3. Skeleton Feature V2

현재 TCN v1은 신체 중심 상대좌표 위주라 화면에서 사람 전체가 아래로 떨어지는 움직임이 약하게 표현될 수 있다. v2에는 다음 세 그룹을 함께 넣는다.

### Local/body-relative

- 몸 중심과 신체 크기로 정규화한 17개 관절
- 좌우·상하 관절 관계
- 몸통 각도
- confidence와 visibility

카메라 위치 차이를 줄이고 자세 모양을 표현한다.

### Global/frame-relative

- 머리, 어깨 중심, 골반 중심의 frame-normalized 좌표
- person bbox 중심, 폭, 높이, 종횡비
- 침대 mask/bbox와의 거리
- body-in-bed ratio
- 침대 중심에서 바깥으로 이동하는 방향

사람 전체의 하강과 침대 밖 이동을 보존한다.

### Temporal quality and motion

- 실제 `dt`
- 관절·중심 velocity
- acceleration
- 관측 누락 mask
- stale age
- track continuity

모델에 누락 관측을 숨기지 않는다.

## 4. 빠른 기구학 점수

TCN과 별도로 즉시 계산하는 `kinematic_risk`가 필요하다.

입력 후보:

- pelvis/head/shoulder의 수직 속도
- bbox center 수직 속도
- torso angle의 변화 속도
- bbox aspect ratio 변화
- 짧은 시간의 가속도
- rapid motion 직후 낮은 위치의 lying posture

이 점수는 `BURST → VERIFY` 전환에 사용한다. 절대 임계값은 카메라 해상도와 설치 각도에 따라 달라지므로 frame-normalized 값과 현장 리플레이로 보정한다.

## 5. 침대 관계

침대는 hard gate가 아니라 context다.

```text
IN_BED
EDGE
EXITING
OUTSIDE_NEAR
OUTSIDE_FAR
UNKNOWN
```

`IN_BED + lying`은 일반적으로 안전 증거지만, 빠른 하강이나 큰 회전이 동시에 존재하면 낙상 후보를 무조건 0으로 만들지 않는다.

특히 다음 흐름을 보존해야 한다.

```text
IN_BED → EDGE → EXITING → OUTSIDE_NEAR
```

이는 침대 이탈 낙상의 중요한 시간 패턴이다.

## 6. 융합 단계

### Screening

비싼 분석을 깨울지 결정한다.

```text
rapid_motion
OR person_probe_detected
OR recent_track_fast_change
```

출력:

- `BURST` 진입 여부

### Candidate

낙상 가능성을 검증할지 결정한다.

경로 A — TCN 준비됨:

```text
tcn_probability 지속
AND (
    kinematic_risk
    OR unsafe_bed_transition
    OR post_impact_lying
)
```

경로 B — TCN 준비 안 됨:

```text
strong_downward_motion
AND fast_rotation_or_shape_change
AND post_motion_low_or_lying_pose
```

출력:

- `VERIFY` 진입
- evidence 목록

### Confirmation

초기 확인 논리:

```text
candidate evidence가 검증 window 동안 지속
AND 명확한 safe recovery가 없음
AND frame/track quality가 최소 기준 이상
```

출력:

- `SHADOW_ALERT`
- 향후 검증 완료 후 `ALERT`

## 7. 안전 증거와 억제 규칙

안전 증거:

- 침대 중앙에서 오랫동안 안정된 누운 자세
- 급격한 하강 없이 천천히 앉거나 눕기
- 움직임 후 즉시 정상적인 서기/걷기 회복
- track 품질이 낮아 판정 불가능

억제 원칙:

- 안전 증거는 risk를 낮추지만 강한 낙상 증거를 즉시 삭제하지 않는다.
- `UNKNOWN`은 `SAFE`가 아니다.
- keypoint confidence가 낮으면 확률을 0으로 만드는 대신 `insufficient_evidence`로 표시한다.

## 8. 결과 계약

카메라별 snapshot:

```json
{
  "camera_id": "bed_161",
  "analysis_state": "OCCUPIED_CALM",
  "frame_seq": 123456,
  "frame_age_ms": 82,
  "roi": {
    "ready": true,
    "version": 4,
    "source": "auto"
  },
  "occupancy": {
    "present": true,
    "primary_track_id": 17,
    "age_ms": 95
  },
  "motion": {
    "score": 0.12,
    "rapid": false
  },
  "pose": {
    "class": "sitting_edge",
    "confidence": 0.87,
    "age_ms": 106
  },
  "temporal": {
    "ready": true,
    "fall_probability": 0.18,
    "sample_count": 30,
    "model_version": "tcn-v1-shadow"
  },
  "fusion": {
    "risk": 0.11,
    "level": "SAFE",
    "evidence": []
  }
}
```

낙상 event:

```json
{
  "event_id": "bed_161-...",
  "camera_id": "bed_161",
  "track_id": 17,
  "phase": "SHADOW_ALERT",
  "started_at": "...",
  "confirmed_at": "...",
  "evidence": [
    "rapid_motion",
    "global_downward_motion",
    "post_motion_lying",
    "tcn_persistent"
  ],
  "model_versions": {
    "pose": "...",
    "six_class": "...",
    "tcn": "..."
  }
}
```

## 9. 반드시 별도 평가할 항목

- 사람 없이 조명만 바뀌는 영상
- H.264 손상 프레임
- 의료진과 환자가 동시에 있는 영상
- 침대에서 천천히 내려오기
- 침대 가장자리 앉기
- 침대에서 물건 줍기
- 실제 또는 모사 급격 낙상
- 사람이 화면 밖으로 사라짐
- TCN warm-up 전에 발생한 낙상

