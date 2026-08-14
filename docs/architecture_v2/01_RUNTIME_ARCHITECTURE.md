# Architecture V2 — 런타임 구성

상태: 설계 초안  
선행 문서: `00_REQUIREMENTS.md`

## 1. 프로세스 책임

중앙 서버는 논리적으로 다음 서비스를 분리한다. 최초 구현은 한 Python 프로세스 안의 독립 thread/task로 시작할 수 있지만, 데이터 소유권과 장애 경계는 아래 계약을 따른다.

### A. Stream Capture

카메라별로 하나씩 존재한다.

- RTSP를 계속 읽어 decoder buffer가 오래된 프레임으로 밀리지 않게 한다.
- 유효한 최신 프레임 하나만 보관한다.
- 프레임마다 증가하는 `frame_seq`를 부여한다.
- `capture_mono_ts`, `capture_wall_ts`, 크기, 연결 상태를 함께 저장한다.
- decode 실패 프레임을 AI와 모션 감지에 전달하지 않는다.
- 재연결은 해당 카메라 안에서만 수행한다.

출력은 큐가 아니라 원자적으로 교체되는 `LatestFrameSlot`이다.

```text
LatestFrameSlot
    camera_id
    frame_seq
    bgr_frame
    capture_mono_ts
    capture_wall_ts
    frame_width
    frame_height
    decode_ok
```

### B. Video Delivery

모니터 영상은 AI와 분리한다.

우선순위:

1. MediaMTX WebRTC/LL-HLS를 브라우저가 직접 구독
2. 중앙 미디어 게이트웨이가 H.264를 relay
3. 중앙 `LatestFrameSlot`을 MJPEG로 제공하는 fallback

브라우저는 영상 위에 AI 값을 별도 레이어로 그린다. AI 서버가 bounding box를 영상에 직접 burn-in하지 않는다.

### C. Bed ROI Manager

카메라별 ROI 상태를 관리한다.

```text
BedROI
    version
    source: auto | cached | manual
    mask
    bbox
    interaction_zone
    confidence
    stable
    detected_at
    scene_fingerprint
```

- 자동 segmentation 결과 여러 개를 비교해 안정화한다.
- 안정화된 결과를 메모리와 파일에 저장한다.
- `interaction_zone`은 침대보다 넓게 확장한다.
- 카메라 이동 또는 장면 변경을 별도 cheap detector가 감지하면 ROI를 무효화한다.
- 사람이 침대를 가린 상태에서는 기존 cache를 우선한다.

초기 권장 안정화 기준:

- 2초 동안 5회 이하로 segmentation
- 유효 검출 3개 이상
- 연속 유효 mask/bbox IoU 중앙값 0.80 이상
- 실패하면 cached ROI, 그다음 manual ROI 순서

수치는 현장 리플레이 측정 후 조정한다.

### D. Cheap Watcher

카메라별 CPU 모듈이다.

- 320px 이하 grayscale 프레임 사용
- `interaction_zone` 모션과 전체 화면 장면 변화를 분리
- 10~20 Hz로 동작
- 모션 점수, 움직임 중심, 방향, 지속 시간을 출력
- H.264 깨짐·조명 깜빡임·카메라 흔들림은 가능한 한 별도 품질 신호로 분리

Cheap Watcher는 낙상을 판정하지 않고 스케줄러를 깨운다.

### E. Camera Controller

카메라별 상태머신이다.

- 현재 상태와 TTL을 소유한다.
- 어떤 모델을 언제 실행할지 요청한다.
- 모델을 직접 실행하지 않는다.
- 카메라별 결과와 temporal history를 소유한다.

### F. Central Inference Scheduler

여섯 카메라의 모델 요청을 중앙에서 조정한다.

- FIFO frame queue를 만들지 않는다.
- 요청마다 `camera_id`, `frame_seq`, deadline, priority를 가진다.
- 실행 직전 요청된 프레임이 낡았으면 해당 카메라의 최신 프레임으로 교체한다.
- 같은 카메라·같은 모델의 대기 요청은 하나만 유지한다.
- GPU 과부하 시 낮은 우선순위 요청부터 생략한다.
- 한 카메라가 GPU를 독점하지 않도록 burst quota를 둔다.

권장 우선순위:

| 우선순위 | 작업 |
|---:|---|
| P0 | `VERIFY` Pose |
| P1 | `BURST` Pose |
| P2 | `RECOVERY` Pose |
| P3 | `OCCUPIED_CALM` Pose |
| P4 | `EMPTY` person probe |
| P5 | bed segmentation refresh |

낮은 우선순위 작업에는 starvation 방지를 위한 최대 지연을 둔다.

### G. Model Services

| 서비스 | 장치 기본값 | 메모리 정책 |
|---|---|---|
| Bed segmentation | GPU | 가중치 1회 로드 |
| Pose | GPU | 가중치 1회 로드 |
| 6-class Keras | CPU | 가중치 1회 로드 |
| TCN | CPU | 가중치 1회 로드 |

GPU worker 수는 설정값으로 추측하지 않고 한 모델 instance부터 실제 처리량을 측정해 정한다.

### H. Fusion and Event Manager

- 현재 한 프레임의 자세
- 스켈레톤 시간 변화
- 전역 하강/회전
- 침대와 사람의 공간 관계
- TCN 확률과 지속성

을 결합한다. 이벤트에는 시작, 후보, 확인, 종료 시각이 있다. API polling 때문에 이벤트가 사라지지 않도록 일정 시간 유지한다.

## 2. 데이터 최신성 계약

모든 모델 결과에는 원본 프레임 식별자를 붙인다.

```text
InferenceResult
    camera_id
    frame_seq
    capture_mono_ts
    inference_started_ts
    inference_finished_ts
    model_name
    model_version
    result
```

결과가 도착했을 때 이미 더 최신 결과가 적용됐다면 오래된 결과를 폐기한다.

## 3. 영상 overlay 원칙

브라우저가 다음 두 입력을 합성한다.

```text
Video plane: WebRTC/LL-HLS/MJPEG
Data plane: status snapshot + event stream
```

데이터 plane에는 결과의 `frame_seq`와 나이를 표시한다. 영상과 결과가 완벽하게 같은 프레임일 필요는 없지만 결과가 얼마나 오래됐는지는 알 수 있어야 한다.

## 4. 부하 저하 정책

GPU 또는 CPU가 목표 지연을 넘으면 다음 순서로 품질을 낮춘다.

1. bed segmentation refresh 연기
2. `EMPTY` person probe 간격 증가
3. `OCCUPIED_CALM`의 비필수 시각화 생략
4. Pose 입력 해상도 축소
5. 낮은 위험 카메라의 Pose 주기 감소

다음 항목은 부하 때문에 중단하지 않는다.

- RTSP 최신 프레임 drain
- 빠른 영상 전송
- Cheap Watcher
- `VERIFY` 상태의 필수 관측
- 이벤트 상태 전달

## 5. 장애 경계

| 장애 | 영향 범위 | 복구 |
|---|---|---|
| 한 RTSP 끊김 | 해당 카메라 | backoff 재연결 |
| 한 decoder 오류 | 해당 프레임/카메라 | 프레임 폐기, 다음 IDR 대기 |
| bed ROI 실패 | 해당 카메라 공간 판정 | cache/manual fallback |
| GPU timeout | 해당 모델 요청 | 최신 프레임으로 재요청 |
| TCN buffer gap | 해당 카메라 temporal 판정 | not-ready 전환 |
| API client 단절 | 해당 client | 영상/추론 지속 |

