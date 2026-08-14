# Architecture V2 — 구현 로드맵과 현재 코드 차이

상태: 설계 초안  
원칙: 한 단계의 측정과 합격 없이 다음 단계의 모델 로직을 활성화하지 않는다.

## 1. 현재 구현과 목표의 주요 차이

| 영역 | 현재 구현 | V2 목표 |
|---|---|---|
| RTSP capture | 추론 loop가 직접 `cap.read()` | 카메라별 지속 capture + latest slot |
| viewer | 추론 loop가 발행한 frame | AI와 독립된 video plane |
| frame scheduling | loop 처리 순서 | deadline/priority latest request |
| bed segmentation | 처리 loop마다 실행 | 안정화 후 cache, 조건부 refresh |
| motion | 전체 frame 단순 diff | interaction-zone motion + scene/quality 분리 |
| 사람 없음 | 처리 loop와 부분 gating | 명시적 `EMPTY` probe 정책 |
| temporal owner | 카메라별 buffer | primary person track별 buffer |
| TCN input | local-normalized v1 | v1 보존 + global/dt/missing v2 |
| bed safe gate | 일부 자세에서 score 0 | 안전 evidence로 감점, 강한 낙상 증거 보존 |
| overload | worker/loop 속도에 의존 | 우선순위와 stale request 폐기 |
| event transport | status polling 중심 | snapshot + SSE event |

## 2. Phase 0 — 기준 측정

변경 전 다음 값을 기록한다.

- 카메라별 실제 RTSP FPS/지연
- 현재 viewer FPS/지연
- seg/pose/Keras/TCN 단일 추론 시간
- 여섯 카메라 동시 GPU/CPU/RAM
- H.264 decode error 빈도

산출물:

- baseline report
- 재현 가능한 부하 측정 명령

## 3. Phase 1 — 영상과 AI 분리

구현:

- `LatestFrameCapture` per camera
- frame sequence/timestamp
- independent viewer source
- inference는 latest slot 읽기

합격:

- AI loop를 sleep시켜도 viewer 15 FPS 이상
- frame backlog 없음
- 한 카메라 재연결이 다른 카메라에 영향 없음

이 단계에서는 모델 판정 로직을 바꾸지 않는다.

## 4. Phase 2 — Bed ROI Manager

구현:

- multi-frame segmentation consensus
- ROI cache/version
- expanded interaction zone
- scene-change invalidation
- cached/manual fallback

합격:

- stable 이후 segmentation 상시 실행 중단
- 재시작 후 cache 복원
- 카메라 이동 시 ROI 재검증

## 5. Phase 3 — Cheap Watcher와 상태머신

구현:

- local motion/global scene/corrupt frame 분리
- `OFFLINE/BOOTSTRAP/EMPTY/OCCUPIED_CALM/BURST/VERIFY/RECOVERY`
- person TTL와 empty confirmation

합격:

- 빈 카메라에서 Pose probe만 실행
- rapid motion이 정해진 지연 안에 burst 생성
- 조명 변화와 손상 frame이 바로 낙상 후보가 되지 않음

## 6. Phase 4 — 중앙 추론 스케줄러

구현:

- model별 latest request mailbox
- deadline/priority
- stale request drop
- P0/P1 burst quota와 starvation 방지

합격:

- 여섯 카메라 동시 요청에서 VERIFY 우선
- 과부하 시 queue latency가 무한 증가하지 않음
- 실제 completed Hz가 상태 API에 표시됨

## 7. Phase 5 — 사람 tracking과 TCN v1 shadow

구현:

- 모든 person Pose 처리
- primary patient track 선택
- track별 temporal buffer
- 실제 시간 10 Hz sampling
- gap/track-switch reset

합격:

- 의료진이 들어와도 환자 history가 다른 사람과 섞이지 않음
- 기존 checkpoint 입력 계약 유지
- TCN은 shadow event만 생성

## 8. Phase 6 — Hybrid fusion shadow

구현:

- global downward/rotation/shape-change
- bed relation state
- TCN ready/not-ready 두 candidate 경로
- sticky event와 evidence

합격:

- 리플레이에서 event precision/recall/latency 산출
- 안전 gate가 강한 낙상 증거를 무조건 0으로 만들지 않음
- 모든 candidate reason 재현 가능

## 9. Phase 7 — TCN V2

데이터:

- 실제 timestamp
- local + global skeleton
- velocity/acceleration
- bed relation
- missing/stale mask
- track continuity

학습:

- 공개 데이터로 기본 동작
- 자체 병상 데이터로 domain fine-tuning
- subject/video/camera 누수 없는 split

배포:

- v1과 v2 shadow 비교
- 승인된 threshold/version만 production 후보

## 10. Phase 8 — 운영 승인

- 장시간 shadow run
- false alerts per camera-hour 측정
- 장애 시험
- 외부 알림 없이 내부 dashboard 검증
- 운영 승인 후 alert output 연결

## 11. 기능 flag

각 단계는 독립적으로 되돌릴 수 있어야 한다.

```text
VIDEO_PLANE_MODE=webrtc|mjpeg
BED_ROI_AUTO=0|1
STATE_SCHEDULER=0|1
PRIORITY_SCHEDULER=0|1
TRACK_TEMPORAL=0|1
TCN_SHADOW=0|1
FUSION_SHADOW=0|1
PRODUCTION_ALERT=0|1
```

## 12. 첫 구현 단위

가장 먼저 구현할 범위는 Phase 0과 Phase 1뿐이다.

```text
RTSP capture → latest slot
                   ├─ viewer
                   └─ 기존 inference adapter
```

이를 먼저 완성해야 이후 절전 상태와 시계열 주기를 정확히 측정할 수 있다.

