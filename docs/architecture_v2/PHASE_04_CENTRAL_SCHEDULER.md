# Phase 4 — 중앙 추론 스케줄러

## 목적

6개 카메라가 동시에 움직여도 GPU 요청이 FIFO로 계속 쌓이지 않게 한다. Viewer와 RTSP capture는 이 스케줄러를 통과하지 않으므로 영상 표시 속도는 AI 부하와 분리된다.

## 구현 계약

각 `(model, camera_id)`는 대기 중 요청을 최대 하나만 가진다. 같은 mailbox에 새 프레임이 오면 이전 대기 프레임은 `superseded`로 끝난다. deadline을 넘긴 프레임은 추론하지 않고 `stale`로 폐기한다.

우선순위:

| 우선순위 | 의미 | 현재 deadline |
|---|---|---:|
| P0 | 이전 TCN candidate의 VERIFY 구간 | 350ms |
| P1 | Watcher가 연 BURST | 450ms |
| P2 | 사람이 있는 OCCUPIED | 800ms |
| P3 | EMPTY person probe | 1200ms |
| P4 | Bed ROI 학습/주기 재검증 | 2000ms |

P0/P1을 네 번 처리하면 대기 중인 일반 요청 하나를 먼저 처리해 특정 방의 probe나 ROI 검증이 영원히 굶지 않게 한다.

## 데이터 흐름

```mermaid
flowchart LR
  RTSP[RPi RTSP] --> CAP[LatestFrameCapture 20FPS]
  CAP --> VIEW[/video viewer]
  CAP --> LOOP[카메라 상태 루프]
  LOOP --> BOX[(model/camera latest mailbox)]
  BOX --> SELECT{priority + deadline}
  SELECT -->|valid| GPU[중앙 YOLO worker]
  SELECT -->|expired/replaced| DROP[drop + metric]
  GPU --> HYBRID[Pose + 6-class + TCN shadow + Bed relation]
  HYBRID --> STATUS[/status]
```

전체 그림은 `06_phase4_scheduler_flow.mmd`에 있다.

## 운영 지표

`/status`의 각 카메라에 다음 값이 노출된다.

- `scheduler_completed_total`, `scheduler_completed_hz`
- `scheduler_queue_latency_ms`, `scheduler_inference_ms`
- `scheduler_stale_drop_total`, `scheduler_superseded_drop_total`
- `scheduler_timeout_total`, `scheduler_error_total`
- `scheduler_pending`, `scheduler_last_priority`, `scheduler_last_model`
- `scheduler_thread_alive`

Viewer의 `GPU Scheduler` 행에서도 completed Hz, queue ms, 최근 우선순위, drop 합계를 확인할 수 있다.

## 경보 안전성

스케줄러는 어떤 판정도 새로 만들지 않는다. TCN은 계속 shadow이고 실제 경보는 아직 기존 결과와 분리되어 있다. 이 단계의 책임은 필요한 최신 프레임을 제한된 자원에서 제때 모델로 보내는 것이다.
