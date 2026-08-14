# Phase 4 검증 기록

## 자동 검증

39개 전체 테스트 통과. 이 중 스케줄러 테스트는 다음을 고정한다.

- P0 VERIFY가 P3 EMPTY보다 먼저 처리됨
- 동일 `(model, camera)`의 오래된 대기 요청이 최신 요청으로 교체됨
- deadline을 넘긴 요청은 모델 실행 없이 stale drop
- 긴급 요청 4회 후 일반 요청을 한 번 처리하는 기아방지
- completed Hz와 worker 생존 상태 노출

```bash
/home/dmc/anaconda3/envs/pose-cuda/bin/python \
  -m unittest discover -s tests -v
```

## 실카메라 초기 결과

6대 모두 EMPTY 상태에서:

- Capture: 약 20FPS
- Watcher: 약 20FPS
- Pose/central scheduler completed: 약 0.7–0.75Hz
- GPU queue latency: 약 0ms
- YOLO inference: 약 4–7ms
- stale/superseded/timeout/error: 0
- mailbox pending: 0

Phase 3 검사도 스케줄러 적용 후 다시 통과했다.

## 반복 가능한 실서버 검사

```bash
/home/dmc/anaconda3/envs/pose-cuda/bin/python \
  scripts/check_phase4_scheduler.py --seconds 10
```

합격 조건은 6대 모두 scheduler thread 생존, mailbox backlog 1 이하, 측정 구간 내 completed 증가, EMPTY에서 예상하지 않은 drop 없음, capture 15FPS 이상이다.

## 다음 현장 부하 시험

사람 또는 리플레이를 이용해 2개 이상 카메라에 동시에 BURST를 발생시키고 P1/P0 queue latency와 drop을 측정한다. 실제 낙상 리플레이 전에는 스케줄러 drop 자체를 오류로 해석하지 않는다. 과부하에서 stale drop은 오래된 영상 추론을 막는 의도된 동작이다.
