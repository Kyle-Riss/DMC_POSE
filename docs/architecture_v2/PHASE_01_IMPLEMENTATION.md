# Phase 0–1 구현 기록

상태: 코드 구현 및 단위 검증 완료, 운영 프로세스 재시작 전

## 구현 범위

- 카메라별 지속 RTSP capture
- 단일 latest-frame slot
- capture와 inference 소비 속도 분리
- capture와 MJPEG viewer 소비 속도 분리
- frame sequence와 capture timestamp
- capture FPS, frame age, decode error, reconnect 지표
- 운영 기준 측정 스크립트

## 새 파일

- `latest_frame_capture.py`
- `scripts/benchmark_runtime_v2.py`
- `tests/test_latest_frame_capture.py`
- `tests/test_latest_frame_capture_thread.py`

## 변경된 연결

`server_all_cameras.py`의 카메라별 분석 thread는 더 이상 RTSP를 직접 열지 않는다.

```text
LatestFrameCapture
    ├─ /video/{camera_id}
    ├─ /image/{camera_id}
    └─ run_analysis
```

분석이 느리면 `run_analysis`는 중간 sequence를 건너뛰고 최신 frame을 받는다. capture thread와 viewer는 계속 진행한다.

## 변경 전 기준선

측정 파일:

- `runs/runtime_baseline/phase0_before_latest_capture.json`

8초, 여섯 카메라 동시 측정 결과:

| 카메라 | 분석 FPS 평균 | viewer FPS |
|---|---:|---:|
| bed_161 | 2.70 | 2.21 |
| bed_162 | 2.38 | 3.18 |
| bed_174 | 2.32 | 3.23 |
| bed_175 | 3.10 | 3.19 |
| bed_178 | 2.78 | 3.19 |
| bed_179 | 3.10 | 3.16 |

viewer가 분석 loop의 약 3 FPS에 묶여 있음을 확인했다.

## 독립 RTSP capture 확인

`bed_161`에 새 capture 모듈만 5초 연결한 결과:

```text
capture_fps: 22.48
frame_age_ms: 17.27
decode_error_total: 0
reconnect_total: 0
```

소스의 약 20 FPS를 추론과 독립적으로 drain할 수 있음을 확인했다.

## 테스트

전체 19개 단위 테스트 통과:

- latest slot overwrite
- 느린 consumer의 중간 frame skip
- consumer frame copy 격리
- 새 frame condition wait
- capture resize
- RTSP open 실패 후 reconnect
- 기존 temporal feature/TCN 테스트

추가로 다음 정적 검사를 통과했다.

- `latest_frame_capture.py` py_compile
- `server_all_cameras.py` py_compile
- `scripts/benchmark_runtime_v2.py` py_compile
- 변경 파일 `git diff --check`

## 재시작 후 확인할 항목

운영 서버를 재시작하면 다음을 다시 측정한다.

```bash
python scripts/benchmark_runtime_v2.py \
  --base-url http://127.0.0.1:8000 \
  --duration 10 \
  --output runs/runtime_baseline/phase1_after_latest_capture.json
```

합격 기준:

- `/status`에서 각 카메라 `capture_connected=true`
- `capture_fps`가 소스 FPS에 근접
- `capture_frame_age_ms`가 지속적으로 증가하지 않음
- viewer 표시 FPS 15 이상
- 분석 FPS가 낮아도 viewer FPS가 함께 낮아지지 않음
- 기존 TCN shadow 상태 API 유지

## 아직 구현하지 않은 범위

- bed segmentation 안정화 및 cache
- interaction-zone Cheap Watcher
- V2 상태머신
- 중앙 priority scheduler
- person tracking
- TCN v2 및 hybrid fusion

이 항목들은 Phase 1 운영 측정이 통과한 뒤 순서대로 구현한다.

