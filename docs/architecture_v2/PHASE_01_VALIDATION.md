# Phase 1 운영 검증

검증 시각: 2026-07-31  
측정 파일: `runs/runtime_baseline/phase1_after_latest_capture.json`

## 결과

| 카메라 | Capture FPS | Viewer FPS | 분석 FPS | Capture age p95 |
|---|---:|---:|---:|---:|
| bed_161 | 20.02 | 19.60 | 10.67 | 105ms |
| bed_162 | 20.04 | 19.61 | 10.80 | 92ms |
| bed_174 | 20.09 | 19.29 | 10.79 | 144ms |
| bed_175 | 20.00 | 20.00 | 10.89 | 58ms |
| bed_178 | 20.04 | 19.65 | 10.88 | 81ms |
| bed_179 | 20.08 | 18.67 | 10.70 | 91ms |

판정: Phase 1 합격

- 여섯 카메라 모두 capture 연결 정상
- viewer 목표 15 FPS 이상 충족
- viewer 18.67~20.00 FPS
- 분석은 약 10.7~10.9 FPS로 viewer와 독립
- decode error 0
- reconnect 0
- TCN 여섯 카메라 모두 ready, 30 sample 유지
- 분석 frame age 약 112~147ms

## 변경 전후

```text
변경 전 viewer: 2.2~3.2 FPS
변경 후 viewer: 18.7~20.0 FPS
```

## 브라우저 캐시

서버 로그에서 `/image/bed_*?t=...`를 반복하는 client는 이전 viewer HTML을 사용 중이다.

새 viewer는 `/video/bed_*`를 사용한다. 해당 브라우저에서 다음 중 하나를 수행한다.

- `Ctrl+Shift+R`
- 기존 탭을 닫고 `/viewer` 재접속

## 다음 단계

Phase 2:

- bed segmentation multi-frame 안정화
- ROI cache/version
- stable ROI 이후 segmentation 상시 실행 중단
- scene change 시 ROI 재검증
- cached/manual fallback

