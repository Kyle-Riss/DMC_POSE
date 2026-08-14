# Phase 2 실시간 검증 결과

검증일: 2026-07-31  
대상: `/home/dmc/AI/DMC_POSE`, RTSP 카메라 6대

## 결과

| Camera | Auto ROI | Agreement IoU | Bootstrap seg runs | Capture FPS |
|---|---:|---:|---:|---:|
| bed_161 | READY | 0.917 | 3 | 약 20 |
| bed_162 | READY | 0.992 | 3 | 약 20 |
| bed_174 | READY | 0.998 | 3 | 약 20 |
| bed_175 | READY | 0.959 | 3 | 약 20 |
| bed_178 | READY | 0.957 | 3 | 약 20 |
| bed_179 | READY | 0.924 | 4 | 약 20 |

합의 완료 후 8초 동안 모든 카메라의 `bed_seg_run_count`가 그대로 유지됐다.
따라서 매 프레임 실행되던 침대 세그멘테이션이 자동 캐시 이후 중단되는 것을 확인했다.

현재 사람이 없는 상태의 중앙 AI 파이프라인은 약 3.1 FPS이고, latest-frame capture는 약 20 FPS다.
즉 영상 모니터링 속도와 AI 절전 주기가 분리되어 있다.

## 가중치 검증

기존 루트 가중치:

```text
yolo11n-bed-seg.pt
sha256 7c37010c923ad576502365f41a719f7bd45c91fef46dd3a12947a47ee8f99a40
```

실시간 bed_161 프레임에서 최고 confidence가 약 0.014여서 운영 임계값 0.1을 넘지 못했다.

선택된 가중치:

```text
bed_seg/runs/bed_seg/weights/best.pt
sha256 e0f49849fd9b9d432486a7ad29a33b00b330b63ac7c468bb6e94de57d237341e
```

6대 단일 프레임 검증 confidence:

```text
bed_161 0.9379
bed_162 0.9156
bed_174 0.9154
bed_175 0.9146
bed_178 0.8753
bed_179 0.5458
```

모두 segmentation mask를 반환했다. 서버와 `run_all_cameras.sh`의 기본 침대 가중치는 이 파일로 변경했다.

## 실행한 승인 검사

```bash
python scripts/check_auto_bed_roi.py \
  --url http://127.0.0.1:8000/status \
  --timeout 60 \
  --settle-seconds 8
```

결과:

```text
PASS: every camera has an automatic ROI and segmentation is throttled
```

## 운영 해석

- `READY`: 자동 합의 ROI를 Pose/TCN 공간 판단에 사용할 수 있다.
- `ROI_NOT_READY`: 자동 합의 전이며 ROI 기반 판정을 하지 않는다.
- 카메라가 이동하거나 해상도/화각이 바뀌면 캐시를 폐기하고 다시 자동 합의한다.
- 수동 ROI 파일이나 카메라별 수동 좌표는 실시간 서버 경로에서 사용하지 않는다.
