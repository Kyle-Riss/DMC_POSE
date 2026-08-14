# Phase 0 Baseline — 2026-08-07

## 판정

Phase 0 기준선 감사는 코드 커밋을 제외하고 완료했다. 현재 작업 트리는 사용자 변경과 이번 프로젝트 구현이 함께 존재하므로 자동 commit은 수행하지 않았다.

## 테스트

- 명령: `python -m unittest discover -s tests -p 'test_*.py'`
- 결과: 112 tests, failures 0

## 실행 모델 체크섬

| 역할 | 파일 | SHA-256 |
|---|---|---|
| Bed segmentation | `bed_seg/runs/bed_seg/weights/best.pt` | `e0f49849fd9b9d432486a7ad29a33b00b330b63ac7c468bb6e94de57d237341e` |
| Pose | `yolo11m-pose.pt` | `29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2` |
| 6-class pose | `my_model_six_check.keras` | `82f4314c0bef77340747cd1b3b2c0941e0f43b0ef7fbf9120d48b6f55fb3f673` |
| Live shadow TCN | `runs/temporal_tcn/gmdcsa24_tcn/model.pt` | `f93f257705acce418900cc85d0e39b4df55a48f47ba3e26377c29537338a71e4` |
| TCN report | `runs/temporal_tcn/gmdcsa24_tcn/report.json` | `f0a7bd8ca47e6f4e51aedf8fb025fd6323200541d0c2d2f770d5170c06e2b60e` |
| Temporal contract | `config/temporal_contract_v2.json` | `968de2300113d1c80ac669bc56710433f63a311e01102009be855d978e20da02` |

## 데이터 read-only 인벤토리

| 경로 | 파일 수 | 바이트 | 주요 내용 | partial 파일 |
|---|---:|---:|---|---:|
| `external_datasets/fallvision` | 6,376 | 83,300,064,759 | CSV 5,864 / MP4 448 / RAR 61 | 0 |
| `external_datasets/omnifall` | 864 | 24,591,236 | Parquet 200 / CSV 79 | 0 |
| `external_datasets/gmdcsa24` | 195 | 2,217,368,940 | MP4 160 / CSV 8 | 0 |
| `external_datasets/manifests` | 10 | 12,244,077 | JSON 8 / CSV 2 | 0 |
| `external_datasets/windows` | 56 | 11,647,523 | NPZ 24 / JSON 32 | 0 |

`dataset_download_status.sh`는 PID와 기존 로그 tail 및 디스크 크기만 읽으며 다운로드를 재개하지 않는다. FallVision/OmniFall download PID는 현재 STOPPED다. 파일 수만으로 archive 완전성을 보장하지 않으며, 압축 CRC와 canonical inventory 검사는 Phase 4에서 수행한다.

## 현재 런타임 계약

- 서버: `run_all_cameras.sh` → `server_all_cameras.py`
- 중앙 호스트/포트: `192.168.0.108:8000`
- 분석 좌표: 90도 clockwise, viewer는 원본
- 사람 gate: box confidence 0.5, 면적 2.5% 이상
- occupied Pose cadence: 약 0.09초
- TCN: observed-only 10Hz, 30행, 109차원
- live cadence 허용 범위: 70~250ms
- track change/gap: buffer reset
- TCN/Fusion: shadow-only, production side effect 없음

## 모델 승격 상태

- 현재 production 승격: `false`
- 주요 사유: 높은 false events/hour, 낮은 pre-onset-ready coverage, FallVision 외부평가 부족
- legacy와 v2 모두 observed-only 비교에서 운영 승격 불가
- FallVision weak mixed checkpoint도 현재 baseline을 확실히 개선하지 못함

## 저장소 상태

- 기존 tracked 수정 파일 다수와 신규 구현 파일 다수가 공존한다.
- 사용자 변경 보존 원칙에 따라 자동 commit, reset, 파일 삭제를 수행하지 않았다.
- 기능별 권장 commit 단위:
  1. capture/viewer/auto ROI
  2. watcher/scheduler/tracking
  3. temporal/fusion/shadow operations
  4. dataset/FallVision tools
  5. tests/docs/config

## 다음 gate

Phase 1에서 서버를 재기동하고 6대 카메라의 capture, ROI, watcher, scheduler, 빈 방 상태를 자동 smoke 검증한다.
