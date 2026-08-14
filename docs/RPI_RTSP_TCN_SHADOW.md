# RPi RTSP → 중앙 서버 TCN shadow 운영

## 확정 구조

```text
RPi + 카메라 (영상 획득/인코딩)
  └─ RTSP: rtsp://<rpi-ip>:8554/stream
       └─ 중앙 서버 /home/dmc/AI/DMC_POSE
          ├─ YOLO bed segmentation (GPU)
          ├─ YOLO pose (GPU)
          ├─ 기존 Keras 6-class (CPU)
          ├─ 10 Hz, 3초, 109-feature 카메라별 버퍼
          ├─ 공유 causal TCN (기본 CPU)
          └─ FastAPI /status + /viewer
```

RPi에는 주 가중치를 배포하지 않는다. RPi의 책임은 카메라 연결과 RTSP 송출이다. 중앙 서버가 여섯 RTSP 스트림을 받고 모든 추론 결과를 합친다. 정확한 프로토콜 표기는 `RSTP`가 아니라 `RTSP`다.

## 현재 안전 정책

TCN은 **shadow-only**다. `tcn_fall_probability`와 `tcn_alert_candidate`를 내보내지만 기존 `fall_score`, `fall_status`, 실제 알림을 변경하지 않는다. GMDCSA-24 테스트 결과는 모델 가능성을 확인하기에는 충분하지만 병상 자체 데이터에서 오탐률을 검증하기 전 운영 경보에 연결하면 안 된다.

## 중앙 서버 실행

```bash
cd /home/dmc/AI/DMC_POSE
bash run_all_cameras.sh
```

기본 모델:

```text
runs/temporal_tcn/gmdcsa24_tcn/model.pt
runs/temporal_tcn/gmdcsa24_tcn/report.json
```

설정값:

| 환경 변수 | 기본값 | 의미 |
|---|---|---|
| `POSE_TCN_SHADOW` | `1` | TCN shadow 사용 |
| `POSE_TCN_MODEL` | 위 `model.pt` | 체크포인트 경로 |
| `POSE_TCN_REPORT` | 위 `report.json` | 검증 threshold 경로 |
| `POSE_TCN_DEVICE` | `cpu` | TCN 장치. YOLO GPU 경합 방지를 위해 우선 CPU |
| `POSE_TCN_THRESHOLD` | 미설정 | 설정 시 report threshold 덮어쓰기 |

TCN 로딩이 실패해도 서버는 기존 판정으로 계속 시작하며 로그에 shadow 비활성 원인을 남긴다.

## 결과 확인

```bash
curl -s http://127.0.0.1:8000/status | python3 -m json.tool
```

카메라별 추가 필드:

| 필드 | 의미 |
|---|---|
| `tcn_shadow_enabled` | 모델이 정상 로드되어 해당 카메라 runner가 생성됨 |
| `tcn_shadow_ready` | 유효 포즈 30개(약 3초) 버퍼 확보 |
| `tcn_fall_probability` | 최신 TCN 낙상 확률, 0~1 |
| `tcn_alert_candidate` | threshold 이상 결과가 2회 연속 나온 shadow 후보 |
| `tcn_threshold` | 현재 판정 threshold |
| `tcn_samples` | 현재 카메라 버퍼의 샘플 수, 최대 30 |
| `tcn_prediction_count` | 서버 시작 후 해당 카메라의 TCN 추론 횟수 |

TCN은 10 Hz로 샘플링하고 30개가 모인 뒤 0.5초마다 추론한다. 포즈 입력이 0.5초 넘게 끊기면 과거와 현재를 잘못 잇지 않도록 해당 카메라 버퍼를 비운다.

## RPi 및 네트워크 확인

중앙 서버에서 각 RPi 스트림을 먼저 검사한다.

```bash
ffprobe -v error -rtsp_transport tcp \
  -show_entries stream=codec_name,width,height,r_frame_rate \
  -of default=noprint_wrappers=1 \
  rtsp://192.168.0.161:8554/stream
```

전체 진단은 다음 스크립트를 사용한다.

```bash
cd /home/dmc/AI/DMC_POSE
POSE_RTSP_HOST=192.168.0.161 bash scripts/rtsp_diagnose.sh
```

권장 조건은 유선 LAN, 고정 IP, H.264, TCP RTSP, 카메라 시각 동기화다. 실제 성능 판정 시 `/status`의 `pipeline_fps`도 함께 기록한다. 카메라별 유효 pose 처리율이 10 Hz를 장시간 밑돌면 모델이 학습한 시간축과 달라지므로, 먼저 segmentation 주기·해상도·GPU 스케줄링을 조정해야 한다.

## 다음 승격 조건

1. 병상 자체 정상 영상으로 장시간 shadow 로그를 수집한다.
2. 물건 줍기, 침대 가장자리 앉기, 눕기, 가림을 하드 네거티브로 평가한다.
3. 자체 낙상 전이 영상 또는 검수된 FallVision 병상 영상으로 재현율과 지연을 측정한다.
4. 카메라별/전체 `false events per hour`가 운영 기준을 만족한 뒤에만 알림과 결합한다.
5. 결합 시에도 기존 rule score와 TCN을 즉시 대체하지 말고 AND/확인 게이트부터 A/B 검증한다.

TSFM 계열은 이번 실시간 3초 이진 분류의 주 모델로 쓰지 않는다. 현재 causal TCN이 작고 지연이 낮으며, 해당 문제에 직접 학습되어 있다. 추후 자체 데이터 규모가 충분해지면 분류용 사전학습 시계열 모델을 별도 benchmark로 비교한다.
