# DMC POSE 소스 인수인계

## 패키지 목적

전송 ZIP은 현재 작업 트리의 소스, 테스트, 문서, 비민감 설정을 전달하는 스냅샷입니다. Git 이력 전체, 학습 데이터, 실제 카메라 영상, 운영 로그, 모델 가중치 백업은 아닙니다.

## 포함

- Python 및 shell 소스
- `tests/`, `scripts/`
- `docs/`와 Mermaid
- 공개 가능한 JSON 설정과 예제
- 의존성 목록 및 실행 스크립트
- ZIP 내부 `HANDOFF_MANIFEST.txt`

## 제외

- `.git/` — 현재 30GB의 개발 이력과 Git LFS 객체
- `external_datasets/`, `Raw_data/` — 공개/자체 학습 데이터
- `runtime_data/`, `runs/` — 실시간 feature 기록과 학습 결과
- `*.pt`, `*.keras`, `*.onnx` 등 모델 가중치
- `config/cameras.yaml`, `.env*` — RTSP 주소와 인증정보
- 실제 카메라 캡처, ROI 이미지, 로그와 캐시
- 실행 바이너리 및 압축된 과거 모니터 로그

## 수신 후 별도 준비

1. `config/cameras.yaml`
2. 침대 segmentation, pose, six-class, TCN 가중치
3. Python 환경 또는 `requirements-pose-cuda.txt` 기반 환경
4. 필요할 경우 데이터셋과 `runtime_data`

민감 설정과 가중치는 승인된 별도 채널로 전달합니다.

## 기본 실행 확인

```bash
cd DMC_POSE
/home/dmc/anaconda3/envs/pose-cuda/bin/python -m unittest discover -s tests -v
./run_all_cameras.sh
```

```text
Viewer       http://<server-ip>:8000/viewer
Status       http://<server-ip>:8000/status
Recorder     http://<server-ip>:8000/recorder/status
Calibration  http://<server-ip>:8000/calibration/status
```

## 현재 모델 정책

- 중앙 서버가 모든 가중치와 추론을 담당합니다.
- RPi는 침대당 한 대이며 촬영과 H264 RTSP 송출만 담당합니다.
- Viewer 영상 갱신은 AI 추론 주기와 분리됩니다.
- 사람이 없으면 고비용 pose 추론을 낮추고 경량 watcher를 유지합니다.
- TCN은 10Hz track별 입력을 사용하지만 현재는 shadow-only입니다.
- fusion 정책은 `hybrid_v2_structural_confirm`입니다.
- TCN과 motion만으로 alert를 확정하지 않고 독립적인 공간 또는 운동학 증거를 요구합니다.

정확한 실행 스냅샷은 `runtime_artifact.json`을 확인합니다.
