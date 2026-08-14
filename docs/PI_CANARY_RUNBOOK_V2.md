# Raspberry Pi 단일 침대 Canary Runbook v2

기준일: 2026-08-07  
첫 대상: `192.168.0.161` / `bed_161`  
중앙 API: `http://192.168.0.108:8020`

## 0. 현재 게이트

포트 `22`와 `8554`는 열려 있고 RTSP 영상은 읽힌다. Linux 사용자와 SSH 인증
정보는 아직 확인되지 않았다. `dmc`나 `pi`를 기본값으로 가정하거나 반복 로그인하지
않는다. 제작자로부터 정확한 사용자와 SSH 키/비밀번호를 받은 뒤 진행한다.

## 1. 읽기 전용 기기 감사

아래의 `<linux-user>`는 제작자가 알려준 값으로 치환한다.

```bash
scp scripts/probe_rpi_runtime.sh <linux-user>@192.168.0.161:/tmp/
ssh <linux-user>@192.168.0.161 'bash /tmp/probe_rpi_runtime.sh'
```

Pi 모델, RAM, architecture, 온도/throttle, 가속기, 카메라 장치, 현재 MediaMTX/
FFmpeg/systemd 서비스 이름을 기록한다. 이 단계에서는 기존 서비스를 변경하지 않는다.

## 2. handoff 검증과 설치

중앙 `~/Downloads`의 ZIP과 `.sha256`을 전달하고 수신 측에서 먼저 검증한다.

```bash
sha256sum -c DMC_POSE_edge_handoff_*.zip.sha256
unzip DMC_POSE_edge_handoff_*.zip -d /tmp/dmc-edge-handoff
```

운영 파일은 `/opt/dmc_pose`, 상태는 `/var/lib/dmc_pose`, 설정과 secret은
`/etc/dmc_pose`를 권장한다. ZIP에는 secret이 없다.

## 3. API token 별도 공급

중앙의 token 값을 화면이나 로그에 출력하지 않는다. SSH/SCP로 별도 전달하고 Pi에서
권한을 고정한다.

```bash
sudo install -o dmc-pose -g dmc-pose -m 0600 /tmp/edge_api_token \
  /etc/dmc_pose/edge_api_token
```

현재 구현은 하나의 bearer token을 공유하므로, 첫 canary 후 장치별 token 또는 mTLS로
확장한다. RTSP 사용자/비밀번호는 node config와 API payload에 넣지 않는다.

## 4. heartbeat-only canary

`config/edge_node_secure.example.json`을 복사해 node/camera만 맞춘다. token 문자열 대신
반드시 `api_token_file`을 사용한다.

```bash
python3 edge_node_agent.py --config /etc/dmc_pose/node.json --once
```

중앙에서 인증해 확인한다.

```bash
TOKEN_FILE=runtime_data/edge_control/api_token
curl -H "Authorization: Bearer $(<"$TOKEN_FILE")" \
  http://127.0.0.1:8020/edge/nodes
```

서버 연결을 잠시 끊었을 때 outbox pending이 증가하고, 복구 후 0으로 내려와야 한다.

## 5. 모델은 staging만 수행

```bash
python3 scripts/pull_edge_bundle.py \
  --server http://192.168.0.108:8020 \
  --token-file /etc/dmc_pose/edge_api_token \
  --install-root /var/lib/dmc_pose/models
```

현재 manifest는 `benchmark_required`이므로 `--activate`를 붙이면 실패하는 것이 정상이다.
5개 파일의 크기와 SHA-256가 모두 맞아야 staging된다.

## 6. benchmark와 승격

한 카메라에서 최소 30분 동안 다음을 측정한다.

- RTSP 끊김 및 capture FPS
- EMPTY watcher/probe 부하
- OCCUPIED Pose p50/p95, 실제 관측 10 Hz
- TCN 및 end-to-end p50/p95
- CPU/RAM/온도/throttling
- 정상 눕기·줍기·쪼그리기·바닥앉기 오탐
- 빠른 동작과 staged fall candidate coverage

Pose ONNX는 `conf=0.1`, Bed Seg는 `conf=0.5`에서 시작하되 현장 calibration한다.
결과가 기준을 만족하면 중앙에서 새 manifest 버전을 `shadow`로 발행한다.
`benchmark_required` 파일을 현장에서 임의 편집하지 않는다.

## 7. 장애 시험 순서

```text
네트워크 2분 단절 → 로컬 추론 지속/outbox 증가 → 복구/순서대로 drain
중앙 8020 재시작 → Pi 추론 지속 → heartbeat 복구
Pi 재부팅 → RTSP와 agent 자동복구
손상 bundle → checksum 거부/현재 bundle 유지
새 shadow 실패 → previous symlink rollback
```

이 모든 과정에서 중앙 `:8000` reference 파이프라인을 유지한다.

