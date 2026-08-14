# Raspberry Pi 단일 침대 Canary Runbook v1

기준일: 2026-08-07  
대상 첫 노드: `192.168.0.161` / `bed_161`  
중앙 제어 API: `http://192.168.0.108:8020`

## 1. 현재 확인 상태

| 항목 | 결과 |
|---|---|
| `.161/.162/.174/.175/.178/.179:22` | 모두 open |
| 각 노드 `:8554` | 모두 open |
| SSH banner | 모두 Debian 12 계열 OpenSSH 9.2 |
| `.161` ping | 정상 |
| `dmc` 비대화식 SSH | 인증 정보가 없어 아직 불가 |
| 중앙 8020 API | 실제 HTTP smoke PASS |
| Pi model bundle | `benchmark_required`, 미승격 |

SSH 인증 전에는 모델을 복사하거나 현재 RTSP 서비스를 변경하지 않는다.

## 2. 인증 직후 첫 읽기 전용 감사

서버에서:

```bash
cd /home/dmc/AI/DMC_POSE
scp scripts/probe_rpi_runtime.sh dmc@192.168.0.161:/tmp/
ssh dmc@192.168.0.161 'bash /tmp/probe_rpi_runtime.sh'
```

반드시 기록할 값:

```text
Pi 모델(4/5)
RAM
저장 공간
OS/architecture
카메라 장치
MediaMTX/FFmpeg 프로세스와 서비스 이름
CPU 온도와 throttling
Hailo/Coral 등 가속기 유무
현재 RTSP 재시작 방식
```

현재 RTSP 서비스는 읽기 전용으로 먼저 확인한다.

```bash
ssh dmc@192.168.0.161 \
  'systemctl --type=service --state=running | grep -Ei "media|rtsp|camera|ffmpeg" || true'
```

## 3. 제어면만 먼저 canary

모델 추론을 올리기 전에 agent heartbeat만 시험한다. Pi에 Python 3와 Pydantic v2가
필요하다. 운영 경로 예시는 `/opt/dmc-pose-edge`, 상태 경로는
`/var/lib/dmc-pose`다.

복사 대상:

```text
edge_contract_v1.py
edge_outbox_v1.py
edge_node_agent.py
config/edge_node_bed_161.example.json
```

Pi에서 설정 복사 후 `spool_path` 디렉터리 권한을 서비스 사용자에게 준다.

```bash
python3 edge_node_agent.py --config edge_node_bed_161.json --once
```

서버 확인:

```bash
curl -s http://127.0.0.1:8020/edge/nodes | python3 -m json.tool
```

기대 결과:

```text
node_id=rpi-bed-161
camera_id=bed_161
runtime_mode=DEGRADED (모델 미연결 시 정상)
model_bundle_version=null
spool_depth=0 (서버 연결 시)
```

서버 연결을 잠시 차단한 동안 `spool_depth > 0`, 복구 후 다시 0이 되는지 확인한다.

## 4. 모델 후보 선택

현재 서버 artifact:

| 역할 | 서버 파일 | 크기 | Pi 판단 |
|---|---|---:|---|
| Bed Seg | `bed_seg/.../best.pt` | 약 6.0 MB | export 후 benchmark |
| Pose | `yolo11m-pose.pt` | 약 42.5 MB | 그대로 승격 금지, n/s 후보 비교 |
| Posture | `my_model_six_check.keras` | 약 0.54 MB | TFLite 후보 |
| TCN | `runs/.../model.pt` | 약 0.24 MB | TorchScript/ONNX 후보 |

Pi 모델·가속기 확인 전에는 backend를 확정하지 않는다.

- 일반 Pi 5 CPU: NCNN 또는 TFLite 우선 비교
- Hailo가 있으면 Hailo 변환 가능 모델을 별도 비교
- Pose 정확도 손실이 큰 경우 watcher는 Pi, Pose/2차 판단은 서버로 fallback

## 5. 단일 카메라 benchmark 항목

각 후보를 `bed_161` 하나에서 최소 30분 측정한다.

```text
capture FPS p50/p95
watcher FPS p50/p95
Pose latency p50/p95
TCN latency p50/p95
end-to-end result latency p50/p95
CPU 사용률
RAM RSS와 증가율
온도와 throttling
outbox pending/drop/error
RTSP viewer 끊김/지연
```

상태별 목표:

```text
EMPTY: watcher 유지, Pose probe 저주기
OCCUPIED: 동일 track 실제 관측 약 10 Hz
BURST: 빠른 움직임 동안 분석 주기 상승
DEGRADED: ROI/카메라/모델 문제를 숨기지 않음
```

## 6. 행동 검증 순서

각 행동 시작 전 5초 서 있고, 행동을 5~10초 유지한 후 나온다.

1. 사람 없음 5분
2. 진입 후 서기
3. 침대 가장자리 앉기
4. 정상 눕기 / 빠르게 눕기
5. 물건 줍기
6. 쪼그리기
7. 바닥 앉기 / 바닥 눕기
8. 정상 침대 이탈
9. 안전 장비를 사용한 staged fall

검증 목적은 단순 class 표시가 아니다.

- 사람 진입 시 `EMPTY → OCCUPIED`
- 같은 track 관측이 30개가 되어 TCN ready
- track/gap에서 ready reset
- 정상 행동은 event로 승격되지 않음
- staged fall은 candidate와 event 근거가 남음
- Viewer와 추론 timestamp가 독립적으로 갱신됨

## 7. 승격과 rollback

처음 bundle은 반드시 `shadow`다. 다음 조건 전에는 `production`으로 바꾸지 않는다.

```text
checksum 검증 PASS
Pi smoke inference PASS
30분 자원 benchmark PASS
중앙-vs-Pi 동일 영상 결과 허용 오차 PASS
24시간 shadow PASS
네트워크 단절/복구 PASS
재부팅 자동복구 PASS
rollback PASS
```

새 bundle 활성화 실패 시 `edge_bundle_manager.py`의 `previous` 링크로 즉시 되돌린다.
중앙 8000 reference 파이프라인은 canary 완료까지 유지한다.

## 8. 다음 실제 현장 게이트

현재 유일한 외부 의존 작업은 `.161` SSH 인증이다. 인증이 가능해지는 즉시 다음을
연속 실행한다.

```text
읽기 전용 Pi probe
→ 현재 RTSP 서비스 보존 확인
→ heartbeat-only agent
→ 단절/재전송 시험
→ 모델 backend benchmark
→ shadow inference
```
