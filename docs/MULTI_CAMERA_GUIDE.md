# 멀티 카메라 병렬 추론 가이드

## 🎥 카메라 구성

**6개 카메라 설치됨:**

| ID | 이름 | IP | 포트 | RTSP URL |
|----|------|-----|------|----------|
| bed_161 | Bed 161 | 192.168.0.161 | 8554 | `rtsp://192.168.0.161:8554/stream` |
| bed_162 | Bed 162 | 192.168.0.162 | 8554 | `rtsp://192.168.0.162:8554/stream` |
| bed_174 | Bed 174 | 192.168.0.174 | 8554 | `rtsp://192.168.0.174:8554/stream` |
| bed_175 | Bed 175 | 192.168.0.175 | 8554 | `rtsp://192.168.0.175:8554/stream` |
| bed_178 | Bed 178 | 192.168.0.178 | 8554 | `rtsp://192.168.0.178:8554/stream` |
| bed_179 | Bed 179 | 192.168.0.179 | 8554 | `rtsp://192.168.0.179:8554/stream` |

## 🚀 빠른 시작

### 모든 카메라 동시 실행

```bash
bash /home/dmc/pose-sixclass/run_all_cameras.sh
```

### 웹 대시보드

브라우저에서 접속:
```
http://localhost:8000/viewer
```

### API 상태 확인

```bash
# 모든 카메라
curl http://localhost:8000/status

# 특정 카메라 (예: bed_161)
curl http://localhost:8000/status/bed_161

# 헬스 체크
curl http://localhost:8000/health
```

## 📊 병렬 처리 구조

```
┌─────────────────────────────────────────────────────────┐
│           RTSP 카메라 입력 (6개 병렬)                  │
└─────────────────────────────────────────────────────────┘
              │  │  │  │  │  │
              ▼  ▼  ▼  ▼  ▼  ▼
┌─────────────────────────────────────────────────────────┐
│  GPU 병렬 추론 풀 (ThreadPoolExecutor, 3 workers)     │
│  ┌──────────────────────────────────────────────────┐   │
│  │ • YOLO Seg (침대 감지)     ║ (병렬 실행)          │   │
│  │ • YOLO Pose (포즈 추정)    ║                    │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│  Keras 6-class 분류 (CPU)                              │
│  front_lying, prone_back, side_near, side_far, ...     │
└─────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI 결과 병합 & 웹소켓 전송                      │
│  • /status: 모든 카메라 상태                          │
│  • /viewer: 실시간 대시보드                          │
└─────────────────────────────────────────────────────────┘
```

## ⚙️ 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `POSE_YOLO_DEVICE` | `0` | GPU 디바이스 (0, 1, 2 등) |
| `POSE_PARALLEL_WORKERS` | `3` | ThreadPool 스레드 수 |
| `POSE_FRAME_WIDTH` | `640` | 프레임 너비 (픽셀) |
| `POSE_SEG_EVERY` | `3` | N프레임마다 segmentation |
| `POSE_SERVER_PORT` | `8000` | FastAPI 포트 |

### 커스텀 설정 실행

```bash
# 프레임 해상도 감소 (빠른 처리)
export POSE_FRAME_WIDTH=480
bash /home/dmc/pose-sixclass/run_all_cameras.sh

# 워커 수 증가 (높은 동시성)
export POSE_PARALLEL_WORKERS=5
bash /home/dmc/pose-sixclass/run_all_cameras.sh

# 다른 포트 사용
export POSE_SERVER_PORT=8001
bash /home/dmc/pose-sixclass/run_all_cameras.sh
```

## 📡 API 엔드포인트

### GET /status
**모든 카메라의 실시간 상태**

```json
{
  "bed_161": {
    "camera_id": "bed_161",
    "camera_name": "Bed 161",
    "in_bed": "YES",
    "pose": "front_lying",
    "pose_conf": 0.95,
    "latency_ms": 38.5,
    "pipeline_fps": 24.8,
    "status": "analyzing",
    "frame_count": 1234,
    "timestamp": "2026-07-20T11:06:00.123456"
  },
  "bed_162": { ... },
  ...
}
```

### GET /status/{camera_id}
**특정 카메라의 상태**

```bash
curl http://localhost:8000/status/bed_161
```

### GET /health
**서버 헬스 체크**

```json
{
  "server": "ok",
  "analysis_running": true,
  "total_cameras": 6,
  "online_cameras": 6,
  "cameras": ["bed_161", "bed_162", "bed_174", "bed_175", "bed_178", "bed_179"]
}
```

### GET /viewer
**HTML 대시보드** - 웹 브라우저에서 실시간 모니터링

## 🎬 실시간 모니터링

### 웹 뷰어 기능

- **6개 카메라 타일**: 각 카메라 상태 실시간 표시
- **통계**: 온라인 카메라, 분석 중인 카메라 수
- **메트릭**:
  - In Bed: YES/NO
  - Pose: front_lying, prone_back, side_near, side_far, sitting_center, sitting_edge
  - Confidence: 분류 신뢰도 (%)
  - FPS: 초당 처리 프레임
  - Latency: 추론 지연시간 (ms)
  - Frames: 누적 처리 프레임 수
  - Status: analyzing/idle/connecting/disconnected

## 🔍 문제 해결

### 카메라 연결 실패

```
[❌ bed_161] RTSP 열기 실패
```

**해결 방법:**
1. 카메라 IP 확인: `ping 192.168.0.161`
2. RTSP 연결 테스트:
   ```bash
   ffprobe -rtsp_transport tcp rtsp://192.168.0.161:8554/stream
   ```
3. 카메라 재부팅: SSH로 접속 후 `reboot`

### 낮은 FPS

```
pipeline_fps: 8.0 (낮음)
```

**해결 방법:**
```bash
# 1. 프레임 해상도 감소
export POSE_FRAME_WIDTH=480

# 2. Segmentation 주기 증가
export POSE_SEG_EVERY=5

# 3. 워커 수 조정
export POSE_PARALLEL_WORKERS=2
```

### GPU 메모리 부족

```
torch.cuda.OutOfMemoryError
```

**해결 방법:**
```bash
# 프레임 해상도 감소
export POSE_FRAME_WIDTH=480

# 또는 배치 처리 단위 감소
export POSE_PARALLEL_WORKERS=1
```

## 📊 성능 기준

### 목표 성능 (6 카메라)

| 메트릭 | 목표 | 현재 |
|--------|------|------|
| 프레임 처리 | 25+ fps | ? |
| 지연시간 | < 50ms | ? |
| GPU 사용률 | 70-90% | ? |
| 메모리 | < 8GB | ? |

## 🔧 커스텀 설정

### config/cameras.json (선택적)

자동으로 생성되는 설정 파일. 필요시 수정:

```json
{
  "cameras": [
    {
      "name": "room1",
      "source": "rtsp://192.168.0.161:8554/stream",
      "device": "0"
    },
    {
      "name": "room2",
      "source": "rtsp://192.168.0.162:8554/stream",
      "device": "0"
    }
  ]
}
```

## 📝 로그 확인

### 서버 시작 로그

```
[🚀] Multi-Camera Parallel Pipeline 시작
[📡] 감지된 카메라: 6개
     - bed_161: Bed 161 (rtsp://192.168.0.161:8554/stream)
     - bed_162: Bed 162 (rtsp://192.168.0.162:8554/stream)
     ...
[⚡] YOLO GPU=0 | Parallel Workers=3
```

### 카메라별 로그

```
[🎥 bed_161] Bed 161 분석 시작
[🎥 bed_161] RTSP: rtsp://192.168.0.161:8554/stream
```

## 🎯 다음 단계

1. **MJPEG 스트림 추가** (`/video/{camera_id}`)
2. **데이터 저장** (CSV, 데이터베이스)
3. **실시간 알림** (낙상 위험 감지)
4. **대시보드 고급화** (웹소켓, 실시간 차트)

---

**생성일:** 2026-07-20  
**버전:** 1.0 - Multi-Camera Parallel Pipeline
**카메라:** 6개 (161, 162, 174, 175, 178, 179)
