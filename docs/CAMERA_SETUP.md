# RTSP 카메라 설정 가이드

## 🎥 설치된 카메라

```
┌─────────────────────────────────────────────────────────┐
│           침대 모니터링 RTSP 카메라 네트워크             │
└─────────────────────────────────────────────────────────┘

1️⃣  Bed 1 (Main)
   Camera ID: raspi_bed_001
   IP: 192.168.0.161:8554
   Path: /stream
   URL: rtsp://192.168.0.161:8554/stream
   Status: ✅ Active
   
2️⃣  Bed 2 (Secondary)
   Camera ID: raspi_bed_002
   IP: 192.168.0.174:8554
   Path: /stream
   URL: rtsp://192.168.0.174:8554/stream
   Status: ✅ Active
   
3️⃣  Bed 3 (Testing)
   Camera ID: raspi_bed_003
   IP: 192.168.0.178:8554
   Path: /stream
   URL: rtsp://192.168.0.178:8554/stream
   Status: 🧪 Testing
   
4️⃣  Bed 4 (Backup)
   Camera ID: raspi_bed_004
   IP: 192.168.0.179:8554
   Path: /stream
   URL: rtsp://192.168.0.179:8554/stream
   Status: ⏸️ Inactive
```

## 🚀 사용 방법

### 방법 1: 직접 카메라 선택 (추천)

```bash
# Bed 1 (기본)
bash /home/dmc/pose-sixclass/run_camera_161.sh

# Bed 2
bash /home/dmc/pose-sixclass/run_camera_174.sh

# Bed 3
bash /home/dmc/pose-sixclass/run_camera_178.sh

# Bed 4
bash /home/dmc/pose-sixclass/run_camera_179.sh
```

### 방법 2: 환경 변수로 선택

```bash
# 특정 카메라 선택
export POSE_CAMERA_ID=raspi_bed_002
python /home/dmc/pose-sixclass/server_parallel.py

# RTSP URL 직접 지정 (더 우선)
export POSE_RTSP_URL="rtsp://192.168.0.178:8554/stream"
python /home/dmc/pose-sixclass/server_parallel.py
```

### 방법 3: 설정 파일 수정

`config/cameras.yaml` 파일의 `default_camera`를 변경:

```yaml
default_camera: "raspi_bed_002"  # 기본 카메라 변경
```

## 📊 API 엔드포인트 (모든 카메라 동일)

```
GET /status           - 현재 분석 상태
GET /health           - 서버 헬스 체크
GET /viewer           - HTML 뷰어
```

### /status 응답 예시

```json
{
  "in_bed": "YES",
  "pose": "front_lying",
  "pose_conf": 0.95,
  "timestamp": "2026-07-20T11:00:00.123456",
  "latency_ms": 38.5,
  "frame_age_ms": 2.1,
  "pipeline_fps": 24.8
}
```

## 🔍 카메라 연결 진단

### ffprobe로 확인

```bash
# Bed 1 확인
ffprobe -rtsp_transport tcp \
  -of default=noprint_wrappers=1 \
  rtsp://192.168.0.161:8554/stream

# Bed 2 확인
ffprobe -rtsp_transport tcp \
  -of default=noprint_wrappers=1 \
  rtsp://192.168.0.174:8554/stream
```

### OpenCV로 확인

```python
import cv2

urls = [
    "rtsp://192.168.0.161:8554/stream",
    "rtsp://192.168.0.174:8554/stream",
    "rtsp://192.168.0.178:8554/stream",
    "rtsp://192.168.0.179:8554/stream",
]

for url in urls:
    cap = cv2.VideoCapture(url)
    ret, frame = cap.read()
    print(f"{url}: {'✅ OK' if ret else '❌ FAIL'}")
    cap.release()
```

## 🖥️ 다중 카메라 동시 모니터링

여러 카메라를 동시에 실행 (다른 포트):

```bash
# 터미널 1
export POSE_CAMERA_ID=raspi_bed_001
export FLASK_PORT=8000
python server_parallel.py

# 터미널 2
export POSE_CAMERA_ID=raspi_bed_002
export FLASK_PORT=8001
python server_parallel.py

# 터미널 3
export POSE_CAMERA_ID=raspi_bed_003
export FLASK_PORT=8002
python server_parallel.py
```

viewer에 접속:
- http://localhost:8000/viewer (Bed 1)
- http://localhost:8001/viewer (Bed 2)
- http://localhost:8002/viewer (Bed 3)

## 🔧 포트 변경 (필요시)

`server_parallel.py` 하단:

```python
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get('POSE_SERVER_PORT', '8000'))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

실행:

```bash
export POSE_CAMERA_ID=raspi_bed_002
export POSE_SERVER_PORT=8001
python server_parallel.py
```

## ⚙️ 환경 변수 전체

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `POSE_CAMERA_ID` | `raspi_bed_001` | 카메라 선택 ID |
| `POSE_RTSP_URL` | 위 값에서 결정 | RTSP URL (직접 지정) |
| `POSE_YOLO_DEVICE` | `0` | GPU 디바이스 |
| `POSE_PARALLEL_WORKERS` | `2` | 병렬 스레드 수 |
| `POSE_FRAME_WIDTH` | `640` | 프레임 해상도 |
| `POSE_SEG_EVERY` | `3` | Seg 주기 |
| `POSE_SERVER_PORT` | `8000` | FastAPI 포트 |

## 📝 설정 파일

- **cameras.yaml**: 카메라 메타데이터
- **config/rooms/room_*.json**: 방별 ROI/Rail 설정

### rooms/room_174.json 예시

```json
{
  "room_id": "room_174",
  "rtsp_url": "rtsp://192.168.0.174:8554/stream",
  "bed_roi_path": "/home/dmc/pose-sixclass/config/rooms/room_174_roi.json",
  "rail_config_path": "/home/dmc/pose-sixclass/config/rooms/room_174_rail.json"
}
```

## 🚨 문제 해결

### RTSP 연결 실패
```
[ERROR] RTSP 열기 실패: rtsp://...
```

**해결:**
```bash
# 1. 카메라 재부팅
ssh dmc@192.168.0.174
reboot

# 2. RTSP 포트 확인
netstat -tlnp | grep 8554

# 3. ffprobe 테스트
ffprobe rtsp://192.168.0.174:8554/stream
```

### 낮은 FPS
```
pipeline_fps: 8.0 (낮음)
```

**해결:**
```bash
# 프레임 해상도 감소
export POSE_FRAME_WIDTH=480

# Seg 주기 증가
export POSE_SEG_EVERY=5

# GPU 명시
export POSE_YOLO_DEVICE=0
```

---

**작성일:** 2026-07-20  
**카메라 버전:** 4 beds (161, 174, 178, 179)
