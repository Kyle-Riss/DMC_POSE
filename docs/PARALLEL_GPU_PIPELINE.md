# GPU 병렬 추론 파이프라인 (server_parallel.py)

## 개요

**Parallel GPU Pipeline** - RTSP에서 받은 프레임을 **병렬로 처리**하여 추론 속도 대폭 개선

### 기존 vs 신규

```
기존 (순차):
RTSP → [Seg] → [Pose] → [Keras] → Output
       ~20ms   ~30ms   ~10ms    ~60ms total

신규 (병렬):
RTSP → [Seg ∥ Pose] (동시) → [Keras] → Output
       ~30ms (병렬)         ~10ms     ~40ms total
       
🚀 예상 개선: 60ms → 40ms (33% 빠름)
```

## 아키텍처

### 1. ParallelInferencePool

```python
class ParallelInferencePool:
    - seg_model: YOLO 침대 segmentation
    - pose_model: YOLO 포즈 추정
    - executor: ThreadPoolExecutor (workers=2)
    
infer_parallel(frame, run_seg, run_pose)
    → concurrent.futures로 seg/pose 동시 실행
    → Lock으로 GPU 메모리 접근 보호
    → 결과 반환
```

### 2. RTSP 프레임 루프

```
while analysis_running:
    frame = cap.read()  # RTSP
    
    # ★ 병렬 처리
    seg_res, pose_res = inference_pool.infer_parallel(frame)
    
    # Segmentation 결과
    bed = extract_bed_detection(seg_res)
    bed = apply_bed_roi(bed)
    
    # Pose 결과
    if pose_detected:
        in_bed = check_in_bed(pose_keypoint, bed_mask)
        pose_class = keras_clf.predict(keypoints)
    
    # 상태 업데이트
    current_state.update(in_bed, pose_class, latency)
```

### 3. 동시성 제어

**Lock 메커니즘:**
```python
with self.lock:
    result = self.seg_model.predict(...)  # GPU 메모리 독점
```

- Seg와 Pose가 같은 GPU(0)를 사용할 때는 직렬화됨
- **추천:** GPU 2개 환경에서는 `POSE_YOLO_DEVICE_SEG=0`, `POSE_YOLO_DEVICE_POSE=1` 분할

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `POSE_YOLO_DEVICE` | `0` | YOLO GPU (seg + pose 공유) |
| `POSE_PARALLEL_WORKERS` | `2` | ThreadPoolExecutor 스레드 수 |
| `POSE_FRAME_QUEUE_SIZE` | `30` | 프레임 큐 크기 (미사용) |
| `POSE_RTSP_URL` | `rtsp://192.168.0.161:8554/stream` | RTSP 카메라 |
| `POSE_SEG_EVERY` | `3` | N프레임마다 seg 실행 |
| `POSE_USE_BED_ROI` | `1` | ROI 클리핑 활성화 |

## 실행

```bash
bash /home/dmc/pose-sixclass/run_parallel_server.sh

# 또는 직접
python /home/dmc/pose-sixclass/server_parallel.py
```

## API 엔드포인트

| URL | 반환 |
|-----|------|
| `GET /status` | `in_bed`, `pose`, `pose_conf`, `latency_ms`, `pipeline_fps` |
| `GET /health` | `server`, `analysis_running` |
| `GET /viewer` | HTML 뷰어 |

## 성능 최적화

### GPU 분할 (권장)

**2-GPU 환경:**
```bash
export POSE_YOLO_DEVICE_SEG=0
export POSE_YOLO_DEVICE_POSE=1
```

server_parallel.py에 다음 수정 추가:
```python
def infer_seg(self, frame, ...):
    device = os.environ.get('POSE_YOLO_DEVICE_SEG', self.device)
    result = self.seg_model.predict(..., device=device)
    
def infer_pose(self, frame, ...):
    device = os.environ.get('POSE_YOLO_DEVICE_POSE', self.device)
    result = self.pose_model.predict(..., device=device)
```

### 프레임 해상도 튜닝

```bash
# 더 빠름 (정확도 ↓)
export POSE_FRAME_WIDTH=480

# 더 정확함 (속도 ↓)
export POSE_FRAME_WIDTH=800
```

### Seg 주기 조정

```bash
# 1프레임마다 seg (가장 정확, 느림)
export POSE_SEG_EVERY=1

# 5프레임마다 seg (균형)
export POSE_SEG_EVERY=5
```

## 마이그레이션

### 기존 server.py → server_parallel.py

1. 모든 엔드포인트 호환 (`/status`, `/health`, `/viewer`)
2. `AnalysisState` 동일
3. bed_monitor, bed_roi 모듈 호환

### 전환 과정

```bash
# 1. 기존 서버 중지
pkill -f "python.*server.py"

# 2. 새 서버 시작
bash /home/dmc/pose-sixclass/run_parallel_server.sh

# 3. 모니터링
curl http://localhost:8000/status
```

## 트러블슈팅

### RTSP 연결 실패
```
[WARNING] RTSP 프레임 읽기 실패, 재연결 시도...
```
→ 카메라 RTSP URL 확인, 네트워크 연결

### GPU 메모리 부족
```
torch.cuda.OutOfMemoryError
```
→ `POSE_FRAME_WIDTH` 감소 또는 GPU 분할

### Low FPS
```
pipeline_fps: 5.0 (낮음)
```
→ `POSE_SEG_EVERY` 증가, POSE_FRAME_WIDTH 감소

## 다음 단계

- [ ] GPU 분할 구현 (seg/pose 별도 GPU)
- [ ] MJPEG 스트림 추가 (`/video`)
- [ ] 배치 추론 지원
- [ ] 성능 프로파일링

---

**작성일:** 2026-07-20  
**버전:** 1.0 - Parallel GPU Pipeline
