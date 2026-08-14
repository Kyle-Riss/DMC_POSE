# FastAPI 실시간 포즈 모니터링 API 서버 계획

## 목표

`run_pose.py`의 포즈 분석 결과(In Bed 여부, Pose 라벨)를 FastAPI 서버를 통해 클라이언트가 **요청할 때마다** 최신 값을 JSON으로 반환한다.

---

## 아키텍처

```
[RTSP 카메라] → [백그라운드 분석 스레드] → [공유 상태 변수] ← [FastAPI 엔드포인트] ← [클라이언트 요청]
```

- **백그라운드 스레드**: RTSP 스트림을 지속적으로 읽으며 YOLO + Keras 모델로 분석, 결과를 공유 변수에 갱신
- **FastAPI 서버**: 클라이언트 요청 시 공유 변수에서 최신 값을 읽어 JSON 응답

---

## 구현 단계

### 1단계: 프로젝트 구조 정리

```
pose/
├── run_pose.py          # 기존 코드 (유지)
├── server.py            # FastAPI 서버 (신규)
├── my_model.keras
├── yolo11m-pose.pt
├── yolo11n-seg.pt
└── requirements.txt     # fastapi, uvicorn 추가
```

### 2단계: 공유 상태 설계

스레드 안전한 공유 딕셔너리를 사용하여 분석 결과를 저장한다.

```python
from threading import Lock

shared_state = {
    "in_bed": "NO",
    "pose": "None",
    "timestamp": None,       # 마지막 갱신 시각
    "is_running": False       # 분석 스레드 동작 여부
}
state_lock = Lock()
```

### 3단계: 백그라운드 분석 스레드 구현

`run_pose.py`의 `main()` 로직을 스레드 함수로 변환한다.

- RTSP 스트림에서 프레임을 읽고 분석
- 결과를 `state_lock` 으로 보호하며 `shared_state`에 기록
- `cv2.imshow` 등 GUI 코드는 서버 모드에서 **제거** (headless 운영)

### 4단계: FastAPI 엔드포인트 구현

| 메서드 | 경로 | 설명 | 응답 예시 |
|--------|------|------|-----------|
| `GET` | `/status` | 최신 포즈 상태 조회 | `{"in_bed": "YES", "pose": "p03", "timestamp": "..."}` |
| `GET` | `/health` | 서버 및 분석 스레드 상태 확인 | `{"server": "ok", "analysis_running": true}` |

#### 응답 모델 (Pydantic)

```python
from pydantic import BaseModel
from datetime import datetime

class PoseStatus(BaseModel):
    in_bed: str          # "YES" 또는 "NO"
    pose: str            # "p01" ~ "p12" 또는 "None"
    timestamp: datetime | None
```

### 5단계: 서버 시작 시 흐름

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
import threading

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 → 분석 스레드 시작
    thread = threading.Thread(target=analysis_loop, daemon=True)
    thread.start()
    yield
    # 서버 종료 → 스레드 정리

app = FastAPI(lifespan=lifespan)
```

### 6단계: 의존성 설치

```bash
pip install fastapi uvicorn
```

### 7단계: 실행

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## 핵심 고려사항

| 항목 | 내용 |
|------|------|
| **스레드 안전** | `threading.Lock`으로 상태 읽기/쓰기 보호 |
| **GUI 제거** | 서버 모드에서는 `cv2.imshow`, `cv2.waitKey` 사용하지 않음 |
| **GPU 차단 유지** | 기존 `CUDA_VISIBLE_DEVICES = '-1'` 설정 그대로 유지 |
| **RTSP 재연결** | 스트림 끊김 시 자동 재연결 로직 추가 |
| **모델 로드 시간** | 서버 시작 시 1회만 로드, 이후 재사용 |

---

## 클라이언트 사용 예시

```bash
# 상태 조회
curl http://localhost:8000/status

# 응답
{
  "in_bed": "YES",
  "pose": "p03",
  "timestamp": "2026-04-01T14:30:00.123456"
}
```

```python
# Python 클라이언트
import requests
res = requests.get("http://localhost:8000/status")
print(res.json())
```

---

## 작업 체크리스트

- [ ] `fastapi`, `uvicorn` 설치
- [ ] `server.py` 파일 생성
- [ ] 공유 상태 + Lock 구현
- [ ] 분석 루프를 스레드 함수로 변환
- [ ] `/status`, `/health` 엔드포인트 구현
- [ ] lifespan으로 스레드 시작/종료 관리
- [ ] headless 모드 테스트
- [ ] RTSP 재연결 로직 추가
