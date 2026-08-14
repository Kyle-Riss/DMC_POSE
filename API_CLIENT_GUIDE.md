# 포즈 모니터링 API 클라이언트 가이드

## 서버 실행

```bash
cd /home/dmc/pose
python server.py
```

서버가 `http://0.0.0.0:8000` 에서 시작되며, 백그라운드에서 RTSP 스트림 분석이 자동으로 시작됩니다.

---

## API 엔드포인트

### 1. `GET /status` — 포즈 상태 조회

최신 분석 결과를 반환합니다.

**요청**

```bash
curl http://localhost:8000/status
```

**응답**

```json
{
  "in_bed": "YES",
  "pose": "p03",
  "timestamp": "2026-04-01T14:30:00.123456"
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `in_bed` | `string` | 침대 위 여부 (`"YES"` / `"NO"`) |
| `pose` | `string` | 감지된 포즈 (`"p01"` ~ `"p12"` 또는 `"None"`) |
| `timestamp` | `string` | 마지막 분석 시각 (ISO 8601). 아직 분석 전이면 `null` |

---

### 2. `GET /health` — 서버 상태 확인

서버와 분석 스레드의 동작 상태를 확인합니다.

**요청**

```bash
curl http://localhost:8000/health
```

**응답**

```json
{
  "server": "ok",
  "analysis_running": true
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `server` | `string` | 서버 상태 (`"ok"`) |
| `analysis_running` | `boolean` | 분석 스레드 동작 여부 |

---

## 클라이언트 예제

### cURL

```bash
# 상태 조회
curl http://localhost:8000/status

# 헬스체크
curl http://localhost:8000/health
```

### Python (requests)

```python
import requests

# 상태 조회
res = requests.get("http://localhost:8000/status")
data = res.json()

print(f"침대 위: {data['in_bed']}")
print(f"포즈:    {data['pose']}")
print(f"시각:    {data['timestamp']}")
```

### Python (주기적 폴링)

```python
import requests
import time

while True:
    res = requests.get("http://localhost:8000/status")
    data = res.json()
    print(f"[{data['timestamp']}] In Bed: {data['in_bed']} | Pose: {data['pose']}")
    time.sleep(1)
```

### JavaScript (fetch)

```javascript
async function getStatus() {
  const res = await fetch("http://localhost:8000/status");
  const data = await res.json();
  console.log(`In Bed: ${data.in_bed} | Pose: ${data.pose}`);
}

getStatus();
```

---

## Swagger 자동 문서

서버 실행 후 브라우저에서 아래 주소로 접속하면 인터랙티브 API 문서를 확인할 수 있습니다.

| 문서 | URL |
|------|-----|
| Swagger UI | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |

---

## 원격 접속

같은 네트워크의 다른 기기에서 접속할 경우, `localhost` 대신 서버의 IP 주소를 사용합니다.

```bash
curl http://192.168.0.XXX:8000/status
```
