# pose-sixclass 이식 구현 설계안

> **문서 목적**  
> [`FALL_RISK_SYSTEM_DESIGN.md`](./FALL_RISK_SYSTEM_DESIGN.md)의 목표 아키텍처를 **`pose-sixclass` 코드베이스에 단계적으로 이식**하기 위한 실행 설계이다.  
> 도메인 정의·feature 수식·라벨 taxonomy는 상위 설계서를 따르고, 본 문서는 **파일 구조, 모듈 경계, Phase별 산출물·검증**만 다룬다.

| 항목 | 값 |
|------|-----|
| **버전** | 0.1 |
| **기준 코드** | `pose-sixclass/server.py`, `run_pose.py` |
| **참고 이식원** | `pose/run_pose.py` (`get_bed_bbox`, `draw_bed_zones`, `calc_fall_risk`) |
| **동기화 대상** | `pose-sixclass-viewer/` (Phase마다 동일 모듈 import 또는 주기적 복사) |

---

## 1. 범위와 원칙

### 1.1 In scope (이 설계안)

| 구분 | 내용 |
|------|------|
| **런타임** | RTSP 실시간 (`server.py`) + 로컬 디버그 (`run_pose.py`) |
| **설정** | `rooms/*.json`, `presets/*.json` |
| **기하** | 이미지 좌표 → (Phase 2) bed_bbox 기반 → (Phase 3) `bed_norm` + H |
| **위험도** | overflow → out_bed_ratio → temporal buffer → `risk_score` |
| **API** | `/status` 확장, `/calibration/*` |

### 1.2 Out of scope (별도 Phase / 문서)

- MP4 배치 `extract_features.py` (데이터셋 Phase 1 — 상위 설계 S9)
- TCN/GRU 학습 (Phase 5)
- 12-class `pose-webviewer` 통합
- intrinsic(L1) 체스판 보정 (필요 시 preset 확장만)

### 1.3 구현 원칙

1. **단일 파이프라인** — `server.py` / `run_pose.py`는 얇은 진입점, 로직은 `bed_monitor/` 패키지에만 둔다.
2. **하위 호환** — Phase마다 `/status` 기존 필드 유지, 신규 필드는 optional 추가.
3. **label ≠ risk_score** — API·CSV 어디에도 자동 label 생성 금지 (상위 설계 S6.1).
4. **preset 없으면 동작** — `default` preset으로 현재 동작(800px, center in_bed) 재현.
5. **FallDetector** — Phase 3까지 병행 표시 가능, Phase 4에서 `risk_score`로 대체 후 제거.

---

## 2. 현재 상태 (As-Is)

```text
pose-sixclass/
├── server.py          # FastAPI + analysis_loop (in_bed, pose, MJPEG)
├── run_pose.py        # OpenCV GUI + FallDetector (server와 로직 중복)
├── my_model_six_check.keras
├── yolo11n-seg.pt, yolo11m-pose.pt
└── docs/FALL_RISK_SYSTEM_DESIGN.md
```

| 기능 | server.py | run_pose.py | 설계 목표 |
|------|-----------|-------------|-----------|
| YOLO seg + pose + 6-class | ✅ | ✅ | ✅ |
| in_bed (bbox 중심 1점) | ✅ | ✅ | out_bed_ratio |
| bed_bbox / zones | ❌ | ❌ | ✅ (pose 이식) |
| limb overflow | ❌ | ❌ | ✅ |
| H / bed_norm | ❌ | ❌ | ✅ |
| preset / room JSON | ❌ | 하드코딩 | ✅ |
| risk_score (0~10) | ❌ | FallDetector 0~1 | ✅ |
| calibration API | ❌ | ❌ | ✅ |

**중복 제거 대상:** `is_person_in_bed`, 모델 로딩, 프레임 루프가 `server.py`와 `run_pose.py`에 이중 존재.

---

## 3. 목표 구조 (To-Be)

### 3.1 디렉터리

```text
pose-sixclass/
├── server.py                 # FastAPI (얇음)
├── run_pose.py               # CLI 디버그 (얇음)
├── calibrate_bed.py          # Phase 3: OpenCV 4점 보정 CLI (신규)
├── bed_monitor/
│   ├── __init__.py
│   ├── config.py             # RoomConfig, Preset, 로더
│   ├── geometry.py           # get_bed_bbox, H, img↔bed_norm
│   ├── features.py           # out_bed_ratio, edge_zone, motion EMA
│   ├── risk_rules.py         # overflow, risk_score (Phase 2~4)
│   ├── temporal.py           # ring buffer (Phase 4)
│   ├── pipeline.py           # FrameResult, process_frame()
│   └── overlay.py            # draw_bed_zones, risk overlay
├── config/
│   ├── rooms/
│   │   └── room_example.json
│   └── presets/
│       ├── default.json
│       └── preset_03.json
├── tests/
│   ├── test_geometry.py
│   ├── test_features.py
│   └── test_risk_rules.py
└── docs/
    ├── FALL_RISK_SYSTEM_DESIGN.md
    └── IMPLEMENTATION_PLAN.md   # 본 문서
```

### 3.2 핵심 타입 (모듈 간 계약)

```python
# bed_monitor/pipeline.py (개념 스케치)

@dataclass
class FrameResult:
    timestamp: datetime
    frame_idx: int
    person_detected: bool
    in_bed: str                    # "YES"|"NO" — Phase 2+: out_ratio 기반 파생
    out_bed_ratio: float | None
    edge_zone: str | None          # "L"|"C"|"R"
    pose: str
    pose_display: str
    pose_conf: float
    limb_overflow_max: float
    risk_level: str                # SAFE|LOW|MED|HIGH → Phase 4: risk_score
    risk_score: float | None
    preset_id: str
    h_version: int
    # Phase 3+
    center_speed: float | None
    bed_moved_flag: bool
```

`process_frame(frame, models, state) -> (FrameResult, overlay_bgr)` 가 **유일한 분석 진입점**.

### 3.3 실행 모드

| 모드 | 진입점 | 용도 |
|------|--------|------|
| **API** | `uvicorn server:app` | 병실 모니터링, 웹 `/viewer` |
| **로컬** | `python run_pose.py` | 현장 디버그, overlay 확인 |
| **캘리브** | `python calibrate_bed.py --room room_101` | 4점 H 저장 |
| **배치** | (후속) `extract_features.py` | MP4 → CSV |

환경 변수 (공통):

| 변수 | 기본 | 설명 |
|------|------|------|
| `POSE_ROOM_ID` | `room_example` | `config/rooms/{id}.json` |
| `POSE_PRESET_ID` | (room에서 로드) | preset 오버라이드 |
| `POSE_API_HOST` | `0.0.0.0` | API 바인딩 |
| `POSE_API_PORT` | `8000` | API 포트 |

---

## 4. 설정 스키마 (최소 → 확장)

### 4.1 Phase 0~1: `config/presets/default.json`

```json
{
  "preset_id": "default",
  "version": "1.0.0",
  "inference": {
    "resize_width": 800,
    "seg_conf": 0.2,
    "pose_conf": 0.5,
    "seg_hz": 0
  },
  "risk_thresholds": {
    "overflow_low": 0.05,
    "overflow_med": 0.15,
    "overflow_high": 0.25
  },
  "zones": { "split_lcr": true }
}
```

`seg_hz: 0` = 매 프레임 seg (현재와 동일).

### 4.2 Phase 2+: preset에 `homography` 블록 추가

상위 설계 S2.3과 동일. Phase 2에서는 **비어 있어도 됨** (image-space만 사용).

### 4.3 `config/rooms/room_example.json`

```json
{
  "room_id": "room_example",
  "rtsp_url": "rtsp://192.168.0.157:8554/stream",
  "preset_id": "default",
  "guardrail": false,
  "bed_movable": true
}
```

---

## 5. Phase별 이식 계획

```text
Phase 0 ──► 모듈 추출 + default preset (동작 변화 없음)
Phase 1 ──► preset/room 로더 + inference 파라미터화
Phase 2 ──► image-space bed geometry + overflow (pose 이식)
Phase 3 ──► H + bed_norm + out_bed_ratio + 캘리브
Phase 4 ──► temporal buffer + risk_score (FallDetector 퇴역)
Phase 5 ──► MP4/CSV 수집 + (선택) viewer 동기화
```

각 Phase는 **독립 배포 가능**하도록 완료 기준(Definition of Done)을 둔다.

---

### Phase 0 — 공통 파이프라인 추출

**목표:** 동작 변화 없이 중복 제거, 이후 Phase의 받침대 마련.

| 작업 | 파일 | 내용 |
|------|------|------|
| 패키지 생성 | `bed_monitor/` | 빈 모듈 + `pipeline.py` 골격 |
| 이전 | `geometry.py` | `is_person_in_bed` (기존 로직 그대로) |
| 이전 | `pipeline.py` | seg → pose → keras → `FrameResult` |
| 이전 | `overlay.py` | server용 텍스트 박스, MJPEG용 encode 전 처리 |
| 리팩터 | `server.py` | `analysis_loop` → `pipeline.process_frame` 호출만 |
| 리팩터 | `run_pose.py` | 동일 `process_frame` + `cv2.imshow` |
| 상수 통합 | `config.py` | `CLASS_NAMES`, `CLASS_DISPLAY_NAMES` |

**완료 기준**

- [ ] `/status` 응답이 리팩터 전과 동일 (in_bed, pose, pose_conf)
- [ ] `/video` MJPEG 정상
- [ ] `python run_pose.py` RTSP 동일 동작
- [ ] `tests/test_pipeline_smoke.py` — 더미 프레임 1장 NaN 없이 통과

**예상 diff:** ~400줄 이동, 신규 로직 거의 없음.

---

### Phase 1 — Preset / Room 로더

**목표:** 하드코딩 제거, 설치 종류별 inference·임계 분리 (H 없이).

| 작업 | 파일 | 내용 |
|------|------|------|
| 로더 | `bed_monitor/config.py` | `load_room()`, `load_preset()`, 검증 |
| 설정 | `config/presets/default.json`, `config/rooms/*.json` | §4 스키마 |
| 연동 | `pipeline.py` | `resize_width`, `seg_conf`, `pose_conf` preset 적용 |
| API | `server.py` | `/status`에 `preset_id`, `room_id` 추가 |
| 문서 | `API_CLIENT_GUIDE.md` | 신규 필드 반영 |

**완료 기준**

- [ ] `POSE_ROOM_ID` 변경만으로 RTSP URL 전환
- [ ] preset의 `seg_conf` 변경 시 seg 민감도 변화 확인
- [ ] preset 파일 없으면 `default` 로드 + warning 로그

**의존성:** Phase 0.

---

### Phase 2 — Image-space 기하 (pose 이식)

> **실행 문서:** [`PHASE_2.md`](./PHASE_2.md) — 모듈 명세, API 스키마, 테스트 케이스, 구현 체크리스트

**목표:** H 없이도 설계 S4.5·zone 개념을 픽셀 bed_bbox로 구현.

| 작업 | 파일 | 내용 |
|------|------|------|
| 이식 | `geometry.py` | `get_bed_bbox`, `draw_bed_zones` (`pose/run_pose.py`) |
| 이식 | `risk_rules.py` | `calc_limb_overflow` ← `calc_fall_risk` |
| feature | `features.py` | `edge_zone_from_bbox(cx)` — L/C/R |
| 파생 | `pipeline.py` | `limb_overflow_max`, `risk_level` (SAFE/LOW/MED/HIGH) |
| overlay | `overlay.py` | 3등분 zone + 위험 키포인트 강조 |
| API | `/status` | `risk_level`, `limb_overflow_max`, `edge_zone` |
| 제거 | `run_pose.py` | Rail_0/1 placeholder, 고정 `Rail_0: 1` 텍스트 |

**in_bed 정책 (과도기)**

```text
out_bed_ratio 없을 때: 기존 center in_bed 유지 (호환)
overflow HIGH 이고 in_bed YES: 로그 warning (불일치 검출)
```

**완료 기준**

- [ ] overlay에 L/C/R zone 표시
- [ ] 손목이 침대 bbox 밖으로 나가면 `risk_level` ≥ LOW
- [ ] `tests/test_risk_rules.py` — synthetic kpts + bed_bbox 케이스 3개

**의존성:** Phase 1 (`risk_thresholds` preset에서 읽기).

---

### Phase 3 — Homography + bed_norm + 캘리브

**목표:** 설치 불변 좌표계 확립, `out_bed_ratio`·motion feature의 기반.

| 작업 | 파일 | 내용 |
|------|------|------|
| H 유틸 | `geometry.py` | `compute_H`, `img_to_bed`, `bed_to_img`, `H_stable` 상태 |
| feature | `features.py` | `out_bed_ratio(kpts, mask, H)`, `edge_zone_bed_norm` |
| motion | `features.py` | hip center bed_norm + EMA → `center_speed`, `accel_like` |
| 갱신 | `geometry.py` | S5.3 seg EMA (`update_policy` preset) |
| CLI | `calibrate_bed.py` | 4점 클릭 → preset/room별 `homography` JSON 저장 |
| API | `POST /calibration/corners` | 4점 수신 → H 재계산 → `h_version++` |
| API | `GET /calibration` | `h_version`, `bed_moved_flag` |
| API | `POST /calibration/preset` | preset hot-swap (분석 스레드 lock) |

**in_bed 전환**

```text
out_bed_ratio < 0.1  → in_bed = "YES"
out_bed_ratio >= 0.3 → in_bed = "NO"
그 사이              → 이전 값 유지 (히스테리시스, 깜빡임 방지)
```

**완료 기준**

- [ ] `calibrate_bed.py` 실행 후 저장된 H로 zone이 침대에 정렬됨
- [ ] `/status`에 `out_bed_ratio`, `h_version` 포함
- [ ] 침대 이동 시뮬레이션(영상 또는 수동 corner shift) → `bed_moved_flag=true`
- [ ] H 없는 preset → Phase 2 image-space로 fallback (명시 로그)

**의존성:** Phase 2.

---

### Phase 4 — Temporal + risk_score (설계 S6)

**목표:** `FallDetector` 대신 설계서 rule 파이프라인 완성.

| 작업 | 파일 | 내용 |
|------|------|------|
| 버퍼 | `temporal.py` | 5s deque: `edge_stay_duration`, `out_bed_ratio_max_3s`, `speed_max_1s`, … |
| 점수 | `risk_rules.py` | `compute_risk_score()` — base_pose + S_out + S_speed + … |
| 통합 | `pipeline.py` | 매 프레임 F→G |
| API | `/status` | `risk_score` (0~10), `risk_level` (LOW/MED/HIGH 매핑) |
| 정리 | `run_pose.py` | `FallDetector` 제거 또는 `--legacy-fall` 플래그만 |

**FallDetector 마이그레이션**

| FallDetector | 대체 |
|--------------|------|
| rapid_drop (norm_v) | `vertical_drop_rate` + `speed_max_1s` |
| horizontal_body | `lying_flag` + torso angle 보조 |
| CONFIRMED state machine | `risk_score` threshold + cooldown preset |

**완료 기준**

- [ ] 동일 RTSP에서 `risk_score`가 overlay/API에 표시
- [ ] preset `risk_thresholds` 변경 시 점수 민감도 변화
- [ ] `guardrail: false` room에서 rail 항 0 처리
- [ ] 단위 테스트: S6 표의 base_pose + S_out 조합 5케이스

**의존성:** Phase 3 (bed_norm speed).

---

### Phase 5 — 데이터셋 파이프라인 + viewer 동기화

**목표:** 오프라인 재현·라벨링 루프 (상위 설계 S8·S9 Phase 1).

> **현재 Raw_data 현실 경로:** [`PHASE_TIMESERIES.md`](./PHASE_TIMESERIES.md) — `extract_raw_timeseries` → `enrich_timeseries` → window → segment 검증.  
> 장기적으로 `extract_features.py` / `episode_*` 레이아웃으로 수렴.

| 작업 | 파일 | 내용 |
|------|------|------|
| 배치 | `extract_features.py` | MP4 → `episode_*/features.csv` |
| 메타 | `episode_meta.json` | preset_id, h_version, pipeline_version |
| 동기화 | `pose-sixclass-viewer/` | `bed_monitor/` 복사 또는 git submodule |
| CI | `validate_e2e.py` 확장 | preset 로드 + 1 MP4 golden CSV diff |

**완료 기준**

- [ ] 동일 MP4 + preset → CSV 해시 재현
- [ ] JSON에 `label` 비어 있음, `risk_score` 채워짐

**의존성:** Phase 4.

---

## 6. API 진화 로드맵

### 6.1 `/status` 필드 추가 순서

| Phase | 추가 필드 |
|-------|-----------|
| 0 | (변경 없음) |
| 1 | `preset_id`, `room_id` |
| 2 | `risk_level`, `limb_overflow_max`, `edge_zone` |
| 3 | `out_bed_ratio`, `h_version`, `bed_moved_flag` |
| 4 | `risk_score`, `center_speed`, `pose_display` (이미 내부 사용 중이면 공개) |

Pydantic 모델은 **필드 optional + default** 로 점진 확장.

### 6.2 Calibration API (Phase 3)

```http
GET  /calibration
POST /calibration/preset   {"preset_id": "preset_03"}
POST /calibration/corners  {"points": [[x,y], ...]}  # 4 points, image coords @ resize_width
```

- corners는 **resize 후 좌표계** 기준 (pipeline과 동일).
- 저장 경로: `config/presets/{preset_id}.json` 또는 room override 파일 (운영 정책 선택).

### 6.3 웹 캘리브 (선택, Phase 3b)

`/viewer`에 4점 클릭 UI 추가 → `POST /calibration/corners`.  
우선순위는 **CLI `calibrate_bed.py`** (현장 노트북 + DISPLAY).

---

## 7. `server.py` vs `run_pose.py` 역할 (고정)

```text
                    ┌─────────────────────┐
                    │  bed_monitor/       │
                    │  pipeline.process   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
       ┌─────────────┐                   ┌─────────────┐
       │  server.py  │                   │ run_pose.py │
       │  Thread     │                   │  main()     │
       │  shared_state│                  │  cv2.imshow │
       │  FastAPI    │                   │  (DISPLAY)  │
       └─────────────┘                   └─────────────┘
```

- **비즈니스 로직 금지** — 두 파일 모두 150줄 이하 유지 목표.
- MJPEG·재연결·Lock은 `server.py`만 담당.

---

## 8. `pose-sixclass-viewer` 관계

| 시점 | 정책 |
|------|------|
| Phase 0~2 | `pose-sixclass`만 수정, viewer는 **미동기화** (문서에 명시) |
| Phase 3+ | viewer의 `run_pose.py`를 **import bed_monitor** 로 교체하거나 저장소 통합 검토 |
| 장기 | 단일 repo (`pose-sixclass`) + viewer는 thin wrapper 권장 |

viewer를 별도 유지할 경우: `bed_monitor/`만 주기적으로 복사하고 **서버 기능은 pose-sixclass만** 유지.

---

## 9. 테스트·검증 전략

| 레벨 | 내용 | Phase |
|------|------|-------|
| **Unit** | geometry, out_bed_ratio, risk_rules | 2~4 |
| **Smoke** | 1프레임 pipeline | 0 |
| **API** | curl `/status`, `/health` | 0~4 |
| **Visual** | `run_pose.py` zone/overflow overlay | 2~3 |
| **Golden** | 짧은 MP4 → CSV column subset | 5 |
| **현장** | 4점 캘리브 후 bed edge ∥ zone | 3 |

`pipeline_version` 문자열 (예: `pose-sixclass/0.3.0`)을 `FrameResult`·episode JSON에 기록.

---

## 10. 리스크·완화 (구현 관점)

| ID | 리스크 | 완화 (본 계획) |
|----|--------|----------------|
| I1 | 리팩터 중 RTSP 회귀 | Phase 0에서 동작 동일성 먼저 고정 |
| I2 | in_bed 정의 변경으로 클라이언트 깨짐 | out_bed_ratio 병행 노출, in_bed는 파생 |
| I3 | H 저장 좌표계 혼동 | resize 후 좌표만 사용, 문서·API 주석 |
| I4 | analysis_loop lock 경합 | calibration 시 `state_lock`으로 H만 스왑 |
| I5 | FallDetector 제거 조급 | Phase 4까지 `--legacy-fall` |

---

## 11. 작업 순서 요약 (체크리스트)

```text
[ ] P0  bed_monitor/ 추출, server·run_pose 얇게
[ ] P1  config/presets, config/rooms, POSE_ROOM_ID
[ ] P2  get_bed_bbox, calc_limb_overflow, edge_zone, /status 확장
[ ] P3  H, out_bed_ratio, calibrate_bed.py, /calibration/*
[ ] P4  temporal.py, risk_score, FallDetector 제거
[ ] P5  extract_features.py, viewer 동기화
```

**권장 첫 PR:** Phase 0만 (동작 동일 + 구조).  
**권장 둘째 PR:** Phase 1 + Phase 2 (현장에서 바로 보이는 zone/overflow).

---

## 12. 상위 설계서와의 매핑

| 본 문서 Phase | FALL_RISK_SYSTEM_DESIGN |
|---------------|-------------------------|
| P0~P1 | S2.3 preset (inference만), S3 리팩터 |
| P2 | S4.5 overflow, S4.2 edge_zone (image), S13 `pose/run_pose.py` |
| P3 | S5.1~S5.3, S4.1~S4.3 bed_norm |
| P4 | S4.6, S6.3 risk_score |
| P5 | S8, S9 Phase 1, S10.2 API |

---

## Changelog

| 버전 | 날짜 | 변경 |
|------|------|------|
| 0.1 | 2026-05-18 | 초안 — pose-sixclass 이식 Phase 0~5 |

*구현 착수 시 `config/presets/default.json`의 `version`과 코드 `PIPELINE_VERSION`을 함께 올릴 것.*
