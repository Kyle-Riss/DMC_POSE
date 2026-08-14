# Phase 2 — Image-space 기하 + limb overflow

> **문서 목적**  
> [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) Phase 2를 **실행 가능한 수준**으로 풀어 쓴 문서이다.  
> 상위 feature 정의·수식은 [`FALL_RISK_SYSTEM_DESIGN.md`](./FALL_RISK_SYSTEM_DESIGN.md) S4.2·S4.5를 따른다.

| 항목 | 값 |
|------|-----|
| **버전** | 0.1 |
| **기준 코드** | `pose-sixclass/server.py`, `run_pose.py` |
| **이식 원본** | `pose/run_pose.py` — `calc_fall_risk`, `draw_risk_points`, `RISK_KEYPOINTS` |
| **선행 Phase** | Phase 0·1 권장 (본 문서 §2 “Fast path”로 Phase 2 단독 착수 가능) |
| **후속 Phase** | Phase 3 (H, out_bed_ratio), Phase 4 (risk_score, temporal) |

---

## 1. 목표

Homography(H) **없이** 픽셀 `bed_bbox`만으로 다음을 런타임에 추가한다.

| 산출물 | 설명 |
|--------|------|
| `edge_zone` | hip 중심 x → L / C / R (bed bbox 3등분) |
| `limb_overflow_max` | 손목·발목 lateral 이탈 비율 (침대 너비 대비) |
| `risk_level` | SAFE / LOW / MED / HIGH |
| overlay | L/C/R zone + 위험 키포인트 강조 |
| `/status` | 위 3필드 optional 추가 (하위 호환) |

**Phase 2에서 하지 않는 것**

- `out_bed_ratio` (키포인트 vs mask 비율) → Phase 3
- `risk_score` 0~10, temporal buffer → Phase 4
- 난간(rail) 올림/내림 판별 → 별도 (rail_geometry + Phase 4+)
- Homography, calibration API → Phase 3

---

## 2. 선행 조건

### 2.1 이상 경로 (설계서 순서)

```text
Phase 0 (bed_monitor 추출) → Phase 1 (preset JSON) → Phase 2 (본 문서)
```

### 2.2 Fast path (현재 repo 현실)

Phase 0·1이 **미착수**이므로, Phase 2는 아래 **최소 번들**과 함께 착수한다.

| 최소 작업 | 이유 |
|-----------|------|
| `bed_monitor/` 패키지 생성 | overflow·zone 로직을 server/run_pose에 또 복붙하지 않기 위함 |
| `config/presets/default.json` 1개 | `risk_thresholds`만이라도 파일로 분리 (Phase 1 축소판) |
| `bed_monitor/config.py` — `load_preset()` | JSON 없으면 코드 내 default dict 폴백 |

Phase 0 전체( pipeline 리팩터 )는 **Phase 2 PR과 같은 브랜치에서 “얇게”** 진행해도 된다.  
`analysis_loop` 전체를 `process_frame`으로 옮기지 못하면, **overflow/zone만** `bed_monitor`로 빼고 server는 import만 해도 Phase 2 DoD 충족 가능.

---

## 3. As-Is (현재 pose-sixclass)

### 3.1 이미 있는 것 ✅

| 기능 | 위치 | 비고 |
|------|------|------|
| `bed_bbox` from seg mask/box | `server.extract_bed_detection` | mask 우선, box 폴백 |
| `bed_roi` clip | `bed_roi/roi_utils.apply_bed_roi` | seg 과대 탐지 보정 |
| `is_person_in_bed` | `server.is_person_in_bed` | mask 중심 → bbox |
| `draw_bed_zones` L/C/R | `server.draw_bed_zones` | overlay에 표시 중 |
| 병원 침대 seg | `yolo11n-bed-seg.pt` | class 0 |
| 6-class pose | `my_model_six_check.keras` | E2E 98.6% |

### 3.2 없는 것 ❌

| 기능 | pose 레거시 | sixclass |
|------|-------------|----------|
| `calc_fall_risk` / limb overflow | ✅ `pose/run_pose.py` | ❌ |
| `draw_risk_points` | ✅ | ❌ |
| `/status` risk 필드 | — | ❌ |
| `edge_zone` API 필드 | — | ❌ (overlay만) |
| Rail 표시 | pose: Risk 텍스트 | sixclass run_pose: **가짜 Rail 하드코딩** |

### 3.3 이식 원본 (pose/run_pose.py)

```python
RISK_KEYPOINTS = {
    'L_wrist':  9,
    'R_wrist': 10,
    'L_ankle': 15,
    'R_ankle': 16,
}

RISK_THRESHOLDS = {
    'LOW':  0.05,
    'MED':  0.15,
    'HIGH': 0.25,
}
```

`calc_fall_risk(kpts_xy, kpts_conf, bed_bbox, conf_threshold=0.3)`  
→ `(level, max_overflow, danger_pts)`

---

## 4. To-Be 구조

```text
pose-sixclass/
├── config/
│   └── presets/
│       └── default.json          # risk_thresholds, inference (Phase 1 축소)
├── bed_monitor/
│   ├── __init__.py
│   ├── config.py                 # load_preset(), defaults
│   ├── geometry.py               # edge_zone_from_bbox, (기존 is_person_in_bed 이전)
│   ├── risk_rules.py             # calc_limb_overflow, overflow_to_risk_level
│   └── overlay.py                # draw_risk_points, risk HUD 텍스트
├── server.py                     # analysis_loop에서 bed_monitor 호출
└── run_pose.py                   # FallDetector 유지, Rail 가짜 제거, risk 표시
```

Phase 0의 `pipeline.py`는 Phase 2에서 **골격만** 두고, `process_frame()` 내부에 overflow 호출 1줄 추가해도 충분하다.

---

## 5. 설정 스키마 (Phase 2 최소)

`config/presets/default.json`:

```json
{
  "preset_id": "default",
  "version": "1.0.0",
  "inference": {
    "resize_width": 640,
    "seg_conf": 0.1,
    "pose_conf": 0.5,
    "seg_every_n": 3,
    "kpt_conf_min": 0.3
  },
  "risk_thresholds": {
    "overflow_low": 0.05,
    "overflow_med": 0.15,
    "overflow_high": 0.25
  },
  "zones": {
    "split_lcr": true
  }
}
```

환경 변수 오버라이드 (Phase 1 전까지):

| 변수 | preset 키 |
|------|-----------|
| `POSE_FRAME_WIDTH` | `inference.resize_width` |
| `POSE_BED_SEG_CONF` | `inference.seg_conf` |
| `POSE_SEG_EVERY` | `inference.seg_every_n` |

---

## 6. 모듈 명세

### 6.1 `bed_monitor/geometry.py`

```python
def edge_zone_from_bbox(cx: float, bed_bbox: tuple[int, int, int, int] | None) -> str | None:
    """
    bed_bbox = (x_min, y_min, x_max, y_max) in image pixels.
    x < x_min + w/3 → "L"
    x > x_min + 2w/3 → "R"
    else → "C"
    bed_bbox is None → None
    """
```

**hip 중심 `cx` 우선순위**

1. `(L_hip + R_hip) / 2` (conf ≥ kpt_conf_min)
2. person bbox center `(x1+x2)/2`
3. None → `edge_zone = None`

> Phase 3 이후 동일 함수를 bed_norm x로 교체. API 필드명 `edge_zone` 유지.

### 6.2 `bed_monitor/risk_rules.py`

```python
RISK_KEYPOINTS: dict[str, int]  # YOLO COCO 17 — wrist 9,10 ankle 15,16

def calc_limb_overflow(
    kpts_xy: np.ndarray,       # shape (17, 2) or (1, 17, 2)
    kpts_conf: np.ndarray,     # shape (17,)
    bed_bbox: tuple[int, int, int, int] | None,
    *,
    conf_threshold: float = 0.3,
) -> tuple[float, list[tuple[str, int, int, float]]]:
    """
    Returns (limb_overflow_max, danger_pts).
    danger_pts: [(name, x, y, overflow), ...]
    Lateral only: overflow = distance outside [x_min, x_max] / bed_width.
    """

def overflow_to_risk_level(
    limb_overflow_max: float,
    thresholds: dict[str, float],
) -> str:
    """
    thresholds keys: overflow_low, overflow_med, overflow_high
    → SAFE | LOW | MED | HIGH
    """
```

**pose `calc_fall_risk`와의 차이**

| 항목 | pose | Phase 2 |
|------|------|---------|
| 함수명 | `calc_fall_risk` | `calc_limb_overflow` + `overflow_to_risk_level` |
| 임계값 | 모듈 상수 | preset `risk_thresholds` |
| 반환 | level 포함 3-tuple | overflow와 level 분리 (Phase 4 risk_score 재사용) |

### 6.3 `bed_monitor/overlay.py`

```python
RISK_COLORS = {
    "SAFE": (0, 255, 0),
    "LOW":  (0, 255, 255),
    "MED":  (0, 165, 255),
    "HIGH": (0, 0, 255),
}

def draw_risk_points(frame, danger_pts, risk_level: str) -> np.ndarray:
    """pose/run_pose.py draw_risk_points 이식"""

def draw_risk_hud(
    frame,
    *,
    risk_level: str,
    limb_overflow_max: float,
    edge_zone: str | None,
    y_offset: int = 170,
) -> np.ndarray:
    """server MJPEG HUD에 Risk / zone 한 줄 추가"""
```

---

## 7. 파이프라인 연동

### 7.1 프레임 처리 순서 (Phase 2)

```text
[기존] seg → ROI → pose → in_bed → keras 6-class
[추가] bed_bbox 확정 후:
       edge_zone ← hip cx + bed_bbox
       limb_overflow_max, danger_pts ← calc_limb_overflow
       risk_level ← overflow_to_risk_level
       overlay ← draw_risk_points + draw_risk_hud
```

### 7.2 `in_bed` 과도기 정책

Phase 3 `out_bed_ratio` 전까지 **기존 in_bed 로직 유지**.

```text
in_bed: 기존 mask/bbox center (변경 없음)

단, 불일치 감지 (로그 only):
  IF in_bed == "YES" AND risk_level == "HIGH":
    logging.warning("in_bed/overflow mismatch ...")
```

`in_bed`를 overflow로 **대체하지 않는다** (Phase 3에서 out_bed_ratio + 히스테리시스로 전환).

### 7.3 `run_pose.py` 정리

| 항목 | 조치 |
|------|------|
| `Rail_0: 1`, `Rail_1: 0` 하드코딩 | **삭제** |
| `RAIL_KEYPOINTS` 오버레이 | **삭제** (난간 미구현) |
| `FallDetector` | Phase 4까지 **유지** (`--legacy-fall` 불필요) |
| Risk 표시 | `Risk: {level} ({overflow*100:.1f}%)` — pose와 동일 |
| server import | `calc_limb_overflow`, `draw_risk_points` from `bed_monitor` |

---

## 8. API 변경

### 8.1 `GET /status` — 필드 추가 (optional, 하위 호환)

```json
{
  "in_bed": "YES",
  "pose": "앉음_가장자리",
  "pose_conf": 0.92,
  "timestamp": "2026-06-18T14:00:00",
  "ip_address": "192.168.0.161",
  "latency_ms": 45.2,
  "frame_age_ms": 12.0,
  "pipeline_fps": 8.5,
  "edge_zone": "R",
  "limb_overflow_max": 0.18,
  "risk_level": "MED",
  "preset_id": "default"
}
```

| 필드 | 타입 | person 없을 때 |
|------|------|----------------|
| `edge_zone` | `"L"\|"C"\|"R"\|null` | `null` |
| `limb_overflow_max` | `float` 0~1+ | `0.0` |
| `risk_level` | `"SAFE"\|"LOW"\|"MED"\|"HIGH"` | `"SAFE"` |
| `preset_id` | `string` | `"default"` |

Pydantic: 신규 필드에 **default** 부여 → 기존 클라이언트 파싱 깨지지 않음.

### 8.2 `API_CLIENT_GUIDE.md` 갱신 항목

- 경로 `pose-sixclass`, 6-class pose 이름
- `/status` 신규 필드 표
- `risk_level` ≠ 낙상 확정 (보조 신호)

---

## 9. 테스트

### 9.1 `tests/test_risk_rules.py` (필수 3케이스)

**공통 bed_bbox:** `(100, 50, 400, 300)` → bed_width = 300

| # | 케이스 | kpts 설정 | 기대 overflow | 기대 level |
|---|--------|-----------|---------------|------------|
| 1 | SAFE — 손목 침대 안 | L_wrist (250, 200) conf 0.9 | 0.0 | SAFE |
| 2 | LOW — 우손목 약간 밖 | R_wrist (410, 200) conf 0.9 | (410-400)/300 ≈ 0.033 → below 0.05 | SAFE |
| 3 | MED — 우손목 15% 밖 | R_wrist (445, 200) conf 0.9 | 0.15 | MED |
| 4 | HIGH — 좌발목 크게 밖 | L_ankle (10, 280) conf 0.9 | (100-10)/300 ≈ 0.30 | HIGH |
| 5 | conf 낮음 무시 | R_wrist (500, 200) conf 0.1 | 0.0 (ignored) | SAFE |

> DoD는 3케이스지만, conf 필터 케이스 1개 포함 권장.

### 9.2 `tests/test_geometry.py`

| # | cx | bed_bbox | edge_zone |
|---|-----|----------|-----------|
| 1 | 150 | (100,0,400,100) | L |
| 2 | 250 | (100,0,400,100) | C |
| 3 | 350 | (100,0,400,100) | R |

### 9.3 수동 RTSP 검증

1. `bash run_server.sh` → `/viewer`
2. 환자 손을 침대 밖으로 → `risk_level` LOW 이상, 빨간/주황 키포인트
3. `curl localhost:8000/status` → JSON 필드 확인
4. `python run_pose.py` → Rail 텍스트 **없음**, Risk 줄 **있음**

---

## 10. 구현 체크리스트 (순서)

```text
[ ] 1. config/presets/default.json 생성
[ ] 2. bed_monitor/config.py — load_preset + defaults
[ ] 3. bed_monitor/risk_rules.py — calc_limb_overflow, overflow_to_risk_level
[ ] 4. bed_monitor/geometry.py — edge_zone_from_bbox
[ ] 5. bed_monitor/overlay.py — draw_risk_points, draw_risk_hud
[ ] 6. tests/test_risk_rules.py, tests/test_geometry.py
[ ] 7. server.py — analysis_loop 연동, PoseStatus 확장, HUD
[ ] 8. run_pose.py — Rail 제거, risk 연동
[ ] 9. API_CLIENT_GUIDE.md 갱신
[ ] 10. 바탕화면 STATUS.md Phase 2 완료 표시
```

**예상 diff:** 신규 ~350줄, server/run_pose 수정 ~80줄.

---

## 11. Definition of Done

- [ ] overlay에 L/C/R zone 표시 (기존 유지)
- [ ] 손목이 bed_bbox 밖 → `risk_level` ≥ LOW (RTSP 또는 synthetic)
- [ ] `/status`에 `edge_zone`, `limb_overflow_max`, `risk_level` 반환
- [ ] `tests/test_risk_rules.py` 3+ 케이스 green
- [ ] `run_pose.py` Rail placeholder **제거**
- [ ] preset `overflow_med` 변경 시 level 경계 이동 확인
- [ ] Phase 2 전 `/status` 필드(in_bed, pose, pose_conf) **동작 동일**

---

## 12. Phase 3 핸드오프

Phase 2 완료 후 Phase 3에서 교체·추가할 항목:

| Phase 2 (image) | Phase 3 (bed_norm) |
|-----------------|---------------------|
| `edge_zone_from_bbox(cx_px)` | `edge_zone_bed_norm(x_bed)` |
| lateral overflow vs px bbox | 동일 수식, bed_norm 폭 1.0 |
| center in_bed | `out_bed_ratio` + 히스테리시스 |
| `preset_id` only | + `h_version`, `bed_moved_flag` |

`calc_limb_overflow` 시그니처는 유지하고, 내부에서 `coord_space` 분기만 추가하면 된다.

---

## 13. 참고 링크

| 문서 | 내용 |
|------|------|
| [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) | Phase 0~5 전체 로드맵 |
| [`FALL_RISK_SYSTEM_DESIGN.md`](./FALL_RISK_SYSTEM_DESIGN.md) S4.5 | overflow 수식 |
| [`../bed_seg/README.md`](../bed_seg/README.md) | bed_bbox 품질 (seg v1) |
| [`../bed_roi/README.md`](../bed_roi/README.md) | ROI clip |
| `/home/dmc/pose/run_pose.py` | 이식 원본 |
| [`/home/dmc/바탕화면/pose-sixclass-STATUS.md`](/home/dmc/바탕화면/pose-sixclass-STATUS.md) | 프로젝트 전체 현황 |
