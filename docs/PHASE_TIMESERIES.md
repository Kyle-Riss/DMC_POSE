# Phase Timeseries — Raw MP4 → 0.1s feature 시계열

> **문서 목적**  
> USB `Dataset/Raw_data` 영상에서 **0.1초(10Hz) 단위 feature 시계열**을 만들고, rule 검증·Phase 4 temporal까지 이어지는 **배치 파이프라인**을 설계한다.  
> 상위 feature 정의는 [`FALL_RISK_SYSTEM_DESIGN.md`](./FALL_RISK_SYSTEM_DESIGN.md) S4를 따른다.

| 항목 | 값 |
|------|-----|
| **버전** | 0.1 |
| **기준 코드** | `extract_raw_timeseries.py`, `server.py` |
| **선행** | Phase 2 (`bed_monitor` geometry + overflow) — **Fast path로 동시 착수 가능** |
| **후속** | Phase 3 (`out_bed_ratio`, H), Phase 4 (risk_score, live buffer) |

---

## 1. 목표

| 목표 | 설명 |
|------|------|
| **해상도** | 기본 **10Hz** (0.1초마다 1행). 1Hz 산출물은 레거시·비교용 유지 |
| **스키마** | pose + **침대·이탈·motion·overflow** + 품질 컬럼 |
| **단일 로직** | 배치·실시간이 **같은 `bed_monitor` 함수** 호출 |
| **검증** | `raw_segments_manifest_30s.json` 구간과 join → rule TP/FP 측정 |
| **비목표 (본 Phase)** | TCN/GRU 학습, rail, Homography 캘리브 API |

---

## 2. As-Is

### 2.1 완료 ✅

```text
Raw_data/video/*.mp4
  + meta/frame_timestamps/{stem}_frame_timestamps.json
        │
        ▼
extract_raw_timeseries.py  (--sample-hz 1.0)
        │
        ▼
Raw_data/timeseries/       # 8영상 · ~31k행 · 2026-06-01
  {stem}.csv / {stem}.json
  timeseries_index.json
```

**현재 CSV 컬럼 (v1 — thin)**

```text
video_file, frame_idx, timestamp_sec, fps, rotation_bucket,
pose_class, pose_class_id, pose_conf, person_detected,
kpt_0 … kpt_33
```

### 2.2 준비만 됨 △

| 항목 | 파일 |
|------|------|
| 5Hz 배치 | `run_timeseries_5hz.sh` |
| 10Hz 배치 | `run_timeseries_10hz.sh` |
| 구간 manifest | `build_raw_segments_manifest.py` → `raw_segments_manifest_30s.json` |
| 구간 motion | `annotate_raw_motion.py` (픽셀 diff 기반, pose 무관) |

### 2.3 없음 ❌

- per-row: `in_bed`, `out_bed_ratio`, `edge_zone`, `limb_overflow_max`, motion
- rolling window 컬럼 (S4.6)
- `bed_monitor/` 공통 모듈
- timeseries ↔ segment 라벨 join 리포트

---

## 3. To-Be — 3단계 파이프라인

```text
┌─────────────────────────────────────────────────────────────────┐
│  Stage A — Extract (기존 확장)                                   │
│  extract_raw_timeseries.py                                      │
│  · YOLO pose + 6-class (행마다)                                  │
│  · sample_hz=10.0 → timeseries_10hz/                            │
│  · 모델: my_model_six_check.keras (server와 동일)                │
└────────────────────────────┬────────────────────────────────────┘
                             │ CSV v1
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage B — Enrich (신규)                                         │
│  enrich_timeseries.py                                           │
│  · 영상당 bed seg 1회 캐시 (또는 N프레임마다 갱신)                 │
│  · bed_roi clip                                                 │
│  · bed_monitor: edge_zone, limb_overflow, in_bed                  │
│  · bed_monitor: hip center → EMA velocity (S4.3)                │
└────────────────────────────┬────────────────────────────────────┘
                             │ CSV v2
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage C — Windows (신규)                                        │
│  aggregate_timeseries_windows.py                                │
│  · rolling 1s / 3s / 5s (sample_hz 기준 행 수로 환산)             │
│  · edge_stay_duration, speed_max_1s, out_bed_ratio_max_3s …      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage D — Validate (신규)                                       │
│  validate_timeseries_segments.py                                │
│  · segment manifest join                                          │
│  · rule trigger vs motion_level 비교 리포트                       │
└─────────────────────────────────────────────────────────────────┘
```

**설계 원칙:** Stage B는 **프레임 단위 feature만** 계산한다. 창(window) 집계는 Stage C에서만 한다 (배치·실시간 `temporal.py`와 동일 분리).

---

## 4. 샘플링 전략

| sample_hz | 간격 | 출력 폴더 | 용도 |
|-----------|------|-----------|------|
| 1.0 | 1.0s | `timeseries/` | 완료·대략적 탐색 |
| 5.0 | 0.2s | `timeseries_5hz/` | 중간 (선택) |
| **10.0** | **0.1s** | **`timeseries_10hz/`** | **기본 목표** |

```bash
bash run_timeseries_10hz.sh
# 한 영상 테스트
bash run_timeseries_10hz.sh --video "Raw0 (3).mp4" --force
```

**제약**

- Raw 영상 fps(보통 20) ≥ sample_hz 이어야 실질 Hz 달성. `sample_frame_ids()`는 동일 frame_idx 중복 제거.
- 10Hz × 8영상 ≈ **~310k행** (1Hz 대비 ~10배). GPU 배치 수 시간 예상.
- Stage B/C는 v1 CSV를 읽어도 되고, Extract에 bed feature를 합쳐 **한 번에 v2**를 써도 됨 (권장: **B는 별도 스크립트** → Extract 재실행 없이 v1 위에 실험 가능).

---

## 5. CSV 스키마

### 5.1 v1 (현행 — 유지)

`extract_raw_timeseries.py` 출력. 변경 최소화.

### 5.2 v2 (Enrich 후)

기존 v1 컬럼 + 아래 추가.

#### 침대·이탈 (Phase 2 image-space)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `coord_space` | str | `"image"` (Phase 3부터 `bed_norm` 병행) |
| `bed_bbox_x0`…`y1` | int | seg+ROI 후 침대 bbox |
| `bed_seg_ok` | bool | mask/box 추출 성공 |
| `in_bed` | bool | mask 중심 또는 bbox (server 동일) |
| `edge_zone` | str | `L` / `C` / `R` (hip cx 기준) |
| `limb_overflow_max` | float | Phase 2 `calc_limb_overflow` |
| `risk_level` | str | SAFE / LOW / MED / HIGH |

#### Motion (S4.3)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `center_x`, `center_y` | float | hip 중심 (px), 없으면 bbox center |
| `center_vx`, `center_vy` | float | EMA 속도 (px/s) |
| `center_speed` | float | ‖v‖ |
| `center_accel_like` | float | ‖Δv‖/Δt |
| `vertical_drop_rate` | float | Δy/Δt (머리맡 카메라 근사) |

#### 자세 파생

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `lying_flag` | bool | 누움 4종 |
| `sitting_edge_flag` | bool | `앉음_가장자리` |
| `pose_changed` | bool | `pose_class != prev` |

#### 품질 (S4.7)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `kpt_mean_conf` | float | 17 keypoints 평균 conf |
| `sample_hz` | float | 10.0 |
| `pipeline_version` | str | git 없으면 날짜 태그 |

**결측 정책**

- `person_detected=false` → motion·overflow 컬럼 `NaN`, `risk_level=SAFE`
- hip 없음 → bbox center로 center_x/y, 속도는 hold 최대 0.5s (S4.3)

### 5.3 v3 (Window — Stage C)

v2를 입력으로 **같은 파일에 컬럼 추가** 또는 `{stem}_windows.csv` 분리.

| 컬럼 | 창 | 정의 |
|------|-----|------|
| `speed_max_1s` | 1s | max(`center_speed`) |
| `out_bed_ratio_max_3s` | 3s | Phase 3 전까지 `limb_overflow_max` max로 대체 가능 |
| `edge_stay_duration` | 누적 | `edge_zone != C` 연속 초 |
| `stationary_duration` | 누적 | `center_speed < ε` 연속 초 |
| `lying_outside_duration` | 누적 | `lying_flag & overflow > θ` 연속 초 |

창 길이는 행 수로 환산: `win_rows = int(sample_hz * seconds)` (10Hz × 3s = 30행).

---

## 6. 모듈 구조 (Phase 2와 공유)

```text
pose-sixclass/
├── bed_monitor/                    # Phase 0+2와 동일 패키지
│   ├── geometry.py                 # edge_zone_from_bbox
│   ├── risk_rules.py               # calc_limb_overflow
│   ├── features.py                 # ★ 신규: motion EMA, pose flags
│   └── temporal.py                 # ★ Phase 4 live; Stage C와 수식 공유
├── extract_raw_timeseries.py       # Stage A (v1)
├── enrich_timeseries.py            # ★ Stage B (v2)
├── aggregate_timeseries_windows.py # ★ Stage C (v3)
├── validate_timeseries_segments.py # ★ Stage D
├── run_timeseries_10hz.sh
└── config/presets/default.json     # thresholds, ema_alpha, motion_eps
```

### 6.1 `bed_monitor/features.py` (배치·실시간 공통)

```python
@dataclass
class MotionState:
    prev_center: tuple[float, float] | None
    prev_v: tuple[float, float]
    last_seen_t: float

def update_motion(
    state: MotionState,
    center: tuple[float, float] | None,
    t_sec: float,
    *,
    ema_alpha: float = 0.3,
    hold_sec: float = 0.5,
    dt: float = 0.1,
) -> dict[str, float | None]:
    """→ center_vx, center_vy, center_speed, center_accel_like, vertical_drop_rate"""

def enrich_row(
  frame_bgr, pose_result, bed: dict, clf_result, prev_row, preset
) -> dict:
    """Stage B 한 행. server.analysis_loop와 동일 순서."""
```

**실시간 연동:** `server.py`는 행마다 `enrich_row`와 동일 로직을 호출 → `/status`에 `center_speed`, `edge_zone` 등 추가 (Phase 4).

### 6.2 Stage B — `enrich_timeseries.py`

```text
입력:  --in-dir  Raw_data/timeseries_10hz/
       --out-dir Raw_data/timeseries_10hz_enriched/  (또는 덮어쓰기 --inplace)
옵션:  --bed-seg-every 30   # 30행마다 seg 갱신 (기본: 영상당 1회)
       --bed-roi bed_roi/bed_roi.json
       --preset config/presets/default.json
```

알고리즘:

1. v1 CSV + 원본 MP4에서 `frame_idx`로 프레임 seek
2. 영상 첫 프레임(또는 every N)에서 `extract_bed_detection` → `cached_bed`
3. 각 행: kpt 복원 → `enrich_row` → v2 컬럼 append
4. `pd.DataFrame` 저장 + `enrich_index.json`

**Extract 통합 여부:** 초기에는 **분리 스크립트**로 v1 위에 반복 실험. 안정화 후 `extract_raw_timeseries.py`에 `--enrich` 플래그로 merge 가능.

### 6.3 Stage C — `aggregate_timeseries_windows.py`

```python
def rolling_max(series, window_rows: int) -> pd.Series: ...
def cumulative_duration(mask: pd.Series, dt: float) -> pd.Series: ...
```

stdin/stdout 없이 디렉터리 배치:

```bash
python aggregate_timeseries_windows.py \
  --in-dir Raw_data/timeseries_10hz_enriched \
  --sample-hz 10.0 \
  --windows 1,3,5
```

### 6.4 Stage D — `validate_timeseries_segments.py`

```text
join key: video_file + timestamp_sec ∈ [segment.start_sec, segment.end_sec)

출력: runs/timeseries_validate/
  report.md
  segment_triggers.csv   # segment_id, rule_fired, motion_level, ...
```

**Rule v0 (Phase 2+3 전):**

```text
trigger = (risk_level >= MED) OR (edge_zone != 'C' AND center_speed > v_thresh)
```

`annotate_raw_motion.py`의 `motion_level=high_motion`과 교차표 작성.

---

## 7. Raw_data 폴더 (목표)

```text
Dataset/Raw_data/
├── video/                          # 8× MP4
├── meta/frame_timestamps/
├── timeseries/                     # 1Hz v1 (유지)
├── timeseries_10hz/                # 10Hz v1 ← Stage A
├── timeseries_10hz_enriched/       # 10Hz v2 ← Stage B
├── timeseries_10hz_windows/        # 10Hz v3 ← Stage C (선택)
├── raw_segments_manifest_30s.json
└── runs/timeseries_validate/       # Stage D 리포트
```

---

## 8. 구현 단계 (Timeseries Track)

| Step | 작업 | 산출물 | 의존 |
|------|------|--------|------|
| **T0** | USB 마운트, 1영상 10Hz smoke | `timeseries_10hz/Raw0 (2).csv` | — |
| **T1** | Phase 2 Fast path: `bed_monitor` + preset | geometry, risk_rules | PHASE_2.md §2.2 |
| **T2** | `bed_monitor/features.py` + unit tests | motion EMA tests | T1 |
| **T3** | `enrich_timeseries.py` | v2 CSV 1영상 | T0, T2 |
| **T4** | 8영상 10Hz full extract | ~310k v1 행 | T0 |
| **T5** | 8영상 enrich 배치 | v2 full | T3, T4 |
| **T6** | `aggregate_timeseries_windows.py` | v3 | T5 |
| **T7** | `validate_timeseries_segments.py` | report | T5, manifest |
| **T8** | server `/status`에 motion 필드 (선택) | live parity | T2 |

**병렬 가능:** T1/T2는 USB 없이 synthetic kpt로 진행. T0은 USB 필요.

---

## 9. 성능·리소스 (추정)

| 단계 | 8영상 · 10Hz | 비고 |
|------|----------------|------|
| Stage A Extract | 2–6h GPU | YOLO+Keras every row |
| Stage B Enrich | +30–60% | seg 캐시 시 seek+pose 재사용 없음 — **v1 kpt만으로 overflow 계산 가능**하면 MP4 seek 최소화 |
| Stage C Windows | 수 분 | pandas only |

**최적화 (T3 설계):**

- overflow·edge_zone·motion은 **v1의 kpt + 영상당 bed_bbox 1개**로 계산 가능 → Enrich가 MP4를 다시 읽지 않아도 됨 (bed_bbox를 v1 JSON sidecar에 저장).
- v1 JSON에 `keypoints_xy` 이미 있음 → **Enrich kpt-only fast path** 권장.

```text
enrich_timeseries.py --from-json-only   # MP4 없이 v1 JSON+CSV만
```

단, `in_bed` mask 검증은 MP4 필요 — Phase 3 전까지 bbox center로 충분.

---

## 10. 실시간과의 정합

| 항목 | 배치 (10Hz) | 실시간 (`server.py`) |
|------|-------------|----------------------|
| pose + 6-class | ✅ 동일 모델 | ✅ |
| bed seg | 영상당 1회 캐시 | N프레임마다 1회 |
| overflow, edge_zone | `enrich_row` | 동일 함수 |
| motion EMA | 행마다 dt=0.1 | `pipeline_fps` 역수 또는 captured_at |
| window feature | Stage C 오프라인 | `temporal.py` ring buffer (Phase 4) |

**Golden rule:** `bed_monitor/features.py` 수정 시 `tests/test_features.py`가 배치·실시간 모두 보호.

---

## 11. 테스트

### 11.1 `tests/test_features.py`

| # | 케이스 | 기대 |
|---|--------|------|
| 1 | 정지 — center 동일 3행 | speed ≈ 0 |
| 2 | x만 10px/0.1s 이동 | vx ≈ 100 px/s |
| 3 | 0.5s 결측 후 복귀 | hold 후 리셋, spike 없음 |

### 11.2 Enrich 스모크

```bash
python enrich_timeseries.py \
  --csv timeseries_10hz/Raw0\ \(2\).csv \
  --json timeseries_10hz/Raw0\ \(2\).json \
  --out /tmp/enriched_smoke.csv
```

필수: `limb_overflow_max`, `edge_zone`, `center_speed` 컬럼 존재, 행 수 = 입력과 동일.

### 11.3 Window 스모크

10행 synthetic CSV → `speed_max_1s` at 10Hz = 10행 max.

---

## 12. Definition of Done (Timeseries Track v0)

- [ ] `timeseries_10hz/` 8영상 v1 생성 (`my_model_six_check.keras`)
- [ ] `enrich_timeseries.py` — JSON-only fast path로 v2 1영상
- [ ] v2에 `edge_zone`, `limb_overflow_max`, `center_speed` 포함
- [ ] `aggregate_timeseries_windows.py` — `speed_max_1s`, `edge_stay_duration` 1영상
- [ ] `validate_timeseries_segments.py` — segment join 리포트 1장
- [ ] `bed_monitor/features.py` + tests green
- [ ] 본 문서 §8 T1–T7 체크리스트 팀 공유

---

## 13. 구현 체크리스트 (권장 순서)

```text
[ ] 1.  PHASE_2.md Fast path — bed_monitor + preset (T1)
[ ] 2.  bed_monitor/features.py — motion EMA (T2)
[ ] 3.  tests/test_features.py
[ ] 4.  enrich_timeseries.py --from-json-only (T3)
[ ] 5.  run_timeseries_10hz.sh — 1영상 smoke (T0)
[ ] 6.  enrich 1영상 → v2 확인
[ ] 7.  aggregate_timeseries_windows.py (T6)
[ ] 8.  validate_timeseries_segments.py (T7)
[ ] 9.  8영상 full batch (T4–T5)
[ ] 10. 바탕화면 STATUS.md 시계열 섹션 갱신
```

---

## 14. 참고

| 문서·파일 | 내용 |
|-----------|------|
| [`PHASE_2.md`](./PHASE_2.md) | limb overflow, edge_zone (Enrich 선행) |
| [`FALL_RISK_SYSTEM_DESIGN.md`](./FALL_RISK_SYSTEM_DESIGN.md) S4 | feature 수식 |
| [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) | Phase 0–5 전체 |
| `extract_raw_timeseries.py` | Stage A |
| `run_timeseries_10hz.sh` | 10Hz 실행 |
| `docs/canvas/pose-raw-timeseries-data.canvas.tsx` | 1Hz 통계 |

---

## 15. 오픈 이슈

| # | 이슈 | 결정 방향 |
|---|------|-----------|
| 1 | Enrich를 Extract에 합칠지 | **v0는 분리**, 안정 후 `--enrich` |
| 2 | bed seg 갱신 주기 | 영상당 1회 → drift 있으면 every 30s |
| 3 | `out_bed_ratio` | Phase 3까지 `limb_overflow`로 rule v0 |
| 4 | 10Hz 전체 8영상 vs 긴 1영상만 | **Raw0 (3)** 먼저 (4h 분량) |
| 5 | rotation_bucket | manifest `rotation_profile` 수동 매핑 테이블 |
