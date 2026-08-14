# pose-sixclass 실행 계획

> **작성:** 2026-06-18  
> **최종 갱신:** 2026-06-23 (2차)  
> **프로젝트:** `/home/dmc/pose-sixclass` (git 없음, 로컬 폴더)  
> **목표:** 고정 카메라 영상으로 **6-class 포즈 모니터링** + **0.1초 시계열 feature** + **rule 기반 낙상 위험 신호** 구축

---

## 1. 현재 어디까지 됐는가

### 완료 ✅

| 영역 | 내용 |
|------|------|
| **6-class ML** | `my_model_six_check.keras`, `pose_dataset_six.csv` (~27k) |
| **실시간 API** | `server.py` — RTSP 640×360 → bed-seg → approx zone → pose → **skeleton rule** → 6-class → `/status` + MJPEG |
| **bed_monitor/** | `geometry`, `risk_rules`, `features`, `live`, `temporal`, `bed_zone`, `bed_detect`, `config` |
| **실시간 rule** | skeleton `person_detected` + `seg_attachment` + overflow + edge + 이벤트 (`left_bed`, `high_overflow`, `edge_fast`) |
| **approx bed zone** | `bed_zone.py` — mask dilate + ROI/bbox 폴백 (`config/presets/approx_seg.json`, `run_server.sh` 기본) |
| **침대 seg v1** | `yolo11n-bed-seg.pt` (class **0** = bed), live conf `0.01` |
| **침대 ROI** | `bed_roi/bed_roi.json` (640×360), `USE_BED_ROI=1` |
| **Phase 2 API** | `/status`: `edge_zone`, `limb_overflow_max`, `risk_level`, `seg_attachment`, `person_detected`, `bed_event` … |
| **배치 enrich** | `enrich_timeseries.py` — live와 동일 `enrich_from_keypoints()` + `build_approx_bed_zone` |
| **배치 이벤트 스크립트** | `detect_timeseries_events.py` — enriched CSV → `runs/timeseries_events/` |
| **배치 원클릭** | `run_enrich_timeseries.sh` — enrich + detect 연속 실행 |
| **GT 도구** | `label_segment_event.py`, `validate_timeseries_segments.py`, `config/segment_events_template.json` |
| **백업** | `pose-sixclass_20260623.zip` (~814MB) + 폴더 복사 — Transcend 8GB USB |
| **배치 시계열 v1** | `extract_raw_timeseries.py` — USB 8영상 **1Hz** (~31k행, USB에 있음) |
| **10Hz 스크립트** | `run_timeseries_10hz.sh`, `run_enrich_timeseries.sh` |
| **RTSP 스펙 확인** | `192.168.0.161:8554` → H.264 **640×360 @ 20fps** (ffprobe) |
| **fall_monitor 병합 M1–M4** | `scoring.py`, `/status` `fall_score`, `enrich_timeseries` v2 — [`docs/MERGE_BASELINE.md`](docs/MERGE_BASELINE.md) |
| **설계 문서** | `docs/FALL_RISK_SYSTEM_DESIGN.md`, `IMPLEMENTATION_PLAN.md`, `PHASE_2.md`, `PHASE_TIMESERIES.md` |

### 준비만 됨 △

| 영역 | 내용 |
|------|------|
| **rule 숫자 튜닝** | attach 28%/55%, hold 0.3s 등 **초기값** — 현장 GT 검증 전 |
| **bed seg 품질** | v1 학습 ~37장, live mask 끊김 → approx zone으로 보완 |
| **5Hz / 10Hz 배치** | 스크립트 있음, USB `Raw_data` 마운트 필요 |
| **배치 이벤트 rule** | `detect_timeseries_events.py` 구현됨 — `in_bed` 기준, live는 `seg_attachment` → **미통일** |
| **Stage A extract** | `extract_raw_timeseries.py` — bed_monitor 미사용, person 판정 구버전 |
| **Stage C window** | `aggregate_timeseries_windows.py` **미구현** |
| **구간 manifest** | `build_raw_segments_manifest.py`, `annotate_raw_motion.py` (자동 motion만, GT 아님) |
| **난간(rail)** | `rail/` 복원 — `run_server.sh` 기본 `POSE_USE_RAIL=1` (필요 시 `0`) |
| **`run_pose.py`** | GUI 디버그, **server와 rule 미동기화** |

### 미착수 ❌

| 영역 | 내용 |
|------|------|
| **구간 이벤트 라벨 (GT)** | `segment_events.json` **미수집** — 도구만 준비 |
| **`out_bed_ratio`, Homography** | Phase 3 |
| **`risk_score` 0~10** | Phase 4 |
| **TCN/GRU** | Phase 5, 데이터 부족 |
| **`pose/` 레거시 정리** | 미착수 |
| **API 문서 갱신** | `API_CLIENT_GUIDE.md` 구버전 필드 |

---

## 1.5 실시간 모니터링 rule (현재 동작)

> **원칙:** 침대 이탈·위험은 **6-class가 아니라 seg zone + skeleton + geometry**.  
> 6-class는 HUD·`/status` 참고용.

```text
RTSP (640×360 @ 20fps)
  → YOLO bed-seg (class 0, conf 0.01)
  → ROI clip + approx zone (mask dilate 14px / bbox / roi_only)
  → YOLO pose (yolo11m-pose.pt)
  → skeleton_person_detected (core≥1, kpt≥5, conf≥0.3)
  → seg_attachment: on_seg | partial | off_seg
  → limb_overflow, edge_zone (L/C/R), center_speed
  → LiveEventTracker: left_bed | high_overflow | edge_fast (0.3s hold)
  → Keras 6-class (person_detected 일 때만)
```

| 신호 | 판정 |
|------|------|
| `person_detected` | YOLO box X, 스켈레톤 품질 |
| `in_bed` | `on_seg` 또는 `partial` |
| `left_bed` | seg 붙음 → `off_seg` 0.3초 |
| preset 기본 | `POSE_PRESET=approx_seg` (`run_server.sh`) |

### 1.6 운영 아키텍처 · 성능

```text
RTSP (.161:8554) ──► server.py (분석 스레드, dev PC)
                         ├─ GET /status  ◄── 프론트 (.17 / .42 등) 폴링
                         └─ GET /viewer  (MJPEG HUD)
```

| 항목 | 값 |
|------|-----|
| Dev PC | Ryzen 9800X3D, RTX 5080 16GB, 32GB RAM |
| pipeline_fps | ~28–36 (YOLO GPU, **Keras 6-class CPU** 병목) |
| 모니터링 | `nvidia-smi` / `nvtop` (`jtop`은 Jetson 전용) |
| seg 주기 | `POSE_SEG_EVERY=3` (3프레임마다 bed-seg) |

**`/status` 주요 필드 (2026-06-23):**  
`in_bed`, `person_detected`, `seg_attachment`, `zone_quality`, `bed_source`, `kpt_on_seg_ratio`, `edge_zone`, `limb_overflow_max`, `center_speed`, `risk_level`, `bed_event`, `last_bed_event`, `rail_left_up`, `rail_right_up`, `pipeline_fps`, `latency_ms`

---

## 2. 전체 로드맵

```text
[현재] 실시간 seg+skeleton rule v1 ───────────────────────────► 운영 중

[완료] bed_monitor + preset + Phase 2 (server) + approx zone
    │
[Timeseries] 10Hz extract → enrich (공통) → detect events → validate vs GT
    │              △ extract·이벤트 rule 분리 / window 미구현
[Data]       구간 이벤트 라벨 (out_bed_floor) + rule TP/FP 리포트
    │
[Phase 3] out_bed_ratio, bed_norm, H 캘리브
    │
[Phase 4] risk_score 0~10, temporal window (live ring buffer)
    │
[Phase 5] TCN/GRU (데이터 충분 시)
```

| Phase | 한 줄 목표 | 상태 |
|-------|-----------|------|
| **0** | `bed_monitor/` 추출, preset JSON | ✅ |
| **1** | room / preset 설정 파일화 | △ `POSE_PRESET`, `approx_seg.json` |
| **2** | overflow + edge_zone + `/status` | ✅ server / △ run_pose |
| **3** | Homography, `out_bed_ratio` | ❌ |
| **4** | 시계열 창 feature, `risk_score` | △ `temporal.py` live만 |
| **5** | 배치 표준화, 학습 파이프라인 | △ extract만 |
| **TS** | 0.1초 feature 시계열 | △ enrich 코드, 배치 미실행 |
| **Data** | 구간 이벤트 라벨, 침대 밖 시나리오 | △ 도구만, 라벨 0 |

---

## 3. 시계열 계획 (우선 추진)

상세: [`docs/PHASE_TIMESERIES.md`](docs/PHASE_TIMESERIES.md)

### 3.1 파이프라인 4단계

```text
Stage A  extract_raw_timeseries.py     10Hz, pose+kpt (v1)
    ↓
Stage B  enrich_timeseries.py          overflow, motion, edge_zone (v2)
    ↓
Stage C  aggregate_timeseries_windows.py   rolling 1s/3s/5s (v3)
    ↓
Stage D  validate_timeseries_segments.py   GT segment_events vs auto events
```

**배치 한 번에 (USB 마운트 후):**

```bash
bash run_timeseries_10hz.sh                    # Stage A
bash run_enrich_timeseries.sh                  # Stage B + detect → runs/timeseries_events/
python validate_timeseries_segments.py         # Stage D (GT 필요)
```

**GT ↔ 예측 매핑 (`validate_timeseries_segments.py`):**

| GT `event_label` | 매칭되는 `event_type` |
|------------------|----------------------|
| `out_bed_floor`, `out_bed_stand`, `exit_normal` | `left_bed` |
| `unsafe_exit` | `left_bed`, `high_overflow`, `edge_fast` |
| `edge_observe` | `edge_fast`, `high_overflow` |
| `in_bed_normal` | (이벤트 없음 기대) |

**병합 상태 (2026-06-23):**

| 단계 | live (`server.py`) | 배치 | 비고 |
|------|-------------------|------|------|
| feature enrich | `enrich_from_keypoints` | `enrich_timeseries.py` | ✅ 동일 함수 |
| approx zone | `bed_zone.build_approx_bed_zone` | enrich 시 bed 캐시 | ✅ |
| 이벤트 | `LiveEventTracker` (seg_attachment) | `detect_timeseries_events` (`in_bed`) | △ **rule 불일치** |
| extract | — | `extract_raw_timeseries.py` | △ skeleton/seg 없음 |

### 3.2 샘플링

| 간격 | sample_hz | 출력 폴더 | 비고 |
|------|-----------|-----------|------|
| 1초 | 1.0 | `Raw_data/timeseries/` | **완료** |
| 0.2초 | 5.0 | `timeseries_5hz/` | 선택 |
| **0.1초** | **10.0** | **`timeseries_10hz/`** | **기본 목표** |

```bash
# USB 마운트 후
bash run_timeseries_10hz.sh
bash run_timeseries_10hz.sh --video "Raw0 (3).mp4" --force   # 한 영상 테스트
```

모델: `my_model_six_check.keras` (server와 동일)

### 3.3 CSV 스키마 진화

| 버전 | 컬럼 |
|------|------|
| **v1** (현재) | pose_class, pose_conf, kpt_0…33, person_detected |
| **v2** (enrich) | + `person_detected`, `seg_attachment`, `zone_quality`, `in_bed`, `edge_zone`, `limb_overflow_max`, `center_speed` … |
| **v3** (window) | + speed_max_1s, edge_stay_duration, … |

**핵심:** Enrich는 v1 JSON의 keypoint만으로도 가능 (`--from-json-only`) → MP4 재디코딩 없이 feature 실험.

### 3.4 시계열 구현 순서

```text
[x] T1  bed_monitor/ + config/presets/default.json, approx_seg.json
[x] T2  bed_monitor/features.py — motion EMA
[x] T3  enrich_timeseries.py (live와 공통 enrich)
[ ] T4  run_timeseries_10hz.sh — 1영상 (USB 필요)
[ ] T5  enrich 1영상 → v2 확인
[ ] T6  aggregate_timeseries_windows.py
[x] T7  validate_timeseries_segments.py (스크립트)
[ ] T7b validate 실행 — GT 라벨 + 배치 이벤트 필요
[ ] T8  detect_timeseries_events — `LiveEventTracker` / `seg_attachment` rule 통일
[ ] T9  8영상 full 10Hz + enrich 배치
```

---

## 4. 실시간 계획 (Phase 2)

상세: [`docs/PHASE_2.md`](docs/PHASE_2.md)

### 4.1 server.py — 완료 항목

| 산출물 | 상태 |
|--------|------|
| `edge_zone` L/C/R | ✅ |
| `limb_overflow_max` | ✅ |
| `risk_level` SAFE/LOW/MED/HIGH | ✅ |
| `person_detected` (skeleton) | ✅ |
| `seg_attachment` + `zone_quality` | ✅ |
| `bed_event` / `last_bed_event` | ✅ |
| `/status` 확장 | ✅ |
| HUD `Zone:` / `Attach:` | ✅ |

### 4.2 가중치 (운영)

| 역할 | 파일 |
|------|------|
| bed seg | `yolo11n-bed-seg.pt` (class 0) |
| pose | `yolo11m-pose.pt` |
| 6-class | `my_model_six_check.keras` |

### 4.3 run_pose.py — 미완

| 항목 | 조치 |
|------|------|
| bed_monitor 연동 | ❌ server와 rule 분리 |
| FallDetector | GUI 전용 유지 |
| Rail 하드코딩 | 정리 필요 |

### 4.4 Phase 2 체크리스트

```text
[x] config/presets/default.json, approx_seg.json
[x] bed_monitor/config.py, geometry.py, risk_rules.py, features.py, live.py, temporal.py, bed_zone.py
[x] server.py — /status + HUD + approx zone
[ ] tests/test_risk_rules.py, test_geometry.py
[ ] run_pose.py — bed_monitor 동기화
```

---

## 5. 목표 폴더 구조

```text
pose-sixclass/
├── server.py, run_pose.py, run_server.sh
├── extract_raw_timeseries.py      # Stage A
├── enrich_timeseries.py           # Stage B
├── detect_timeseries_events.py    # 배치 이벤트
├── label_segment_event.py         # GT 구간 추가 CLI
├── validate_timeseries_segments.py
├── run_timeseries_10hz.sh, run_enrich_timeseries.sh
├── aggregate_timeseries_windows.py   # 미구현
├── config/
│   ├── segment_events_template.json
│   ├── segment_events.json           # GT (수동 생성)
│   └── presets/
│       ├── default.json
│       └── approx_seg.json           # dilate zone + 관대 threshold
├── my_model_six_check.keras
├── yolo11n-bed-seg.pt, yolo11m-pose.pt
├── bed_roi/, bed_seg/
├── bed_monitor/
│   ├── config.py, geometry.py, risk_rules.py
│   ├── features.py, live.py, temporal.py
│   ├── bed_zone.py, bed_detect.py
│   └── __init__.py
└── docs/ …
```

---

## 6. 데이터 위치

| 데이터 | 경로 | 비고 |
|--------|------|------|
| Raw 8영상 | `/media/dmc/Moredigm1/Dataset/Raw_data/video/` | **dev PC 로컬 없음** — USB 마운트 필요 |
| 1Hz 시계열 | `.../Raw_data/timeseries/` | USB에만 (~31k행) |
| 10Hz 시계열 | `.../Raw_data/timeseries_10hz/` | 미생성 |
| enrich 출력 | `.../timeseries_10hz_enriched/` | 미생성 |
| 배치 이벤트 | `runs/timeseries_events/` | 스크립트만, 미실행 |
| 학습 CSV | `pose_dataset_six.csv` | 로컬 |
| E2E 검증 | `../pose/extracted_frames/` | labels.json |
| 침대 polygon | `bed_seg/manual_labels/` | 52장 |
| 구간 manifest | `Raw_data/raw_segments_manifest_30s.json` | **이벤트 라벨 없음** |
| RTSP 카메라 | `192.168.0.161:8554/stream` | **640×360 @ 20fps** H.264 |
| 구간 이벤트 라벨 | `config/segment_events.json` | **도구만** (`label_segment_event.py`) |

USB 미마운트 시: 1Hz CSV 일부를 로컬로 복사해 enrich 개발 가능.

---

## 7. 데이터 계획 — 침대 밖 · 시계열 검증

> **핵심:** 침대 밖 시계열을 보려면 pose 학습 데이터만으로는 부족하다.  
> **자동 feature CSV** + **구간 이벤트 라벨(사람)** + **혼동 클립(쭈구림 등)** 이 세 층이 필요하다.

### 7.0 필요 데이터 한눈에

침대 밖·쭈구려 앉기 시계열 검증에 **실제로 더 모아야 하는 것**과 **이미 있는 것**을 구분한다.

#### ① 자동 생성 (코드 + USB Raw 영상)

| 데이터 | 형태 | 지금 | 왜 필요한가 |
|--------|------|------|-------------|
| **10Hz 시계열 v1** | CSV/JSON, 0.1초 1행 | ❌ (1Hz만) | 시간 해상도 |
| **10Hz 시계열 v2** | v1 + feature 컬럼 | ❌ | 검증할 곡선 |
| **침대 seg / ROI** | 영상당 bbox·mask | △ ROI만 | in_bed, overflow |

v2 자동 컬럼 예: `in_bed`, `edge_zone`, `limb_overflow_max`, `center_speed`, `pose_class`, `pose_conf`  
→ **새 촬영 불필요.** `run_timeseries_10hz.sh` + `enrich_timeseries.py`.

#### ② 구간 이벤트 라벨 (사람이 붙임) — **가장 부족**

영상 파일 + `start_sec` / `end_sec` + 상황 태그. **rule 맞는지 검증하려면 필수.**

| event_label | 장면 | 1차 최소 구간 |
|-------------|------|---------------|
| `in_bed_normal` | 침대 안 누움·뒤척임 | **10+** (오탐 기준) |
| **`out_bed_floor`** | 침대 **밖** 바닥, **쭈구려 앉기** | **5~10** ★ |
| `edge_observe` | 침대 **위** 가장자리 앉기 | 5+ |
| `out_bed_stand` | 침대 밖 서기·걷기 | 5+ |
| `exit_normal` | 정상 하차 | 3+ |
| `unsafe_exit` | 빠른 이탈·낙상 의심 | 3+ (있으면) |
| `occluded` | 가림·품질 나쁨 | 있으면 |

저장: **`config/segment_events.json`** (도구 기본) · 템플릿: `config/segment_events_template.json`  
※ 배포 시 `Raw_data/meta/`로 복사해도 됨 — 검증 시 `--gt`로 경로 지정

#### ③ 혼동 클립 (2차, 짧게)

| 클립 | 목적 |
|------|------|
| 침대 **위** 가장자리 앉기 vs 바닥 **쭈구려** 앉기 (짝) | pose·in_bed 오판 측정 |
| 엉덩이만 침대, 발은 바닥 | 경계 케이스 |
| 난간 잡고 쭈구림 | overflow + pose 조합 |

각 30초~2분. Raw 8영상에서 잘라도 되고, 없으면 현장 짧게 촬영.

#### 이미 있어서 당장 더 안 모아도 되는 것

| 데이터 | 용도 | 시계열 검증과 관계 |
|--------|------|-------------------|
| `pose_dataset_six.csv` (~27k) | 6-class **학습** | 침대 위만 — 검증 라벨 아님 |
| 1Hz timeseries (8영상) | pose+kpt | enrich 전 단계 |
| `extracted_frames` | E2E 포즈 정확도 | 별도 |
| `bed_seg/manual_labels` | 침대 polygon | seg 품질 |

**쭈구려 앉기는 6-class에 없음** → pose 데이터를 먼저 늘리지 말고 **구간 라벨 `out_bed_floor`** 우선.

#### 수집 우선순위

```text
1순위  구간 이벤트 라벨 (특히 out_bed_floor · 쭈구림 5~10구간)
2순위  10Hz feature CSV (자동)
3순위  혼동 클립 (짧게, 2차)
```

**첫 액션:** Raw 영상 1개에서 쭈구림·침대 밖 구간 **타임스탬프만** `segment_events.json`에 기록.

---

### 7.1 데이터 3층 구조

| 층 | 형태 | 누가 | 용도 |
|----|------|------|------|
| **① Feature 시계열** | 10Hz CSV v2 | 자동 (enrich) | overflow, speed, edge_zone 곡선 |
| **② 구간 이벤트 라벨** | JSON/CSV, 10~60초 단위 | **수동 라벨** | rule TP/FP, 시나리오별 검증 |
| **③ 혼동 클립** | 짧은 mp4 또는 프레임 묶음 | 촬영·클립 | 쭈구림·가장자리 등 경계 케이스 |

```text
Raw MP4 ──► 10Hz v2 CSV (①)
                │
                ├── join ──► segment_events (②) ──► validate 리포트
                │
                └── 혼동 클립 (③) ──► 오분류·오판율 측정
```

### 7.2 지금 있는 것 vs 부족한 것

| 종류 | 상태 | 한계 |
|------|------|------|
| `pose_dataset_six.csv` (~27k) | ✅ | **침대 위** 6-class. 바닥 쭈구림 거의 없음 |
| 6-class `앉음_*` | ✅ | 침대 **위** 앉기 기준. 쭈구려 앉기 ≠ 별도 클래스 |
| 1Hz timeseries | ✅ | pose+kpt만. in_bed/overflow 없음 |
| `raw_segments_manifest_30s` | △ | 30초 조각만, `label_status: unlabeled` |
| **침대 밖 구간 라벨** | ❌ | **시계열 검증의 병목** |
| **쭈구림·바닥 앉기 클립** | ❌ | in_bed 오판·pose 혼동 테스트용 |

**6-class 분포 (참고):** 앉음_가장자리·엎드림_등이 많고, 모두 침대 위 라벨링(`extracted_frames`) 출신.

### 7.3 왜 쭈구려 앉기가 까다로운가

| 신호 | 침대 위 앉기 | 침대 밖 쭈구림 | 문제 |
|------|-------------|----------------|------|
| 6-class | `앉음_중앙` / `앉음_가장자리` | **같은 클래스로 나올 수 있음** | pose만으로 구분 불가 |
| `in_bed` (현재) | seg zone + skeleton 비율 | hip만 mask 안이면 partial | **approx zone** + `off_seg` |
| `limb_overflow` | 손·발 밖이면 상승 | 발이 안쪽이면 **낮게 나옴** | 쭈구림에서 둔감 |
| `center_speed` | 낮음 | 낮음 (가만히 앉음) | 정상으로 보일 수 있음 |

**결론:** 쭈구림은 7번째 pose 클래스를 당장 늘리기보다, **geometry feature + 구간 라벨 `out_bed_floor`** 로 먼저 검증한다.

### 7.4 구간 이벤트 라벨 taxonomy

`raw_segments_manifest` 또는 별도 `segment_events.json`에 붙일 값:

| event_label | 설명 | 설계서 대응 |
|-------------|------|-------------|
| `in_bed_normal` | 침대 안 정상·뒤척임 | C01 normal |
| `edge_observe` | 침대 **위** 가장자리 앉기·머뭇거림 | C02 |
| `out_bed_floor` | 침대 **밖** 바닥 — **쭈구려 앉기 포함** | (신규) |
| `out_bed_stand` | 침대 밖 서기·걷기 | C03/C04 전단 |
| `exit_normal` | 의도적 정상 하차 | C03 |
| `unsafe_exit` | 빠른 이탈·낙상 의심 | C04 |
| `occluded` | 이불·보호자 가림, 품질 불량 | C05, C06 |
| `unknown` | 미분류 | 라벨링 전 기본값 |

각 레코드 필드:

```json
{
  "video_file": "Raw0 (3).mp4",
  "start_sec": 120.5,
  "end_sec": 145.0,
  "event_label": "out_bed_floor",
  "notes": "침대 옆 바닥 쭈구려 앉음, 손으로 침대 난간 잡음",
  "labeled_by": "",
  "labeled_at": ""
}
```

템플릿: [`config/segment_events_template.json`](config/segment_events_template.json)

### 7.5 수집 카탈로그 (필수 시나리오)

Raw 8영상에서 구간을 골라 라벨링. **최소 목표(1차):**

| 우선 | 시나리오 | event_label | 최소 구간 수 |
|------|----------|-------------|-------------|
| P0 | 침대 안 장시간 안정 | `in_bed_normal` | 10+ (false alarm 기준) |
| P0 | 침대 밖 바닥 **쭈구려 앉기** | `out_bed_floor` | **5~10** ★ |
| P1 | 침대 위 가장자리 앉기 | `edge_observe` | 5+ |
| P1 | 침대 밖 서기·이동 | `out_bed_stand` | 5+ |
| P2 | 정상 하차 | `exit_normal` | 3+ |
| P2 | 빠른 이탈·낙상 의심 | `unsafe_exit` | 3+ (있으면) |

**2차 (현장 촬영):** 침대 위 앉기 vs 바닥 쭈구림 **짝 영상** — 같은 사람·같은 카메라 각도.

### 7.6 Rule v0 — 침대 밖 앉기 (검증용 초안)

Phase 3 `out_bed_ratio` 전까지 overflow + geometry로 시험:

```text
suspect_out_bed_sit =
  (limb_overflow_max > θ_low  OR  hip_y > bed_bbox_y_max)
  AND center_speed < ε_slow
  AND pose_class ∈ {앉음_중앙, 앉음_가장자리}   # 보조만, 단독 판정 금지
```

`validate_timeseries_segments.py`에서 `suspect_out_bed_sit` vs `event_label=out_bed_floor` 교차표 작성.

### 7.7 데이터 수집·라벨 순서

```text
[ ] D1  Raw 영상 1개에서 쭈구림/침대 밖 구간 타임스탬프
[ ] D2  label_segment_event.py 로 5~10구간 → config/segment_events.json
[ ] D3  run_timeseries_10hz.sh — 해당 영상 (USB)
[ ] D4  run_enrich_timeseries.sh → v2
[ ] D5  validate_timeseries_segments.py — precision/recall
[ ] D6  rule 임계값 튜닝 (approx_seg.json)
[ ] D7  8영상 전체 라벨 확장
```

### 7.8 포즈 학습 데이터 (후순위)

| 옵션 | 시기 | 비고 |
|------|------|------|
| 6-class 유지 + rule은 geometry | **지금** | 권장 |
| `pose_subtype` 메타만 추가 (`squat_floor` 등) | enrich CSV | 학습 없이 분석용 |
| 7번째 클래스 추가 | 데이터 500+장 확보 후 | 재학습 비용 큼 |

침대 밖 쭈구림을 `앉음_*` 학습에 **넣지 않음** — 의도적으로 geometry와 분리.

---

## 8. 진행도 (2026-06-23)

```text
[6-class ML]         ██████████ 100%
[실시간 seg+skeleton] █████████░  90%   (rule 구현, 튜닝·GT 전)
[bed_monitor]        █████████░  90%
[approx bed zone]    ████████░░  80%   (seg v1 품질이 병목)
[침대 seg / ROI]     ███████░░░  70%
[Phase 2 API]        █████████░  90%   (server 완료, run_pose X)
[배치 enrich]        ██████░░░░  60%   (코드 O, 실행 X)
[배치 이벤트 통일]   ████░░░░░░  40%   (detect 스크립트 O, rule 불일치)
[10Hz extract]       ██░░░░░░░░  20%
[구간 GT 라벨]       █░░░░░░░░░  10%   (도구만)
[Phase 3~4]          ░░░░░░░░░░   0%
[레거시 pose/ 정리]  ░░░░░░░░░░   0%
```

---

## 9. 당장 할 일 (우선순위)

1. **RTSP 현장 확인** — HUD `Zone:` / `Attach:` / 이벤트 오탐·누락
2. **GT 라벨 v0** — `label_segment_event.py`로 `out_bed_floor` 5~10구간
3. **USB 마운트** → 10Hz extract + enrich + `validate_timeseries_segments.py`
4. **배치 이벤트 rule** — `detect_timeseries_events.py`를 `seg_attachment` 기준으로 live와 통일
5. **bed seg v2** — RTSP/backlight 프레임 라벨 추가 학습
6. `run_pose.py` → `bed_monitor` 동기화 (선택)
7. `aggregate_timeseries_windows.py` (Phase 4 전 단계)

**하지 않을 것 (당분간)**

- Homography / calibration API
- TCN·GRU 학습
- 7번째 pose 클래스 (geometry+GT 우선)

---

## 10. 실행 명령 모음

```bash
# 실시간 서버 (approx_seg preset 기본)
bash run_server.sh
# http://<IP>:8000/viewer
# POSE_PRESET=default bash run_server.sh   # dilate zone 없이

# 주요 환경변수 (run_server.sh 기본값)
export POSE_RTSP_URL=rtsp://192.168.0.161:8554/stream
export POSE_PRESET=approx_seg
export POSE_FRAME_WIDTH=640
export POSE_USE_BED_ROI=1
export POSE_BED_SEG_CONF=0.01
export POSE_SEG_EVERY=3

# GT 구간 라벨 추가
python label_segment_event.py --video "Raw0 (3).mp4" --start 120 --end 145 --label out_bed_floor

# 10Hz 시계열 (USB — Moredigm1 마운트 후)
RAW_ROOT=/media/dmc/Moredigm1/Dataset/Raw_data bash run_timeseries_10hz.sh
RAW_ROOT=/media/dmc/Moredigm1/Dataset/Raw_data bash run_enrich_timeseries.sh
python validate_timeseries_segments.py --gt config/segment_events.json

# RTSP 스펙
ffprobe -rtsp_transport tcp -show_entries stream=width,height,r_frame_rate,codec_name \
  -of default=noprint_wrappers=1 "rtsp://192.168.0.161:8554/stream"

# E2E 포즈 검증
python validate_e2e.py
```

---

## 11. 관련 문서

| 문서 | 용도 |
|------|------|
| [`docs/FALL_RISK_SYSTEM_DESIGN.md`](docs/FALL_RISK_SYSTEM_DESIGN.md) | feature 수식, taxonomy, S4 시계열 정의 |
| [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | Phase 0~5 모듈·API 로드맵 |
| [`docs/PHASE_2.md`](docs/PHASE_2.md) | overflow·edge_zone 구현 상세 |
| [`docs/MERGE_BASELINE.md`](docs/MERGE_BASELINE.md) | fall_monitor 병합 M0–M4 baseline·원칙 |
| [`/home/dmc/바탕화면/pose-sixclass-STATUS.md`](/home/dmc/바탕화면/pose-sixclass-STATUS.md) | 폴더·용량 인벤토리 |
| [`bed_seg/README.md`](bed_seg/README.md) | 침대 seg 학습 |
| [`bed_roi/README.md`](bed_roi/README.md) | ROI clip |

---

## 12. 오픈 이슈

| # | 이슈 | 방향 |
|---|------|------|
| 1 | USB `Raw_data` dev PC에 없음 | `Moredigm1` 마운트 → `RAW_ROOT=...` |
| 2 | bed seg v1 약함 (conf 0.01) | approx zone + 재학습 |
| 3 | live vs 배치 이벤트 rule | `seg_attachment`로 통일 |
| 4 | rule 임계값 미검증 | GT + `validate_timeseries_segments.py` |
| 5 | `run_pose.py` vs `server.py` | bed_monitor import |
| 6 | 쭈구림 vs 앉음_가장자리 | `out_bed_floor` GT + geometry |
| 7 | RTSP 640×360 고정 | 파이프라인 일치, 상향 시 ROI·재학습 |
| 8 | `API_CLIENT_GUIDE.md` 구버전 | `/status` `fall_score` 등 반영 (M5) |
| 9 | RTSP `.161` publisher down | 복구 후 `MODE=rtsp` baseline 재캡처 |
