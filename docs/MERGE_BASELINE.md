# Phase M0 — Merge Baseline (pose-sixclass vs fall_monitor)

> **캡처일:** 2026-07-03  
> **목적:** 병합 전 두 스택이 **같은 프레임**에서 무엇이 다른지 수치·사례로 기록  
> **다음 단계:** M1 설정 통합 (`scoring` 블록 → `approx_seg.json`)

---

## 1. 실행 방법

### 오프라인 (저장 프레임, RTSP 불필요)

```bash
cd /home/dmc/pose-sixclass
bash scripts/run_merge_baseline.sh                    # 기본: bed_seg/rtsp_raw 30장
# 또는
python scripts/merge_baseline_capture.py \
  --image-dir bed_seg/rtsp_raw --max-images 20 \
  --out runs/merge_baseline/m0_offline_YYYYMMDD
```

### 라이브 RTSP (카메라 온라인 시)

```bash
MODE=rtsp DURATION=600 bash scripts/run_merge_baseline.sh
# 10분, 2Hz 샘플 → runs/merge_baseline/<stamp>/
```

### server /status 폴링만 (fall_monitor 없음)

```bash
# 터미널 1
bash run_server.sh
# 터미널 2
MODE=status DURATION=600 bash scripts/run_merge_baseline.sh
```

### 산출물

| 파일 | 내용 |
|------|------|
| `baseline_frames.jsonl` / `baseline_rtsp.jsonl` | 프레임별 PS + FM + compare |
| `baseline_summary.json` | 일치율·점수 통계 |

스크립트: [`scripts/merge_baseline_capture.py`](../scripts/merge_baseline_capture.py)

---

## 2. 환경 스냅샷

| 항목 | pose-sixclass | fall_monitor |
|------|---------------|--------------|
| preset/config | `POSE_PRESET=approx_seg` | `fall_config.json` |
| bed seg conf | **0.01** | **0.25** |
| person 판정 | skeleton (core≥1, kpt≥5) | YOLO pose primary |
| in_bed | `seg_attachment` on_seg/partial | hip in 사다리꼴 ROI |
| 위험 출력 | `risk_level` (overflow), `bed_event` | `fall_score` 0~100 |
| 난간 | `rail/` ref diff | Hough in bed ROI |
| RTSP (M0 당일) | **404 Not Found** | 동일 |

---

## 3. 캡처 결과

### 3.1 오프라인 20장 (`bed_seg/rtsp_raw`)

경로: `runs/merge_baseline/m0_offline_20260703/`

| 지표 | 값 |
|------|-----|
| 프레임 | 20 |
| in_bed 일치 | **100%** (둘 다 “침대 밖/무인”) |
| person 일치 | **25%** |
| pose 일치 | **30%** |
| PS `off_seg` | 0 |
| FM `OUT_OF_BED` | 15 |
| FM fall_score (person 있을 때) | min 34 / max 82 / mean **67.3** |

**해석:** 이 배치는 대부분 **원거리·저품질 kpt** 장면.  
PS는 skeleton 기준으로 person=False(20/20), FM은 YOLO로 person=True(15/20).  
in_bed “일치”는 둘 다 negative라서 수치가 높게 나옴 — **실질 동작 일치가 아님**.

### 3.2 스팟체크 3장 (사람 포함 프레임)

경로: `runs/merge_baseline/m0_spotcheck_20260703/`

| 이미지 | pose-sixclass | fall_monitor | 비고 |
|--------|---------------|--------------|------|
| `rail/reference/rtsp_one_frame.jpg` | person=✓ attach=**partial** in_bed=**YES** risk=SAFE | in_bed=✗ **OUT_OF_BED score=85** | **핵심 불일치** |
| `rtsp_0012_...jpg` | person=✗ | person=✓ OUT_OF_BED 85 | person 판정 차이 |
| `rtsp_0022_...jpg` | person=✗ | person=✗ | 일치 |

#### 대표 케이스: `rtsp_one_frame.jpg`

```text
pose-sixclass:  앉음_가장자리 | partial | in_bed=YES | overflow=0 | risk=SAFE
fall_monitor:   앉음_가장자리 | OUT_OF_BED  | score=85 | hip ∉ trapezoid
```

→ **같은 pose, 반대 in_bed** — 병합 시 FM의 hip-in-polygon을 PS `seg_attachment`로 **대체하면 안 됨**.

### 3.3 라이브 RTSP

| 상태 | 비고 |
|------|------|
| ❌ 미실행 | `rtsp://192.168.0.161:8554/stream` → 404 (2026-07-03 09:08) |
| 재시도 | `MODE=rtsp DURATION=600 bash scripts/run_merge_baseline.sh` |

---

## 4. 차이 요약 (병합 설계 입력)

| # | 차이 | pose-sixclass | fall_monitor | M1+ 방향 |
|---|------|---------------|--------------|----------|
| 1 | **in_bed** | seg_attachment + approx zone | hip ∈ trapezoid | **PS 유지**, score만 FM 흡수 |
| 2 | **person** | skeleton 품질 | YOLO box/kpt | PS 유지 (오탐↓), FM score는 PS person gate |
| 3 | **OUT_OF_BED score** | 없음 (이벤트만) | hip 밖 → 85점 | `seg_attachment=off_seg`일 때만 85 |
| 4 | **bed seg conf** | 0.01 | 0.25 | 통합 시 **0.01** + approx zone |
| 5 | **위험 출력** | overflow level + events | 0~100 score | **병렬** — event ≠ score 대체 |
| 6 | **난간** | ref diff (`rail/`) | Hough + rail_risk | PS 검출 → FM rail_risk 표 |

---

> **다음 단계:** M5 GT 검증 (RTSP·라벨 준비 후) · 라이브 RTSP baseline (카메라 복구 후)

---

## 5. M0 결론 · Gate

| Gate | 기준 | 결과 |
|------|------|------|
| **M0-a** | 동일 프레임 비교 도구 | ✅ `merge_baseline_capture.py` |
| **M0-b** | 오프라인 baseline 숫자 | ✅ 20+3장 |
| **M0-c** | 라이브 RTSP 10분 | ⏳ 카메라 복구 후 |
| **M0-d** | 병합 원칙 검증 | ✅ FM in_bed으로 PS 대체 **금지** 확인 |

---

## 6. M1–M4 병합 완료 (2026-07-03, RTSP 없이)

| 단계 | 내용 | 상태 |
|------|------|------|
| **M1** | `approx_seg.json` → `scoring` 블록, `out_of_bed.trigger: seg_off` | ✅ |
| **M2** | `bed_monitor/scoring.py`, `zone_norm.py` | ✅ |
| **M3** | `live.py` + `server.py` `/status` · HUD `fall_score` | ✅ |
| **M4** | `enrich_timeseries.py` v2 `fall_score` 컬럼 | ✅ |

**병합 원칙 (유지):**

- `risk_level`, `bed_event`, `seg_attachment` — **기존 PS 유지**
- `fall_score` 0~100 — **FM 흡수, 병렬 출력** (대체 아님)
- OUT_OF_BED 85점 — **`seg_attachment == off_seg`일 때만**

### 오프라인 스팟체크 (`rtsp_one_frame.jpg`, approx_seg)

| | pose-sixclass (병합 후) | fall_monitor |
|--|-------------------------|--------------|
| in_bed | YES (partial) | OUT_OF_BED |
| fall_score | ~80 (pose+rail HIGH, status **IN_BED**) | **85** |
| fall_status | IN_BED | OUT_OF_BED |

→ 병합 후에도 **FM hip-ROI 85점으로 PS를 덮지 않음** — M0 결론과 일치.

### 검증 명령 (RTSP 불필요)

```bash
cd /home/dmc/pose-sixclass
POSE_PRESET=approx_seg python scripts/merge_baseline_capture.py \
  --images rail/reference/rtsp_one_frame.jpg \
  --out runs/merge_baseline/m4_spotcheck
```

`baseline_summary.json`에 `fall_score_pose_six_*` 필드 확인.

---

## 7. 다음 액션 (M5+, 카메라 복구 후)

1. RTSP 복구 → `MODE=rtsp DURATION=600 bash scripts/run_merge_baseline.sh`
2. GT `segment_events.json` → rule TP/FP (`validate_timeseries_segments.py`)
3. `detect_timeseries_events.py` ↔ live `seg_attachment` rule 통일 (선택)
4. `API_CLIENT_GUIDE.md` — `fall_score`, `fall_level`, `fall_status` 필드 추가

---

## 8. 원시 데이터

```
runs/merge_baseline/m0_offline_20260703/
  baseline_frames.jsonl
  baseline_summary.json
runs/merge_baseline/m0_spotcheck_20260703/
  baseline_frames.jsonl
  baseline_summary.json
```
