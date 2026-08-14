# 영상 기반 낙상 위험 모니터링 · 데이터셋 설계서

> **문서 목적**  
> 고정 카메라(머리맡) 영상만으로 침대 이탈·시계열 motion feature·rule 기반 위험도를 구축·검증하기 위한 설계를 **§1부터 순차적으로** 상세 정리한다.  
> IMU/가속도계 센서는 사용하지 않는다.

| 항목 | 값 |
|------|-----|
| **버전** | 0.2 |
| **대상 코드** | `pose/`, `pose-sixclass/`, `pose-webviewer/` |
| **모델** | `yolo11n-seg.pt`, `yolo11m-pose.pt`, `my_model_six_check.keras` (6-class) / `my_model.keras` (12-class) |

**목차**

1. [시스템 개요 (S1)](#s1-시스템-개요)
2. [배포·운영 전제 (S2)](#s2-배포운영-전제-병원)
3. [처리 파이프라인 (S3)](#s3-처리-파이프라인)
4. [Feature 정의 (S4)](#s4-feature-정의-프레임--시계열)
5. [침대 이동 · 호모그래피 (S5)](#s5-침대-이동--호모그래피-운영)
6. [스코어링 vs 라벨링 (S6)](#s6-스코어링-vs-라벨링)
7. [난간 설계 (S7)](#s7-난간-설계)
8. [데이터셋 산출물 (S8)](#s8-데이터셋-산출물)
9. [로드맵 (S9)](#s9-단계별-로드맵)
10. [API·실시간 (S10)](#s10-api실시간)
11. [리스크·한계 (S11)](#s11-리스크한계)
12. [용어집 (S12)](#s12-용어집)
13. [코드 앵커 (S13)](#s13-참고-현재-코드-앵커)

---

<a id="s1-시스템-개요"></a>
## S1. 시스템 개요

### S1.1 문제 정의

병실에서 **낙상·침대 이탈**을 조기에 감지하려면, 사람의 **위치(침대 대비)**, **자세**, **시간에 따른 움직임**을 함께 봐야 한다.  
본 프로젝트는 **IMU·웨어러블 없이**, **머리맡 고정 단일 RGB 카메라**만으로 위 정보를 추정한다.

**핵심 난제**

| 난제 | 설명 |
|------|------|
| 정상 vs 위험 | 침대 밖으로 나가는 행동은 **정상 하차**와 **낙상** 모두에 존재 |
| 시계열 | 낙상은 단일 프레임이 아니라 **수 초 전후 패턴** |
| 굴림 | 옆으로 뒤척이는 것은 **일상**이지만 낙상 직후에도 **비슷한 궤적** |
| 기하 | 설치마다 픽셀 의미가 달라 **캘리브·프리셋** 없으면 rule 재현 어려움 |
| 부하 | N병실 × (seg + pose + 분류)는 **중앙 서버 병목** 가능 |

### S1.2 한 줄 정의

> **영상 기반 temporal fall-risk scoring**  
> 고정 카메라에서 침대 segmentation, 사람 pose, 침대 이탈 비율, 정규화된 중심점 이동·변화율, 자세 클래스·체류 시간을 추출하고, MP4·CSV·JSON으로 저장한 뒤 **rule-based 정책을 검증**하고, 데이터가 충분해지면 **시계열 모델**로 확장한다.

### S1.3 시스템 목표 (Must / Should / Won’t)

**Must (필수)**

- 프레임 단위 **feature CSV** + 영상 단위 **JSON metadata** 수집
- **라벨(사람)** 과 **risk_score(rule)** 분리 저장
- 동일 설치 프리셋에서 **rule 임계값 재사용**
- 기존 스택 활용: YOLO-seg(침대) + YOLO-pose + Keras 자세 분류

**Should (권장)**

- 침대 이동 시 **호모그래피(H) 자동/반자동 갱신**
- 프리셋 기반 **난간 기하** + 손목·팔 overflow
- MP4 배치 재현으로 **rule vs label** 오프라인 평가

**Won’t (초기 범위 밖)**

- IMU·깊이 카메라 **필수** 요구
- 영상만으로 **물리 m/s² 가속도** 보장
- 난간 **영상 세그 자동 인식** (v2)
- 단일 프레임만으로 **fall 확정**

### S1.4 이해관계자·산출물

| 역할 | 관심사 | 산출물 |
|------|--------|--------|
| **데이터 수집** | MP4, CSV, 재현 가능한 pipeline 버전 | `episode_xxx/` |
| **라벨러** | 구간 정답, 검수용 프레임 | JSON `event_segments`, 선택 JPG |
| **알고리즘** | rule 검증, 임계 튜닝 | CSV `risk_*`, 평가 리포트 |
| **현장 운영** | 설치 변경 최소, RTSP 안정 | `preset_id`, 4점 보정 UI |
| **서버 운영** | N스트림 부하 | 엣지/저주기 seg 설계 |

### S1.5 용어·표현 규칙 (문서·코드·보고서 공통)

| 금지/비권장 | 권장 |
|------------|------|
| 가속도계로 낙상 판별 | 영상 좌표계 motion feature |
| `accel` (단독) | `center_accel_like` |
| `risk_score` = 정답 | `label` = 정답, `risk_score` = 시스템 출력 |
| 낙상 = 침대 밖 | 침대 밖 + **시간·자세·정지·전이** |

### S1.6 논리 아키텍처 (3계층)

```text
┌─────────────────────────────────────────────────────────┐
│  L3  정책·검증                                          │
│      rule risk_score, 구간 label, 평가, (미래) TCN/GRU   │
├─────────────────────────────────────────────────────────┤
│  L2  Feature·시계열                                     │
│      이탈비율, bed_norm 좌표, speed/accel_like, 전이    │
├─────────────────────────────────────────────────────────┤
│  L1  인식 (모델 선점)                                   │
│      YOLO-seg(bed), YOLO-pose, Keras pose_class         │
│      + H (침대 평면), preset (설치 기하)                 │
└─────────────────────────────────────────────────────────┘
```

- **S1 단계 산출:** L1 모델·가중치·입출력 차원 **고정** (노드/DAG는 나중).
- **데이터셋 목적:** L2 feature + L3 rule 출력을 **같은 타임스탬프에 기록**.

### S1.7 데이터의 정체 (무엇을 “낙상 데이터”라 부를지)

본 시스템의 “낙상 데이터”는 **IMU 낙상 로그가 아니다.**

```text
고정 카메라 영상
  → 프레임 분석
  → 침대 segmentation + 사람 pose + 자세 class
  → timestamp별 feature (위치·이탈·motion-like)
  → CSV 저장
  → 영상 단위 JSON (구간 label, preset, pipeline 버전)
  → rule risk_score (당시 시스템 판단, 정답 아님)
```

**pseudo-acceleration:**  
사람 중심 \(c_t=(x_t,y_t)\)에 대해 \(v_t, a_t\) 형태의 **2차 차분**을 쓰되, **픽셀/bed_norm/body_norm 단위**이며 물리 가속도가 아님.

### S1.8 현재 구현 vs 목표 (상세)

#### S1.8.1 입력

| 항목 | 현재 | 목표 |
|------|------|------|
| 소스 | `RTSP_URL` 단일 (`192.168.0.157:8554` 등) | RTSP + **로컬 MP4** 배치 |
| 해상도 | `imutils.resize(width=800)` | 동일 + **preset별 crop 옵션** 검토 |
| fps | 캡처 fps 그대로 | CSV에 `fps`, `frame_idx`, `timestamp` 명시 |
| 동기 | 없음 | episode 단위 **단일 타임라인** |

#### S1.8.2 침대 (segmentation)

| 항목 | 현재 | 목표 |
|------|------|------|
| 모델 | `yolo11n-seg.pt` | 동일 (선점) |
| 클래스 | `classes=[59]` (COCO bed) | 동일 + 문서화 |
| conf | `0.2` | preset별 튜닝 가능, 기본 0.2 |
| 마스크 사용 | bbox **중심** in mask > 0.5 | **out_bed_ratio**, edge_zone, H 연동 |
| 시각화 | `run_pose`만 zone/risk 일부 | 수집 시 **선택적** overlay JPG |

#### S1.8.3 사람 (pose + 분류)

| 항목 | 현재 | 목표 |
|------|------|------|
| pose | `yolo11m-pose.pt`, conf 0.5 | 동일 |
| 분류 입력 | 17 keypoints × 2 = **34 dim** `xy` flatten | 동일 + **conf 가중** 검토 |
| 6-class | `my_model_six_check.keras` | `정면_누움` … `앉음_가장자리` |
| 12-class | `pose-webviewer`: `p01`…`p12` | 데이터셋 버전별 **model_id** 분리 |

#### S1.8.4 위험·낙상

| 항목 | 현재 | 목표 |
|------|------|------|
| overflow | `calc_fall_risk` 손목·발목 vs bed_bbox | `limb_overflow_*` CSV 컬럼 |
| 통합 score | 없음 (로그만) | `risk_score` 0~10 + `risk_level` |
| 난간 | pose idx 0,1 (**무의미**) | preset `rail_lines` + H |

#### S1.8.5 저장·API

| 항목 | 현재 | 목표 |
|------|------|------|
| 저장 | 없음 | CSV + JSON + MP4 |
| API | `/status`, `/video`, `/health` | + `risk_*`, calibration 메타 |

### S1.9 성공 기준 (초기 검증 단계)

| ID | 기준 | 측정 방법 |
|----|------|-----------|
| V1 | MP4 1편 → CSV **프레임 수 = 영상 프레임** (±드롭 허용) | 배치 스크립트 |
| V2 | 동일 MP4 2회 실행 시 `risk_score` **상관 > 0.95** | 재현성 |
| V3 | 라벨 `unsafe_exit` 구간에서 `risk_score` **평균 > normal** | 구간 통계 |
| V4 | 정상 하차 영상에서 **HIGH 비율 < X%** | 오탐률 (X는 현장 합의) |
| V5 | `pipeline_version` + preset으로 **6개월 후 재현** | JSON 메타 |

### S1.10 S1에서 확정할 체크리스트 (모델 선점)

```text
[ ] yolo11n-seg.pt + class 59 + conf 0.2
[ ] yolo11m-pose.pt + conf 0.5
[ ] Keras: 6-class vs 12-class 용도 분리 (수집 기본 = 6-class 권장)
[ ] 입력 34-dim, 출력 class 순서 문서 고정
[ ] resize width=800, CPU/GPU 정책 (현재 server: CPU 강제)
[ ] Ultralytics / TF / Keras 버전 requirements 고정
```

---

<a id="s2-배포운영-전제-병원"></a>
## S2. 배포·운영 전제 (병원)

### S2.1 현장 가정 (고정)

| 가정 | 상세 |
|------|------|
| 카메라 위치 | **머리맡** — 환자 머리 방향 기준 설치 (방마다 가구는 다를 수 있음) |
| 설치 종류 | 실무상 **약 10종** (높이·거리·각도·줌 조합) → `preset_01` … `preset_10` |
| 병실 | **다른 room_id**, 같은 preset이면 **기하 프로파일 공유** |
| 센서 | **영상만** — 가속도계 없음 |
| 침대 | 병실마다 이동 가능 → **H 갱신** 필요 (S5) |
| 난간 | 유무·올림은 **방 메타** + 프리셋 기하 (S7) |

### S2.2 “10 프리셋 × N 병실” 모델

```text
                    ┌──────────────┐
   room_101 ───────►│ preset_03    │
   room_102 ───────►│ (H 템플릿)   │◄───── intrinsic (카메라 모델 공통, 선택)
   room_205 ───────►│ rail_lines   │
                    │ thresholds   │
                    └──────────────┘
```

**병실 등록 최소 필드**

```json
{
  "room_id": "room_402",
  "rtsp_url": "rtsp://192.168.0.157:8554/stream",
  "preset_id": "preset_03",
  "guardrail": true,
  "bed_movable": true,
  "notes": "창측 조명 강함"
}
```

| 이벤트 | 조치 |
|--------|------|
| 신규 병실, 기존과 **동일 설치** | `preset_id`만 할당, H 템플릿 로드 |
| 침대만 옮김 | `preset_id` 유지, **H 갱신** (S5.3) |
| 카메라 재설치 | `preset_id` 재선택 또는 신규 preset 등록 + **4점 보정** |
| 카메라 모델 교체 | `intrinsic_id` 갱신 (L1, 선택) |

### S2.3 프리셋 JSON 스키마 (전체)

```json
{
  "preset_id": "preset_03",
  "version": "1.0.0",
  "description": "머리맡 높이 2.1m, 하향 25deg, 800px resize 기준",
  "camera": {
    "intrinsic_id": "cam_hik_xxx_v1",
    "nominal_height_m": 2.1,
    "nominal_tilt_deg": 25
  },
  "homography": {
    "bed_corners_image": [[120,80],[680,90],[700,400],[100,380]],
    "bed_corners_norm": [[0,0],[1,0],[1,1],[0,1]],
    "H_img2bed": "3x3 matrix or path to .npy",
    "update_policy": "seg_ema_conservative"
  },
  "bed_size_m": { "length": 2.0, "width": 0.9 },
  "zones": {
    "split_lcr": true,
    "long_axis": "x"
  },
  "rail_geometry": {
    "enabled": true,
    "lines_bed_norm": [
      { "name": "rail_head_side", "p0": [0.0, 0.12], "p1": [1.0, 0.12] },
      { "name": "rail_foot_side", "p0": [0.0, 0.88], "p1": [1.0, 0.88] }
    ]
  },
  "risk_thresholds": {
    "overflow_low": 0.05,
    "overflow_med": 0.15,
    "overflow_high": 0.25,
    "out_bed_ratio_high": 0.5,
    "speed_high_body_norm_per_s": 2.0,
    "edge_stay_warn_s": 3.0
  },
  "inference": {
    "resize_width": 800,
    "seg_conf": 0.2,
    "pose_conf": 0.5,
    "seg_hz": 2.0
  }
}
```

**`bed_corners_norm` 의미:**  
호모그래피로 펼친 뒤 침대 상면을 **0~1 정사각(또는 직사각) 좌표**로 쓴다. 머리맡이 어느 축인지는 preset마다 `long_axis`로 고정.

### S2.4 중앙 서버 vs 엣지

#### S2.4.1 부하 원인

IMU가 없다고 서버 부하가 **줄어들지는 않음**. 부하는:

```text
부하 ≈ N_streams × ( T_seg + T_pose + T_keras + T_draw + T_encode )
```

| 구간 | CPU/GPU | 비고 |
|------|---------|------|
| YOLO-seg | 높음 | **저주기(1~3Hz)** 로 완화 |
| YOLO-pose | 높음 | 사람 1명 가정 시 1 detection |
| Keras | 중간 | 34→6, 작은 MLP |
| MJPEG encode | 중간 | 뷰어용만 |

#### S2.4.2 권장 토폴로지

```text
[병실 1..N]
   RTSP
      │
      ├─► (A) 엣지: seg+pose+keras → feature stream (경량 JSON/GRPC)
      │         서버: 시계열 + rule + 저장
      │
      └─► (B) 중앙 집중: 전부 서버 (소규모 N, GPU 있을 때)
```

| N (병실) | 권장 |
|----------|------|
| 1~4 | (B) 가능, seg 2Hz |
| 8+ | (A) 엣지 또는 **전용 GPU 서버** |
| 데이터 수집만 | (B) 배치 MP4, 실시간 불필요 |

#### S2.4.3 스트림별 상태 (서버 메모리)

```python
# 개념적 상태 객체 (스트림당 1개)
StreamState = {
    "room_id": str,
    "preset_id": str,
    "H_stable": np.ndarray,      # 3x3
    "h_version": int,
    "last_bed_mask": ...,
    "ring_buffer": deque,        # 최근 5초 feature
    "last_pose_class": str,
    "edge_stay_since": float | None,
}
```

모델 가중치(`YOLO`, `keras`)는 **전역 1회 로드**, `StreamState`만 분리.

### S2.5 운영 시나리오表

| 시나리오 | preset | H | rail | 비고 |
|----------|--------|---|------|------|
| 신규 설치 표준 | 고정 | 4점 1회 | preset | |
| 침대 청소 후 재배치 | 동일 | **자동/반자동** | 동일 | `bed_moved` 이벤트 |
| 난간 내려놓음 | 동일 | 동일 | `rail_up: false` 메타 | 점수에서 rail 항 제외 |
| 야간 조명 | 동일 | 동일 | 동일 | seg conf 튜닝, 데이터 수집 |

---

<a id="s3-처리-파이프라인"></a>
## S3. 처리 파이프라인

### S3.1 단계 정의 (A~G)

| 단계 | 이름 | 입력 | 출력 | 주기 |
|------|------|------|------|------|
| **A** | Bed Seg | frame | `bed_masks[]`, `bed_bbox` | 1~30 Hz (모드별) |
| **B** | Homography | masks, preset | `H`, `bed_norm` 좌표 | A와 동기 또는 저주기 |
| **C** | Pose | frame | `kpts_xy`, `kpts_conf`, `person_bbox` | 매 프레임 |
| **D** | Pose Class | 34-dim | `pose_class`, `pose_conf` | 매 프레임 (사람 있을 때) |
| **E** | Frame Features | A~D | §4 컬럼 | 매 프레임 |
| **F** | Temporal | E history | edge_duration, max_3s, … | 링버퍼 갱신 |
| **G** | Rule Score | F + E | `risk_score`, `risk_level` | 매 프레임 |

### S3.2 상세 흐름도

```text
                    ┌─────────────┐
                    │ RTSP / MP4  │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │ Preprocess  │ resize W=800, t, frame_idx
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │ A: YOLO-seg│  │ C: YOLO-   │  │ B: H update│
    │ class 59   │  │ pose       │  │ (from A +  │
    └─────┬──────┘  └─────┬──────┘  │  preset)   │
          │               │         └─────┬──────┘
          └───────┬───────┘               │
                  ▼                         │
           ┌────────────┐                   │
           │ D: Keras   │◄── 34-dim xy ────┤
           │ 6-class    │                   │
           └─────┬──────┘                   │
                 ▼                          │
           ┌────────────┐◄──────────────────┘
           │ E: Features│ bed_norm, out_ratio, v, a_like
           └─────┬──────┘
                 ▼
           ┌────────────┐
           │ F: Buffer  │ 3~5 s deque
           └─────┬──────┘
                 ▼
           ┌────────────┐
           │ G: Rule    │ risk_score (NOT label)
           └─────┬──────┘
                 ▼
        ┌────────┴────────┐
        ▼                 ▼
   CSV row          JSON episode
   (per frame)      (per video)
```

### S3.3 단계별 구현 메모 (현재 코드 매핑)

#### A. Bed Seg

```python
seg_res = yolo_seg_model.predict(frame, classes=[59], conf=0.2, verbose=False)
bed_masks = seg_res[0].masks.data if seg_res[0].masks is not None else None
```

- 마스크 없음 → `person_in_bed=False`, out_ratio=0 또는 NaN, H 갱신 스킵.

#### B. Homography

- `preset.H_img2bed` 로드 → 필요 시 S5.3으로 `H_stable` 갱신.
- 모든 bed_norm feature는 **`cv2.perspectiveTransform`** 또는 동등 연산.

#### C. Pose

```python
pose_res = yolo_pose_model.predict(frame, conf=0.5, verbose=False)
# 첫 번째 person만 사용 (현재와 동일)
```

- 다인원 방: v2에서 **침대 위 bbox 최대** 선택 규칙 추가.

#### D. Keras

```python
kpts_data = pose_res[0].keypoints[0].xy.cpu().numpy().flatten()  # 34
pred = keras_clf.predict(kpts_input, verbose=0)
pose_class = CLASS_NAMES[argmax(pred)]
```

#### E~G

- 신규 모듈 `features.py`, `temporal.py`, `risk_rules.py` 분리 권장 (공통 import).

### S3.4 실행 모드

| 모드 | 진입점 | A 주기 | 출력 |
|------|--------|--------|------|
| **live** | `server.py` `analysis_loop` | 2 Hz 권장 | MJPEG, `/status` |
| **collect** | `extract_features.py` (신규) | 매 프레임 또는 10 Hz | CSV+JSON+MP4 |
| **replay** | 동일 + MP4 입력 | 매 프레임 | CSV only |
| **debug** | `run_pose.py` | 매 프레임 | OpenCV window |

### S3.5 실패·결측 처리

| 상황 | 처리 |
|------|------|
| RTSP 끊김 | 5초 대기 후 재연결 (현재 `server.py`와 동일) |
| 사람 미검출 | pose_class=`None`, motion feature **hold 또는 NaN** |
| seg 실패 | 이전 `H_stable` 유지, `bed_moved` 미발화 |
| kpt conf < 0.3 | 해당 점 제외, 중심은 bbox fallback |
| 프레임 드롭 | `timestamp`는 **frame_idx/fps** 기준 유지 |

### S3.6 성능 튜닝 파라미터 (preset.inference)

| 파라미터 | 기본 | 효과 |
|----------|------|------|
| `resize_width` | 800 | ↓ 시 속도↑, 정밀도↓ |
| `seg_hz` | 2.0 | ↓ 시 CPU↓, H·mask 지연↑ |
| `pose_conf` | 0.5 | ↑ 시 미검출↑ |
| `seg_conf` | 0.2 | ↓ 시 침대 누락↑ |

---

<a id="s4-feature-정의-프레임--시계열"></a>
## S4. Feature 정의 (프레임 + 시계열)

### S4.1 좌표계 (반드시 CSV에 명시)

| 필드 | 의미 | 단위 |
|------|------|------|
| `coord_space` | `image` / `bed_norm` / `body_norm` | - |
| `u`, `v` | 리사이즈 이미지 픽셀 | px |
| `x_bed`, `y_bed` | H 적용 후 | 0~1 (또는 m if scaled) |
| `scale_body` | 어깨–골반 거리 또는 bbox h | px, body_norm 나눗셈용 |

### S4.2 침대·이탈 feature

#### `person_in_bed` (이진, 현행)

```text
cx, cy = bbox center
any mask[cy, cx] > 0.5 → YES
```

#### `person_out_bed_ratio` (핵심, 목표)

```text
유효 keypoint 집합 K (conf >= 0.3)
out_bed_ratio = |{ k in K : mask(k) <= 0.5 }| / |K|
```

- bbox 픽셀 샘플링 방식도 가능 (느리면 kpt만).
- **낙상 rule의 주축**이나 **단독으로 fall 확정 금지**.

#### `bed_edge_distance`

```text
x_bed, y_bed = H @ center
edge_distance = min(x_bed, 1-x_bed, y_bed, 1-y_bed)  # 침대 경계까지
```

#### `edge_zone`

```text
x_bed < 1/3 → L
x_bed > 2/3 → R
else → C
```

(`run_pose.py` `draw_bed_zones`와 동일 개념, bed_norm 기준으로 통일)

### S4.3 Motion-like feature

**추적점 `c_t` 우선순위**

1. `(L_hip + R_hip) / 2` in bed_norm  
2. bbox center in bed_norm  
3. 이전 프레임 hold (최대 0.5s)

**계산**

```text
Δt = 1 / fps
v_t = (c_t - c_{t-1}) / Δt
a_t = (v_t - v_{t-1}) / Δt
```

**스무딩 (필수 권장)**

```text
v_smooth = EMA(v_t, α=0.3)
a_like   = |v_smooth - v_smooth_prev| / Δt
```

| CSV 컬럼 | 설명 |
|----------|------|
| `center_vx`, `center_vy` | v_smooth 성분 |
| `center_speed` | ‖v_smooth‖ |
| `center_accel_like` | ‖a_like‖ |
| `vertical_drop_rate` | Δ y_bed / Δt (머리맡 카메라에서 “아래” 근사) |

**body_norm:** `v_body = v / scale_body` → 설치 간 비교 용이.

### S4.4 자세 feature

| 컬럼 | 설명 |
|------|------|
| `pose_class` | 예: `sitting_edge` |
| `pose_conf` | softmax max |
| `pose_class_id` | 0..5 |
| `lying_flag` | class ∈ {front_lying, prone_back, side_near, side_far} |
| `sitting_edge_flag` | class == sitting_edge |

**전이 (F 단계)**

```text
pose_transition = (class_t != class_{t-1})
pose_sequence_5s = "sitting_edge,lying,..."  # 검수용 문자열
```

### S4.5 사지 overflow (기존 rule 이식)

`pose/run_pose.py` `calc_fall_risk` 로직:

```text
bed_width = x_max - x_min  # bed_bbox in image (또는 bed_norm 1.0)
for wrist, ankle in RISK_KEYPOINTS:
  overflow = lateral distance outside [x_min, x_max] / bed_width
max_overflow → limb_overflow_max
```

bed_norm 사용 시: 경계를 0,1로 두고 동일 비율 계산.

### S4.6 시계열 창 feature (F 단계)

버퍼 길이 `T_buf = 5.0` 초 예시.

| 컬럼 | 정의 |
|------|------|
| `edge_stay_duration` | `edge_zone != C` 연속 시간 (s) |
| `out_bed_ratio_max_3s` | max(out_ratio) over [t-3, t] |
| `speed_max_1s` | max(center_speed) over [t-1, t] |
| `stationary_duration` | speed < ε 연속 (s) |
| `lying_outside_duration` | lying_flag & out_ratio > θ 연속 |

### S4.7 결측·품질 컬럼

| 컬럼 | 용도 |
|------|------|
| `person_detected` | bool |
| `seg_bed_ok` | bool |
| `kpt_mean_conf` | 품질 |
| `h_version` | H 변경 추적 |

---

<a id="s5-침대-이동--호모그래피-운영"></a>
## S5. 침대 이동 · 호모그래피 운영

### S5.1 3계층 캘리브 (재정리)

| 계층 | 내용 | 빈도 | 필수 |
|------|------|------|------|
| **L1** | Intrinsic + distortion | 카메라 모델·줌 변경 시 | 선택 |
| **L2** | Homography H (침대 평면) | preset + 병실 미세 | **필수** |
| **L3** | Metric scale (침대 길이 m) | preset | 선택 |

**높이(3D):** L2만으로 **측정 불가** — §4.3은 평면·상대 motion.

### S5.2 L2 수동 4점 절차 (표준)

1. 설치 직후 **빈 침대** 또는 이불 정리 상태에서 촬영.  
2. 운영 UI에서 침대 **상면 4모서리** 클릭 (머리좌→머리우→발우→발좌 순서 고정).  
3. `cv2.getPerspectiveTransform(src4, dst4_norm)` → `H_img2bed`, `H_bed2img` 저장.  
4. `preset_k` JSON에 `bed_corners_image` + `H` 저장.  
5. 검증: 침대 롱 edge가 **bed_norm x축**과 평행한지 시각 확인.

### S5.3 침대 이동 시 자동 H 갱신

```text
every T_seg (e.g. 0.5s):
  masks → union → largest contour
  → approxPolyDP → 4 corners (실패 시 skip)
  → H_candidate = getPerspectiveTransform(corners, dst4_norm)

  shift = corner_mean_distance(H_candidate, H_stable)
  if shift > τ_move (e.g. 15 px @800w):
      H_stable = (1-α)*H_stable + α*H_candidate   # 또는 corner만 EMA
      h_version += 1
      log bed_moved event
```

| 정책 | α | τ_move | 용도 |
|------|---|--------|------|
| conservative | 0.1 | 20px | 실시간 알림 |
| balanced | 0.2 | 15px | 수집 기본 |
| manual_only | - | - | seg 불안정 병실 |

**반자동:** `shift > τ_warn` → UI에 “침대 위치 변경 감지, 확인?” 표시.

### S5.4 세그만으로 H 할 때 실패 케이스

| 케이스 | 대응 |
|--------|------|
| 이불이 침대 밖으로 늘어짐 | mask 팽창 → H 틀어짐 → τ_move + 수동 확인 |
| 환자가 침대 가득 | contour 4점 불안정 → **H 갱신 동결** |
| 침대 half visible | 갱신 스킵, 이전 H 유지 |

### S5.5 intrinsic (L1) — 언제 할지

- wide FOV, 가장자리 왜곡 심할 때 **체스판 1회** → `undistort` 후 seg/pose.  
- 동일 IP 카메라 **10프리셋**은 **intrinsic 공유** 가능.

---

<a id="s6-스코어링-vs-라벨링"></a>
## S6. 스코어링 vs 라벨링

### S6.1 역할 분리 (데이터셋 수집의 핵심)

```text
label       = 사람이 정의한 실제 상태 (ground truth)
risk_score  = rule/모델이 계산한 위험도 (system output, 재현용)
```

| 단계 | label | risk_score |
|------|-------|------------|
| 수집 직후 | 비우거나 별도 툴 | **자동 채움** |
| 검수 | **수정·확정** | 변경하지 않음 (당시 기록 유지) |
| 평가 | 정답 | 예측과 비교 |

**금지:** `risk_score` 높은 구간을 자동으로 `label=fall` 처리.

### S6.2 라벨 taxonomy

| label | 의미 | 예시 |
|-------|------|------|
| `normal` | 침대 안 안정 | 누워 휴식 |
| `movement` | 단순 체위 변경 | 뒤척임 |
| `edge_observe` | 가장자리 접근·체류 | 가장자리 앉음 |
| `exit_intent` | 하차 의도 | 상체 일어남 |
| `unsafe_exit` | 위험 이탈 | 빠르게 몸 밖 |
| `fall_suspected` | 낙상 의심 | 급하강+밖 |
| `fall` | 낙상 확인 | 바닥 누움 지속 |

JSON `event_segments`에 **start_time, end_time, label** 저장.  
CSV `label`은 구간 join 또는 라벨링 후 merge.

### S6.3 risk_score 구성 (0~10)

**기본식**

```text
risk_score = clamp( base_pose + S_out + S_speed + S_edge + S_seq + S_limb, 0, 10 )
```

#### base_pose

| pose_class | base |
|------------|------|
| sitting_center | 1.0 |
| sitting_edge | 2.5 |
| side_*, front_lying, prone | 2.0 |
| None / low conf | 0 |

#### S_out (이탈)

```text
if out_bed_ratio < 0.1:  S_out = 0
elif out_bed_ratio < 0.3: S_out = 2 * out_bed_ratio
else: S_out = min(4, 5 * out_bed_ratio)
```

#### S_speed (누움 + 급이동)

```text
if lying_flag and center_speed > θ_v_high:
    S_speed = 3 + min(2, (speed - θ_v_high) / θ_v_high)
else:
    S_speed = 0
```

#### S_edge

```text
if edge_stay_duration > T_edge (3s): S_edge = 2
```

#### S_seq (전이)

```text
if prev in {sitting_edge} and now lying and out_bed_ratio increasing:
    S_seq = 3
```

#### S_limb

```text
S_limb = map(max_overflow)  # 0,1,2,3 for SAFE,LOW,MED,HIGH
```

#### risk_level

| risk_score | level |
|------------|-------|
| [0, 2) | SAFE |
| [2, 4) | LOW |
| [4, 7) | MED |
| [7, 10] | HIGH |

임계는 **수집 데이터로 튜닝** — preset `risk_thresholds`에 저장.

### S6.4 일상 굴림 억제 규칙

**다음 모두 만족 시** `S_speed`, `S_out` **캡 (예: max +1)**

```text
- out_bed_ratio < 0.15
- edge_zone == C
- speed < θ_roll (중간 이하)
- stationary_duration < 1s at end of window  (계속 움직이는 뒤척임)
```

### S6.5 “누움 + 밖 = 최고” 조건 (완화된 HIGH)

**HIGH 후보** (단독 아님, AND):

```text
lying_flag
AND out_bed_ratio > θ_high (0.5)
AND ( speed_max_1s > θ_v OR stationary_duration > T_stop after speed spike )
```

정상 바닥 누움 데이터로 `θ_high`, `T_stop` 조정.

### S6.6 검증 지표

| 지표 | 단위 | 설명 |
|------|------|------|
| 구간 precision/recall | per label | rule level vs label |
| lead time | 초 | `unsafe_exit` label 시작 − score가 MED 넘은 시각 |
| false HIGH rate | %/시간 | normal 영상 |
| score-label correlation | ρ | 참고만 |

---

<a id="s7-난간-설계"></a>
## S7. 난간 설계

### S7.1 현재 문제

```python
RAIL_KEYPOINTS = { 'Rail_0': 0, 'Rail_1': 1 }  # YOLO pose 머리 — 난간 아님
```

### S7.2 목표: 프리셋 기하 + H

```text
rail line in bed_norm:  p0 —— p1
wrist bed_norm: (x_w, y_w)
signed_dist = point_to_line_distance(wrist, rail)
rail_overflow = max(0, signed_dist - margin) / bed_width_norm
```

| 컬럼 | 설명 |
|------|------|
| `rail_enabled` | room.guardrail |
| `rail_overflow_left` | 좌측 난간 대비 |
| `wrist_above_rail` | y_w < y_rail (축 정의에 따름) |

침대 이동 시 **H만 갱신**하면 rail image 좌표도 자동 추적 (`H_bed2img`).

### S7.3 난간 없음 / 내려놓음

```json
"guardrail": false
```
→ `S_limb` rail 항 제외, `rail_*` 컬럼 NaN.

### S7.4 v2 (선택)

- 난간 전용 YOLO detector  
- 영상에서 rail line RANSAC

---

<a id="s8-데이터셋-산출물"></a>
## S8. 데이터셋 산출물

### S8.1 디렉터리

```text
dataset/
  presets/
    preset_03.json
  rooms/
    room_402.json
  episodes/
    episode_001/
      episode_001.mp4
      episode_001.csv
      episode_001.json
      frames/              # optional
        000001.jpg
  reports/
    eval_2025-xx-xx.md
```

### S8.2 CSV 스키마 (전체)

```csv
timestamp,frame_idx,fps,preset_id,h_version,room_id,
person_detected,seg_bed_ok,
u,v,x_bed,y_bed,scale_body,
person_in_bed,person_out_bed_ratio,edge_zone,bed_edge_distance,
pose_class,pose_class_id,pose_conf,lying_flag,sitting_edge_flag,
center_vx,center_vy,center_speed,center_accel_like,vertical_drop_rate,
edge_stay_duration,out_bed_ratio_max_3s,speed_max_1s,stationary_duration,
limb_overflow_max,rail_overflow_left,
risk_score,risk_level,
label
```

### S8.3 JSON episode (전체)

```json
{
  "video_id": "episode_001",
  "source": "mp4",
  "room_id": "room_402",
  "preset_id": "preset_03",
  "fps": 30,
  "resolution": [800, 450],
  "duration_s": 120.5,
  "guardrail": true,
  "scenario": "simulated_bed_exit",
  "collection": {
    "pipeline_version": "0.2.0",
    "collected_at": "2026-05-18T12:00:00Z",
    "yolo_seg": "yolo11n-seg.pt",
    "yolo_pose": "yolo11m-pose.pt",
    "keras_model": "my_model_six_check.keras",
    "class_names": ["front_lying", "prone_back", "side_near", "side_far", "sitting_center", "sitting_edge"]
  },
  "event_segments": [],
  "bed_moved_events": [],
  "scores_summary": {
    "risk_max": 0,
    "risk_mean": 0,
    "high_duration_s": 0
  }
}
```

### S8.4 수집 카탈로그 (필수 시나리오)

| ID | 시나리오 | label 예 |
|----|----------|----------|
| C01 | 정상 누움·뒤척임 | normal, movement |
| C02 | 가장자리 앉기 | edge_observe |
| C03 | 정상 하차 | exit_intent → normal |
| C04 | 낙상·의심 | fall_suspected, fall |
| C05 | 이불·가림 | normal + 품질 메타 |
| C06 | 보호자 통과 | movement |
| C07 | 침대 이동 후 | bed_moved_events |
| C08 | 난간 없음 | guardrail false |

---

<a id="s9-단계별-로드맵"></a>
## S9. 단계별 로드맵

| Phase | 내용 | 산출 |
|-------|------|------|
| **0** | 모델 선점, requirements, class 순서 | S1 체크리스트 |
| **1** | MP4→CSV, 픽셀 정규화, risk_rule 기록 | `extract_features.py` |
| **2** | preset JSON, H 4점, out_bed_ratio | preset 로더 |
| **3** | temporal buffer, §6 risk_score | `risk_rules.py` |
| **4** | 다스트림, seg_hz, 엣지 PoC | 부하 리포트 |
| **5** | TCN/GRU 학습 | 모델 + eval |

---

<a id="s10-api실시간"></a>
## S10. API·실시간

### S10.1 현재

| Method | Path | 필드 |
|--------|------|------|
| GET | `/status` | in_bed, pose, pose_conf, timestamp |
| GET | `/health` | server, analysis_running |
| GET | `/video` | MJPEG |

### S10.2 확장 (수집·운영)

| Method | Path | 용도 |
|--------|------|------|
| GET | `/status` | + risk_score, risk_level, out_bed_ratio, preset_id |
| GET | `/calibration` | H_version, bed_moved_flag |
| POST | `/calibration/preset` | body: preset_id |
| POST | `/calibration/corners` | body: 4 points |
| POST | `/collection/start` | episode_id, room_id |

---

<a id="s11-리스크한계"></a>
## S11. 리스크·한계

| ID | 리스크 | 영향 | 완화 |
|----|--------|------|------|
| R1 | 단안 깊이 없음 | 높이·굴림 혼동 | bed_norm + 시계열 + 자세 |
| R2 | seg 떨림 | H·out_ratio 노이즈 | EMA, seg 2Hz |
| R3 | lying+밖 오탐 | HIGH 과다 | C04 데이터, AND 조건 |
| R4 | 앉음=안전 가정 | 가장자리 낙상 누락 | sitting_edge 가중 |
| R5 | 서버 집중 | 지연·다운 | 엣지, seg 저주기 |
| R6 | 환경 변경 | rule drift | preset_id, h_version |
| R7 | 다인원 | 잘못된 person | 최대 bbox 규칙 v2 |

---

<a id="s12-용어집"></a>
## S12. 용어집

| 용어 | 정의 |
|------|------|
| **H** | 3×3 homography, image ↔ bed_norm |
| **preset** | 설치 10종 중 1개 기하·임계 프로파일 |
| **bed_norm** | 침대 평면 정규화 좌표 (보통 0~1) |
| **body_norm** | 신체 스케일로 나눈 상대 좌표/속도 |
| **accel_like** | 2차 차분 크기, 물리 가속도 아님 |
| **overflow** | 손·발이 침대 lateral 경계 밖 비율 |
| **episode** | MP4+CSV+JSON 1세트 |

---

<a id="s13-참고-현재-코드-앵커"></a>
## S13. 참고: 현재 코드 앵커

| 파일 | 함수/역할 |
|------|-----------|
| `pose-sixclass/server.py` | `analysis_loop`, `is_person_in_bed`, MJPEG |
| `pose/run_pose.py` | `calc_fall_risk`, `get_bed_bbox`, `draw_bed_zones` |
| `pose-sixclass/run_pose.py` | 로컬 디버그, Rail placeholder |
| `pose-webviewer/server.py` | 12-class Keras |

**버그 메모:** `pose/run_pose.py` `in_bed_status = "YES"` 고정 → `is_person_in_bed`와 **공통 모듈화** 필요.

---

## Changelog

| 버전 | 날짜 | 변경 |
|------|------|------|
| 0.1 | - | 초안 |
| 0.2 | 2026-05-18 | S1~S13 순차 상세 확장 |

*수정 시 `collection.pipeline_version` 갱신.*
