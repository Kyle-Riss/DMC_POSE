# Bed seg v1 — RTSP 실시간 카메라 전용

실시간 탐지(RTSP `.161`)에 맞춘 파인튜닝 파이프라인입니다.

## 왜 RTSP 중심인가

- 배포 추론 = **RTSP 라이브 프레임**
- 학습도 **같은 카메라·같은 해상도(800px)** 에서 뽑은 프레임 + **수동 polygon** 이 가장 정확

## v0 vs v1

| | v0 (완료) | v1 (진행) |
|---|-----------|-----------|
| 데이터 | auto heuristic 340장 | **RTSP 수동 라벨 50~70장** |
| 목적 | 파이프라인 검증 | **실시간 bbox/zone 품질** |
| 런타임 | + ROI clip | 동일 |

## 워크플로

```bash
conda activate pose-cuda
cd /home/dmc/pose-sixclass/bed_seg
bash run_rtsp_v1.sh
```

또는 단계별:

### 1. RTSP 프레임 캡처

```bash
python capture_rtsp_frames.py --count 50 --interval-sec 2.0
# → rtsp_raw/*.jpg (800px)
```

### 2. 수동 polygon 라벨 (DISPLAY 필요)

```bash
python /home/dmc/labeling/label_bed_polygon.py \
  --images /home/dmc/pose-sixclass/bed_seg/rtsp_raw \
  --labels /home/dmc/pose-sixclass/bed_seg/manual_labels
```

매트리스+프레임만 꼭짓점 클릭 → Enter 저장

### 3. dataset 빌드

```bash
python prepare_rtsp_dataset.py
# → dataset_v1/ (manual labels only)
```

`labeling/labels/` 에 vlcsnap 수동 라벨이 있으면 자동 포함

### 4. 학습 + 배포

```bash
python train_bed_seg.py
# → runs/bed_seg_v1/weights/best.pt
# → ../yolo11n-bed-seg.pt 복사
```

### 5. 서버

```bash
bash /home/dmc/pose-sixclass/run_server.sh
```

## 런타임 (server.py)

```text
RTSP → yolo11n-bed-seg (v1) → mask/bbox
     → bed_roi.json clip
     → in_bed / L·C·R
```

## 파일

| 경로 | 설명 |
|------|------|
| `rtsp_raw/` | RTSP 캡처 이미지 |
| `manual_labels/` | YOLO-seg polygon txt |
| `dataset_v1/` | v1 학습셋 |
| `../yolo11n-bed-seg.pt` | 배포 가중치 |
