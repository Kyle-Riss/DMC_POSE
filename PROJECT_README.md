# 스마트 침대 모니터링 시스템 (Smart Bed Monitoring System)

## 프로젝트 개요

병원/요양시설 환경에서 **침대 위 환자의 자세를 실시간으로 인식**하고, **얼굴 재식별(Face Re-identification)** 을 통해 환자를 구분하는 AI 기반 모니터링 시스템입니다.

RTSP/HTTP 카메라 스트림을 입력으로 받아 다음 정보를 실시간 출력합니다:

```
In Bed: YES | Pose: p01
In Bed: YES | Pose: p05
In Bed: NO  | Pose: None
```

---

## 핵심 기능

| 기능 | 설명 | 사용 모델 |
|------|------|-----------|
| **침대 영역 감지** | YOLO Segmentation으로 침대 마스크를 생성하여 환자가 침대 위에 있는지 판별 | YOLOv8/YOLO11 Segmentation |
| **자세 추정 (Pose Estimation)** | 사람의 17개 관절 좌표를 추출하여 자세 분류 | YOLO11m-Pose |
| **자세 분류** | 추출된 관절 좌표를 입력으로 수면 자세(p01~p12)를 분류 | Keras Dense NN |
| **얼굴 재식별** | SCRFD(얼굴 검출) + ArcFace(얼굴 임베딩)로 등록된 환자를 식별 | SCRFD + ArcFace (ONNX) |
| **마스크 감지** | 환자의 마스크 착용 여부 감지 | ONNX 분류 모델 |

---

## 프로젝트 구조

### 루트 디렉토리 (`/pose/`) — 메인 파이프라인

| 파일 | 역할 |
|------|------|
| `data.py` | YOLO11m-Pose로 `data/` 폴더의 이미지에서 17개 관절 좌표(34차원)를 추출하여 `pose_dataset.csv` 생성 |
| `train.py` | CSV 데이터로 Keras Dense 신경망 학습 → `my_model.keras` 저장 |
| `run_pose.py` | **실시간 추론 메인 스크립트** — RTSP 스트림에서 침대 감지 + 자세 분류를 동시 수행 |
| `pose_dataset.csv` | 추출된 자세 데이터셋 (라벨 + 34개 관절 좌표) |
| `my_model.keras` | 학습된 자세 분류 모델 |
| `yolo11m-pose.pt` | YOLO11 Medium Pose 모델 가중치 |
| `yolo11n-seg.pt` | YOLO11 Nano Segmentation 모델 가중치 (침대 감지용) |
| `data/` | 자세별 학습 이미지 (p01~P18, 18개 클래스) |

### `bed/face-reidentification (1)/` — 얼굴 재식별 + 침대 감지 통합

| 파일 | 역할 |
|------|------|
| `main.py` | HTTP 이미지 소스 기반 얼굴 감지·인식 (SCRFD + ArcFace) |
| `bed.py` | RTSP 스트림 기반 얼굴 인식 + YOLO Segmentation 침대 감지 통합 |
| `r-bed.py` | HTTP 이미지 소스 기반 얼굴 인식 + 침대 Segmentation 통합 |
| `rgb.py` | RGB 채널별 분리 시각화 유틸리티 |
| `rtmpose.py` | RTMPose (ONNX) 기반 133개 키포인트 전신 자세 추정 (HOG 사람 검출 포함) |
| `models/` | SCRFD(얼굴 검출), ArcFace(얼굴 임베딩) 모델 래퍼 |
| `weights/` | 각종 ONNX/PT 가중치 파일 |
| `faces/` | 등록된 환자 얼굴 이미지 (파일명 = 환자명) |

### `bed/face-re/face-reidentification/` — PyQt GUI 버전

| 파일 | 역할 |
|------|------|
| `pyqt_code_end.py` | PyQt5 GUI로 얼굴 인식 + 마스크 감지 + 체온 표시를 구현한 최종 버전 |
| `npz_end.py` | NPZ 파일 기반 얼굴 임베딩 관리 |
| `npz_files/` | 사전 등록된 환자 얼굴 임베딩(.npz) |

### `bed/last_pose/` — MediaPipe 기반 자세 분류 (이전 버전)

| 파일 | 역할 |
|------|------|
| `data_csv.py` | MediaPipe Pose(33 랜드마크)로 관절 좌표를 CSV로 추출 |
| `model.py` | Keras Dense 모델 학습 스크립트 |
| `pose.py` | 학습된 모델로 실시간 자세 분류 추론 |

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| **자세 추정** | YOLO11m-Pose (현재) / MediaPipe Pose (이전) / RTMPose (실험) |
| **객체 분할** | YOLO11n-seg / YOLOv8-seg (침대 영역 감지) |
| **얼굴 검출** | SCRFD (ONNX) |
| **얼굴 인식** | ArcFace (ONNX, w600k_r50) |
| **자세 분류** | Keras/TensorFlow Dense Neural Network |
| **GUI** | PyQt5 |
| **영상 입력** | RTSP 스트림 / HTTP 이미지 폴링 / 웹캠 |
| **추론 환경** | CUDA GPU / CPU (환경에 따라 자동 전환) |

---

## 파이프라인 흐름

```
카메라(RTSP/HTTP)
       │
       ▼
┌──────────────┐     ┌──────────────────┐
│  YOLO11-Seg  │     │  YOLO11m-Pose    │
│  (침대 감지)  │     │  (관절 좌표 추출) │
└──────┬───────┘     └────────┬─────────┘
       │                      │
       ▼                      ▼
  침대 마스크            17개 관절 좌표 (34차원)
       │                      │
       ▼                      ▼
  환자가 침대 위?      ┌──────────────────┐
  (Yes/No)            │  Keras 분류 모델  │
                      │  (자세 클래스)     │
                      └────────┬─────────┘
                               │
                               ▼
                        자세 라벨 (p01~p12)
```

---

## 자세 클래스

`data/` 디렉토리의 폴더명이 곧 자세 클래스 라벨입니다:

- `p01` ~ `p12`: 기본 수면/침대 자세 12종
- `P13` ~ `P18`: 추가 자세 6종 (대문자로 구분)

---

## 실행 방법

### 1. 자세 데이터 추출

```bash
python data.py
```

`data/` 폴더의 이미지에서 관절 좌표를 추출하여 `pose_dataset.csv`를 생성합니다.

### 2. 모델 학습

```bash
python train.py -i pose_dataset.csv -o my_model.keras
```

### 3. 실시간 모니터링 실행

```bash
python run_pose.py
```

RTSP 스트림(`rtsp://192.168.0.161:8554/stream`)에서 실시간으로 침대 감지 + 자세 분류를 수행합니다.

---

## 프로젝트 발전 과정

1. **MediaPipe 기반** (`bed/last_pose/`): MediaPipe Pose(33 랜드마크)로 자세 분류 시작
2. **얼굴 재식별 추가** (`bed/face-reidentification (1)/`): SCRFD + ArcFace로 환자 식별 기능 추가, PyQt GUI 구현
3. **YOLO11 기반 통합** (루트): YOLO11m-Pose로 자세 추정 업그레이드, YOLO11-Seg로 침대 감지 통합, 경량화된 최종 파이프라인 완성
