# Phase 8 — Feature-only shadow recording

## 목적

실제 병상 운용 중 모델의 후보 경보 빈도를 `bed-hour` 기준으로 측정한다. 카메라 영상 모니터링과 AI 추론을 막지 않도록 기록은 비동기·비차단 방식이며, 개인정보 범위를 줄이기 위해 영상이나 스켈레톤 좌표는 저장하지 않는다.

## 실행 경로

```text
run_all_cameras.sh
  └─ server_all_cameras.py
       ├─ /video/{camera_id}: latest-frame 기반 빠른 MJPEG 모니터링
       ├─ motion watcher: 20 Hz 저비용 변화 감지
       ├─ scheduler/model/tracker/TCN/fusion
       └─ ShadowFeatureRecorder
            └─ runtime_data/shadow_features/shadow_features_YYYYMMDD.jsonl
```

전체 흐름은 `09_phase8_feature_recording_flow.mmd`에 있다.

## 기록 계약

- 기본 표본 간격: 카메라별 0.5초
- fusion phase가 바뀌면 표본 간격 전이라도 즉시 기록
- queue 크기: 2048, 가득 차면 추론을 기다리지 않고 해당 행만 drop
- 일자별 JSONL 순환
- `/recorder/status`에서 writer thread, queue, written/drop/error 수 확인
- `/status`와 `/viewer`에서도 recorder 상태 확인

저장 항목은 카메라 ID, 시간, bed ROI 준비 여부, motion 비율, track 식별자, 포즈/TCN/fusion 점수, 캡처 FPS, scheduler 지연 및 drop 계수처럼 운영 측정에 필요한 작은 값뿐이다.

다음 항목은 저장하지 않는다.

- 원본 영상 또는 JPEG
- keypoint 좌표 배열
- RTSP URL 또는 인증정보

`runtime_data/`는 Git 추적 대상에서 제외된다.

## 실행과 요약

서버는 기존 명령으로 실행한다.

```bash
cd /home/dmc/AI/DMC_POSE
./run_all_cameras.sh
```

기록 상태:

```bash
curl -fsS http://127.0.0.1:8000/recorder/status
```

누적 JSONL 요약:

```bash
/home/dmc/anaconda3/envs/pose-cuda/bin/python \
  summarize_shadow_features.py \
  --out runtime_data/shadow_summary.json
```

`SHADOW_ALERT`는 자동으로 오탐으로 간주하지 않는다. 요약기의 `review_candidates`를 실제 사건 또는 오탐으로 사람이 확인해야 비로소 false alarms/bed-hour를 계산할 수 있다.

## 2026-07-31 실기동 검증

- 6대 카메라 모두 capture/watcher 약 20 FPS
- 빈 방 pose probe 0.75 FPS
- 자동 bed ROI 6/6 ready
- recorder queue depth 0
- recorder drop 0, error 0
- 저장 JSONL에서 frame/keypoints/rtsp_url 없음
- Phase 4 scheduler 검사 통과
- Phase 6 fusion ownership 검사 통과
- 안정된 20초 구간 Phase 3 저부하 검사 통과
- 단위/통합 테스트 57개 통과

짧은 검증 구간의 shadow alert는 0건이었지만, 이는 정확도를 입증하는 결과가 아니다. 다음 단계는 정상 운영 시간을 충분히 누적하고 review candidate에 사람 라벨을 붙여 카메라별 임계값과 융합 지속시간을 교정하는 것이다.
