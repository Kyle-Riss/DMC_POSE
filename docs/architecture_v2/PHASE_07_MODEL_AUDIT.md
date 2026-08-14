# Phase 7 TCN 모델 성숙도 감사

## 결론

현재 `gmdcsa24_tcn/model.pt`는 학습과 기술적 서빙이 완료됐지만 운영 경보
가중치로는 미완성이다. 따라서 `TCN shadow → hybrid shadow` 경로만 유지한다.

## 데이터 경계

- train: subject 1, 2 / 80 videos / 41 fall videos
- validation: subject 3 / 43 videos / 21 fall videos
- test: subject 4 / 37 videos / 15 fall videos
- 총 160 videos, subject-disjoint split

분할 누수는 없지만 사람이 4명뿐이므로 카메라·병실·환자 일반화 근거가 매우
약하다. TSFM으로 모델 크기를 키우는 것보다 병상 자체 데이터와 하드 네거티브
시간을 늘리는 것이 우선이다.

## 현재 성능

Window 기준 test:

- precision 0.5197
- recall 0.8354
- F1 0.6408
- ROC-AUC 0.9007

3초 이내로 분절 이벤트를 합친 event 기준 test:

- precision 0.6667
- recall 0.9333
- false events 7 / 0.0803 hour
- 환산 false events/hour 87.18
- median latency 1.55 sec, p90 latency 2.669 sec

짧은 데이터의 시간당 환산값이므로 절대 운영 오탐률로 해석할 수는 없지만,
운영 승격 불가를 판단하기에는 충분히 높다.

## 발견한 평가 문제와 수정

한 낙상 구간 안에서 확률이 잠깐 threshold 아래로 내려가면 여러 이벤트로
쪼개져 오탐 수가 증가했다. `evaluate_temporal_events.py`에 `merge_gap_sec`를
추가했고 기본 3초로 재평가했다. test false event는 8건에서 7건으로 줄었지만
모델 일반화 부족이라는 결론은 변하지 않는다.

## 운영 승격 조건

- 병상 자체 데이터 subject/camera-disjoint test
- 최소 수십 시간 ADL shadow 기록
- 침대 가장자리 앉기, 천천히 내려오기, 의료진 접근, 물건 줍기 포함
- hybrid 기준 event recall 목표를 먼저 정하고 false alarm/bed-hour 측정
- 카메라별 threshold가 아니라 공통 threshold를 우선 검증
- 조건 충족 전 `SHADOW_ALERT`를 실제 호출·문자·웹훅에 연결하지 않음
