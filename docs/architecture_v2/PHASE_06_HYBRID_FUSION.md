# Phase 6 하이브리드 Shadow Fusion

## 목적

침대 segmentation, primary skeleton, 6-class posture, 기존 기구학 점수,
10Hz TCN, 빠른 motion watcher를 하나의 추적 ID 기준 판정으로 결합한다.
이번 단계의 출력은 `SHADOW_ALERT`까지이며 외부 알림을 발생시키지 않는다.

## 판정 계약

- `NO_PERSON`: primary track 없음
- `INSUFFICIENT`: track은 있으나 이번 pose에서 primary가 관측되지 않음
- `WARMING`: 사람이 있으나 TCN 30 sample 준비 전
- `SAFE`: TCN 준비 후 결합 후보 증거 없음
- `CANDIDATE`: 복수 계층의 증거가 함께 성립
- `VERIFY`: 후보가 0.5초 이상 지속
- `SHADOW_ALERT`: 후보가 1.5초 이상 지속; API/viewer에만 노출

TCN의 `candidate`는 이미 0.5초 stride prediction 두 번의 지속 조건을
포함한다. Fusion은 이에 motion, 기구학 또는 침대 밖 위험 자세 중 하나가
동반될 때만 후보를 연다. TCN 준비 전에는 강한 기구학 + rapid motion +
위험 자세의 세 조건을 모두 요구한다.

## 침대 안전 규칙

`IN_BED + lying`은 hard gate가 아니다. 급격한 움직임과 높은 기구학 또는
TCN 증거가 없을 때만 `stable_in_bed_posture`라는 soft safety evidence로
risk를 낮춘다. Bed ROI가 준비되지 않은 상태는 `SAFE`로 간주하지 않고
`bed_context_unknown`으로 기록한다.

## ID와 시간 경계

- Fusion owner는 항상 `primary_track_id`와 같아야 한다.
- primary ID 변경 시 candidate timer와 alert hold를 초기화한다.
- primary가 잠깐 미관측되면 `INSUFFICIENT`이며 과거 skeleton을 재사용하지 않는다.
- `CANDIDATE`, `VERIFY`, `SHADOW_ALERT`에서는 3초 동안 pose 요청을 P0로 승격한다.

## API 필드

```text
fusion_phase
fusion_risk
fusion_evidence[]
fusion_safe_evidence[]
fusion_candidate_age_sec
fusion_quality
fusion_track_id
```

실시간 검증:

```bash
/home/dmc/anaconda3/envs/pose-cuda/bin/python \
  scripts/check_phase6_fusion.py --seconds 10
```

흐름도: `08_phase6_hybrid_fusion_flow.mmd`
