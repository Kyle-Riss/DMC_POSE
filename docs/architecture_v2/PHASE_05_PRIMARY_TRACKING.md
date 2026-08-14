# Phase 5 — Primary patient tracking과 track별 TCN

## 해결한 문제

이전 코드는 `keypoints[0]`만 사용했다. YOLO detection 순서는 사람의 신원이 아니므로 의료진이 들어오면 환자 3초 history 뒤에 의료진 skeleton이 붙을 수 있었다.

현재는 한 Pose 결과의 모든 사람을 추출한다. CPU tracker가 bbox IoU와 화면 대비 중심 거리로 이전 track과 연결한다. Primary 환자는 다음 점수와 hysteresis로 선택한다.

1. 침대 영역과 지속적으로 겹친 정도의 EMA
2. 기존 primary track 연속성
3. keypoint confidence와 track maturity
4. challenger가 margin을 충분히 넘을 때만 switch

## temporal 소유권

```text
camera
  person track 17
    frame buffer 30
    TCN runner 10Hz
  person track 18
    별도 frame buffer
    별도 TCN runner
```

6-class와 TCN 입력에는 primary track의 keypoint만 들어간다. Primary ID가 바뀌면 새 ID의 buffer를 사용하며 두 사람의 history를 이어 붙이지 않는다. 5초 TTL을 넘긴 track의 frame/TCN buffer는 삭제한다. TCN 관측 gap이 1.5초를 넘으면 해당 runner만 reset된다.

## API와 Viewer

- `person_count`, `track_count`
- `primary_track_id`, `track_switch_total`
- `primary_track_bed_overlap`, `primary_track_confidence`
- `tcn_track_id`, `tcn_track_reset_total`, `tcn_gap_reset_total`

Viewer의 `Primary Track`에서 ID, 현재 감지 인원, 유지 중 track 수, switch 수를 본다. TCN 행은 owner ID와 gap reset 수를 함께 표시한다.

## 현재 한계

이 tracker는 pose bbox 기반 단기 연결이며 얼굴 인식이나 장기 re-identification을 하지 않는다. 카메라 밖으로 나갔다 다시 들어오면 새 ID가 맞다. 사람이 심하게 겹치거나 완전히 가려지는 현장은 별도 리플레이로 switch 정확도를 측정해야 한다. TCN은 계속 shadow다.
