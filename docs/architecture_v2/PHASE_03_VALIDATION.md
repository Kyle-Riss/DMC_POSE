# Phase 3 검증 기록

검증 환경:

- 카메라 6대: `bed_161`, `bed_162`, `bed_174`, `bed_175`, `bed_178`, `bed_179`
- 중앙 서버 프레임 폭: 640
- RPi RTSP 입력: 약 20FPS
- 자동 Bed ROI cache: 6대 모두 READY

## 실카메라 결과

최종 10초 구간 측정:

| 카메라 | Capture FPS | Watcher FPS | EMPTY Pose FPS |
|---|---:|---:|---:|
| bed_161 | 20.01 | 19.70 | 0.75 |
| bed_162 | 20.52 | 19.80 | 0.74 |
| bed_174 | 20.18 | 19.80 | 0.74 |
| bed_175 | 20.17 | 19.90 | 0.74 |
| bed_178 | 20.17 | 20.00 | 0.74 |
| bed_179 | 20.34 | 20.10 | 0.74 |

결론:

- 영상 캡처와 감시 속도는 AI 절전 상태에서도 약 20FPS다.
- 빈 침대의 무거운 Pose 호출은 약 0.75FPS로 감소했다.
- 안정 ROI에서는 segmentation이 상시 반복되지 않았다.
- 합성 연속 움직임에서 BURST waiter는 300ms 이내 깨어나는 테스트를 둔다.

## 재검증

서버 실행 후:

```bash
/home/dmc/anaconda3/envs/pose-cuda/bin/python \
  scripts/check_phase3_runtime.py --seconds 10
```

단위/회귀 테스트:

```bash
/home/dmc/anaconda3/envs/pose-cuda/bin/python \
  -m unittest discover -s tests -v
```

## 아직 필요한 현장 리플레이

- 침대 진입과 퇴장
- 이불 움직임
- 조명 켜기/끄기
- 물건 줍기와 무릎 꿇기
- 침대 가장자리 미끄러짐
- 모사 낙상
- RTSP 손상 프레임

이 검증 전에는 TCN candidate와 motion trigger를 실제 외부 경보로 사용하지 않는다.
