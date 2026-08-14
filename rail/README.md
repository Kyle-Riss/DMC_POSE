# Rail detection (restored from pose-sixclass-viewer)

난간 UP/DOWN 판별 파이프라인. 2026-05-11 `pose-sixclass-viewer`의 `server.py` / `capture_rail_pair.py`에서 복원.

## 파일

| 파일 | 설명 |
|------|------|
| `rail_detect.py` | edge / diff / diff2 판별 로직 |
| `rail_config.json` | ROI, 참조 프레임, 임계값 |
| `capture_rail_pair.py` | RTSP에서 DOWN·UP 쌍 촬영 + ROI 제안 |
| `annotate_bbox.py` | `extended rail` / `folded side` bbox 라벨링 |
| `test_rail_detect.py` | 정지 이미지 테스트 |
| `reference/` | 팀원이 캡처해 둔 참조 프레임 |

## 빠른 테스트

```bash
cd /home/dmc/pose-sixclass
python3 rail/test_rail_detect.py rail/reference/rtsp_one_frame.jpg
```

## 참조 프레임 캡처 (RTSP)

```bash
python3 rail/capture_rail_pair.py --side left --rtsp rtsp://...
python3 rail/capture_rail_pair.py --side right
```

난간을 DOWN → Enter → UP → Enter 순으로 촬영하면 `{side}_down.jpg`, `{side}_up.jpg`, absdiff 맵이 `reference/`에 저장됩니다.

## bbox 라벨링

```bash
python3 rail/annotate_bbox.py rail/reference/rtsp_one_frame.jpg
```

- `1` = extended rail, `2` = folded side
- `u` = undo, `s` = save

## 판별 방식

- **edge** (기본, 우측): ROI 내 Canny edge 밀도
- **diff**: 단일 참조 프레임과 absdiff 평균
- **diff2** (좌측 권장): DOWN/UP 참조 쌍과 각각 비교 후 마진으로 판정

`server.py` 연동 시 `detect_both_rails(frame, person_xyxy=...)` 호출 → `rail_left_up`, `rail_right_up` 등 반환.

## 출처

- `_archive/pose-sixclass-viewer-20260615/` (pyc + 이미지)
- 원본 소스는 유실; Python 3.11 bytecode disassembly로 복원
