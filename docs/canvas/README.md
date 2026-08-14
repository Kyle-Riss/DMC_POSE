# Dataset · Canvas 문서

Cursor Canvas(`.canvas.tsx`)는 IDE에서 채팅 옆에 열 수 있는 플로우·데이터 요약 뷰입니다.

## 파일 목록

| 파일 | 내용 |
|------|------|
| `pose-pipeline-flow.canvas.tsx` | 현장 셋업 · A→G 런타임 · **Raw 배치 시계열** · Phase 0→5 |
| `pose-raw-timeseries-data.canvas.tsx` | Raw 8영상 timeseries 통계·CSV 스키마 |
| `pose-e2e-validation.canvas.tsx` | 6-class E2E 검증 (accuracy·confusion matrix) |

## Cursor에서 열기

프로젝트 canvases 폴더에 두고 엽니다:

```
~/.cursor/projects/home-dmc/canvases/<이름>.canvas.tsx
```

이 디렉터리(`pose-sixclass/docs/canvas/`)는 **USB Dataset 백업용 복사본**입니다.

## USB Dataset 경로

Moredigm 마운트 후:

```
/media/dmc/Moredigm1/Dataset/docs/canvas/
```

동기화:

```bash
bash /home/dmc/pose-sixclass/docs/canvas/sync_to_usb.sh
```

## 관련 스크립트

- `extract_raw_timeseries.py` — Raw MP4 → timeseries CSV/JSON
- `docs/FALL_RISK_SYSTEM_DESIGN.md` — S3~S8 설계
