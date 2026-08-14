# Bed ROI (fixed camera zone)

카메라가 고정이면 seg보다 **ROI bbox**가 bed zone/in_bed에 더 안정적입니다.

## 파일

| 파일 | 설명 |
|------|------|
| `bed_roi.json` | 침대 ROI (bbox_norm 또는 bbox_px) |
| `frame_ref.jpg` | ROI 기준 프레임 (800×450) |
| `roi_preview.jpg` | ROI 미리보기 |
| `pick_bed_roi.py` | 마우스로 ROI 수동 지정 |
| `roi_utils.py` | server에서 ROI clip/폴백 |

## ROI 수동 지정

```bash
conda activate pose-cuda
cd /home/dmc/pose-sixclass/bed_roi
python pick_bed_roi.py   # DISPLAY 필요
```

드래그 → Enter 저장 → `bed_roi.json` 갱신

## 서버 적용

기본 ON (`POSE_USE_BED_ROI=1`):

```bash
bash /home/dmc/pose-sixclass/run_server.sh
```

- 노란 테두리: ROI
- 회색 L/C/R: ROI 기준 bed zone
- seg mask는 ROI **안에서만** 표시
- seg bbox가 ROI보다 넓으면 **ROI bbox로 대체**

## bed_roi.json 예

```json
{
  "ref_width": 800,
  "ref_height": 450,
  "bbox_norm": [0.13, 0.10, 0.87, 0.90]
}
```

ROI 끄기: `POSE_USE_BED_ROI=0 bash run_server.sh`
