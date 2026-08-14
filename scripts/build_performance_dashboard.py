#!/usr/bin/env python3
"""Generate a dependency-free SVG/Markdown performance snapshot."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def status(url):
    try:
        with urlopen(url, timeout=5) as response:
            return json.load(response)
    except (OSError, URLError, ValueError):
        return {}


def rect(x, y, w, h, fill, rx=6):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"/>'


def text(x, y, value, size=18, fill="#e5e7eb", weight=400, anchor="start"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{value}</text>')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--status-url", default="http://127.0.0.1:8000/status")
    p.add_argument("--output-dir", type=Path, default=ROOT / "docs")
    a = p.parse_args()
    live = status(a.status_url)
    comp_path = ROOT / "runs/temporal_tcn/observed_only_checkpoint_comparison_recheck_20260807.json"
    fv_path = ROOT / "runs/temporal_tcn/fallvision_pilot_balanced_diagnostic_v1_report_recheck_20260807.json"
    comp = json.loads(comp_path.read_text())
    fv = json.loads(fv_path.read_text())
    legacy = comp["results"]["legacy_checkpoint"]
    v2 = comp["results"]["v2_observed_only_checkpoint"]
    fv_base = fv["results"]["baseline_v2"]
    cams = sorted(live)
    expected_cameras = 6
    live_available = bool(cams)
    average_capture = (
        sum(float(live[c].get("capture_fps", 0)) for c in cams) / len(cams)
        if cams else 0.0
    )
    now = datetime.now().astimezone().isoformat(timespec="seconds")

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900">',
           '<rect width="1400" height="900" fill="#0b1220"/>',
           '<style>text{font-family:DejaVu Sans,Arial,sans-serif}.grid{stroke:#334155;stroke-width:1}</style>',
           text(55, 58, "DMC_POSE 성능 스냅샷", 32, "#f8fafc", 700),
           text(55, 88, f"생성 {now} · live {'online' if live_available else 'offline'} + frozen test reports", 15, "#94a3b8")]

    cards = [
        ("카메라", f"{len(cams)}/{expected_cameras}", "#22c55e" if live_available else "#f59e0b"),
        ("자동 Bed ROI", f"{sum(bool(live[c].get('bed_roi_ready')) for c in cams)}/{expected_cameras} READY", "#22c55e" if live_available else "#f59e0b"),
        ("평균 Capture", f"{average_capture:.1f} FPS", "#38bdf8" if live_available else "#f59e0b"),
        ("Scheduler 오류", f"{sum(int(live[c].get('scheduler_error_total',0)) for c in cams)}", "#22c55e"),
        ("TCN 상태", "SHADOW", "#f59e0b"),
    ]
    for i, (label, value, color) in enumerate(cards):
        x = 55 + i * 258
        svg += [rect(x, 115, 235, 92, "#172033"), text(x+18, 145, label, 14, "#94a3b8"),
                text(x+18, 184, value, 25, color, 700)]

    # Runtime grouped bars.
    svg += [text(55, 255, "실시간 처리율 (현재 스냅샷)", 22, "#f8fafc", 700),
            text(55, 280, "Capture / Watcher / Pose FPS · EMPTY에서는 Pose 저주기", 14, "#94a3b8")]
    base_y, chart_h, max_fps = 490, 180, 22.0
    colors = ["#38bdf8", "#22c55e", "#a78bfa"]
    for tick in [0,5,10,15,20]:
        y = base_y - chart_h * tick/max_fps
        svg += [f'<line class="grid" x1="70" y1="{y:.1f}" x2="720" y2="{y:.1f}"/>',
                text(62, y+5, str(tick), 12, "#64748b", anchor="end")]
    for i, cam in enumerate(cams):
        x0 = 95 + i*102
        vals = [float(live[cam].get("capture_fps",0)), float(live[cam].get("watcher_fps",0)),
                float(live[cam].get("pose_inference_fps",0))]
        for j, value in enumerate(vals):
            h = min(chart_h, chart_h*value/max_fps)
            svg.append(rect(x0+j*22, base_y-h, 17, h, colors[j], 2))
        svg += [text(x0+24, 515, cam.replace("bed_", ""), 13, "#cbd5e1", anchor="middle"),
                text(x0+24, 535, live[cam].get("runtime_mode","?"), 10,
                     "#f59e0b" if live[cam].get("runtime_mode") != "EMPTY" else "#64748b", anchor="middle")]
    for i, (name, color) in enumerate(zip(["Capture", "Watcher", "Pose"], colors)):
        svg += [rect(80+i*120, 555, 14, 14, color, 2), text(101+i*120, 567, name, 12, "#cbd5e1")]

    # TCN metrics.
    svg += [text(775, 255, "TCN 동일 observed-only Test 비교", 22, "#f8fafc", 700),
            text(775, 280, "Validation에서 operating point 고정 후 Test 1회", 14, "#94a3b8")]
    metric_names = ["Window AUROC", "Event precision", "End-to-end recall", "Conditional recall"]
    keys = ["roc_auc", "event_precision", "end_to_end_event_recall", "conditional_event_recall"]
    for i, (name, key) in enumerate(zip(metric_names, keys)):
        y = 320+i*46
        lv = legacy["test_window_at_selected_threshold"].get(key) if key == "roc_auc" else legacy["test_event"][key]
        vv = v2["test_window_at_selected_threshold"].get(key) if key == "roc_auc" else v2["test_event"][key]
        svg += [text(775, y+15, name, 13, "#cbd5e1"), rect(925, y, 170*lv, 15, "#38bdf8", 2),
                rect(925, y+20, 170*vv, 15, "#a78bfa", 2),
                text(1105, y+13, f"Legacy {lv:.3f}", 11, "#7dd3fc"),
                text(1105, y+33, f"v2 {vv:.3f}", 11, "#c4b5fd")]
    svg += [rect(775, 520, 555, 70, "#351b22"),
            text(795, 548, f"False events/hour: Legacy {legacy['test_event']['false_events_per_hour']:.2f} · v2 {v2['test_event']['false_events_per_hour']:.2f}", 16, "#fca5a5", 700),
            text(795, 573, "운영 불가 수준 → 두 checkpoint 모두 production 승격 보류", 14, "#fecaca")]

    # External diagnostic and interpretation.
    svg += [text(55, 640, "FallVision frozen 외부 진단", 21, "#f8fafc", 700),
            rect(55, 662, 620, 165, "#172033"),
            text(78, 697, f"Event-evaluable coverage  {fv_base['event_evaluable_coverage']*100:.1f}%", 17, "#f8fafc", 600),
            text(78, 729, f"Conditional recall       {fv_base['conditional_event_recall']*100:.1f}%", 17, "#f8fafc", 600),
            text(78, 761, f"End-to-end recall        {fv_base['end_to_end_event_recall']*100:.1f}%", 17, "#f8fafc", 600),
            text(78, 793, f"False events/hour        {fv_base['false_events_per_hour']:.1f}", 17, "#fca5a5", 600),
            text(720, 640, "판정", 21, "#f8fafc", 700), rect(720, 662, 610, 165, "#172033"),
            text(745, 700, "✓ 런타임·ROI·부하 절감·입력 계약: PASS", 17, "#86efac", 700),
            text(745, 736, "△ 모델 후보: Pi benchmark_required", 17, "#fcd34d", 700),
            text(745, 772, "✕ 현재 TCN: event 오탐으로 승격 보류", 17, "#fca5a5", 700),
            text(745, 808, "다음 게이트: Pi 1대 30분 benchmark → shadow", 15, "#cbd5e1"),
            text(55, 872, "주의: 라이브 값은 한 시점 스냅샷이며, 모델 수치는 짧은 frozen test 영상 기준입니다.", 14, "#94a3b8"),
            '</svg>']

    a.output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = a.output_dir / "performance_dashboard_2026-08-07.svg"
    md_path = a.output_dir / "PERFORMANCE_SUMMARY_2026-08-07.md"
    svg_path.write_text("\n".join(svg), encoding="utf-8")
    rows = []
    for cam in cams:
        d = live[cam]
        rows.append(f"| {cam} | {d.get('runtime_mode')} | {d.get('capture_fps',0):.2f} | {d.get('watcher_fps',0):.2f} | {d.get('pose_inference_fps',0):.2f} | {d.get('bed_roi_agreement_iou',0):.3f} | {d.get('scheduler_inference_ms',0):.2f} |")
    live_note = "live status online" if live_available else "live status unavailable (:8000 offline at generation time)"
    md = f"""# DMC_POSE 성능 요약 — 2026-08-07

![성능 대시보드](performance_dashboard_2026-08-07.svg)

생성 시각: `{now}`  
Live 상태: **{live_note}**  
판정: **런타임 PASS / 모델 production 승격 보류 / Pi benchmark 필요**

## 실시간 6카메라 스냅샷

| Camera | Mode | Capture FPS | Watcher FPS | Pose FPS | ROI IoU | Inference ms |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## TCN frozen test

| Checkpoint | Window AUROC | Event precision | End-to-end recall | Conditional recall | False events/hour |
|---|---:|---:|---:|---:|---:|
| Legacy | {legacy['test_window_at_selected_threshold']['roc_auc']:.4f} | {legacy['test_event']['event_precision']:.4f} | {legacy['test_event']['end_to_end_event_recall']:.4f} | {legacy['test_event']['conditional_event_recall']:.4f} | {legacy['test_event']['false_events_per_hour']:.2f} |
| v2 observed-only | {v2['test_window_at_selected_threshold']['roc_auc']:.4f} | {v2['test_event']['event_precision']:.4f} | {v2['test_event']['end_to_end_event_recall']:.4f} | {v2['test_event']['conditional_event_recall']:.4f} | {v2['test_event']['false_events_per_hour']:.2f} |

FallVision frozen 진단은 coverage `{fv_base['event_evaluable_coverage']:.4f}`, conditional recall `{fv_base['conditional_event_recall']:.4f}`, end-to-end recall `{fv_base['end_to_end_event_recall']:.4f}`입니다. 짧은 평가 시간 때문에 FP/hour 신뢰구간이 넓으며, 현재 수치는 운영 승격 근거가 아닙니다.

출처: `{comp_path.relative_to(ROOT)}`, `{fv_path.relative_to(ROOT)}`, `:8000/status`.
"""
    md_path.write_text(md, encoding="utf-8")
    print(json.dumps({"svg": str(svg_path), "markdown": str(md_path), "cameras": len(cams)}))


if __name__ == "__main__":
    main()
