#!/usr/bin/env python3
"""Build a reproducible log + normalized-skeleton fall evidence artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_fusion import FusionInput, FusionPhase, HybridFusion

KST = timezone.utc
TCN_THRESHOLD = 0.5565
IN_BED_THRESHOLD = 0.80
OUTSIDE_THRESHOLD = 0.25

COCO_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path, camera: str, start: datetime, end: datetime) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("camera_id") != camera:
                continue
            ts = parse_ts(str(row.get("timestamp")))
            if start <= ts <= end:
                row["_dt"] = ts
                rows.append(row)
    rows.sort(key=lambda row: row["_dt"])
    return rows


def replay_v6(rows: list[dict]) -> list[dict]:
    fusion = HybridFusion()
    output: list[dict] = []
    for row in rows:
        result = fusion.update(FusionInput(
            timestamp=row["_dt"].timestamp(),
            track_id=row.get("primary_track_id"),
            primary_observed=bool(row.get("primary_track_observed")),
            bed_roi_ready=bool(row.get("bed_roi_ready")),
            body_in_bed_ratio=float(row.get("body_in_bed_ratio") or 0.0),
            pose_class=str(row.get("pose") or "None"),
            pose_confidence=float(row.get("pose_conf") or 0.0),
            legacy_fall_score=float(row.get("fall_score") or 0.0),
            rapid_motion=bool(row.get("burst_active")),
            motion_ratio=float(row.get("motion_ratio") or 0.0),
            tcn_ready=bool(row.get("tcn_shadow_ready")),
            tcn_probability=float(row.get("tcn_fall_probability") or 0.0),
            tcn_threshold=TCN_THRESHOLD,
            tcn_candidate=bool(row.get("tcn_alert_candidate")),
            missing_samples=int(row.get("tcn_missing_samples_window") or 0),
        ))
        output.append({
            "phase": result.phase.value,
            "risk": result.risk,
            "evidence": list(result.evidence),
        })
    return output


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.abs(values - target).argmin())


def draw_skeleton(ax, feature: np.ndarray, quality: float, title: str, subtitle: str) -> None:
    xy = feature[:34].reshape(17, 2)
    conf = feature[34:51]
    visible = feature[51:68] > 0.5
    usable = visible & np.isfinite(xy).all(axis=1)

    for left, right in COCO_EDGES:
        if usable[left] and usable[right]:
            ax.plot(
                [xy[left, 0], xy[right, 0]],
                [-xy[left, 1], -xy[right, 1]],
                color="#47b5ff", linewidth=2.2, alpha=0.9,
            )
    colors = np.where(usable, conf, 0.0)
    ax.scatter(
        xy[usable, 0], -xy[usable, 1], c=colors[usable], cmap="viridis",
        vmin=0, vmax=1, s=46, edgecolors="white", linewidths=0.5, zorder=3,
    )
    ax.set_title(title, fontsize=12, weight="bold", pad=8, color="#f0f6fc")
    ax.text(0.5, -0.08, f"{subtitle}\npose quality={quality:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            color="#d5d9e2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    ax.set_facecolor("#151923")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session-dir", type=Path,
        default=PROJECT_ROOT / "runtime_data/temporal_sessions/bed_161_20260814T044904_342262Z",
    )
    parser.add_argument(
        "--log", type=Path,
        default=PROJECT_ROOT / "runtime_data/shadow_features/shadow_features_20260814.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "docs/artifacts/fall_event_bed_161_20260814_0449",
    )
    args = parser.parse_args()

    manifest_path = args.session_dir / "manifest.json"
    features_path = args.session_dir / "features.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive = np.load(features_path)

    end = parse_ts(manifest["finished_at"])
    relative = archive["relative_timestamps_sec"]
    base = end.timestamp() - float(relative[-1])
    sample_epochs = base + relative

    window_start = datetime.fromtimestamp(base, tz=timezone.utc)
    window_end = end
    rows = load_rows(args.log, manifest["camera_id"], window_start, window_end)
    if not rows:
        raise RuntimeError("No matching log rows found")
    replay = replay_v6(rows)

    alert_indices = [i for i, item in enumerate(replay) if item["phase"] == FusionPhase.SHADOW_ALERT.value]
    if not alert_indices:
        raise RuntimeError("Current v6 policy produced no SHADOW_ALERT in this event")
    first_alert_idx = alert_indices[0]
    first_alert_row = rows[first_alert_idx]
    first_alert = first_alert_row["_dt"]

    candidate_indices = [i for i, row in enumerate(rows) if row.get("tcn_alert_candidate")]
    arm_idx = candidate_indices[0]
    arm_row = rows[arm_idx]
    arm = arm_row["_dt"]
    mid = parse_ts("2026-08-14T04:49:05.917859Z")

    snapshots = [
        ("침대 안 / 후보 형성", arm),
        ("침대 경계 통과", mid),
        ("침대 밖 / v6 경보", first_alert),
    ]
    feature_indices = [nearest_index(sample_epochs, dt.timestamp()) for _, dt in snapshots]

    event_rows = [row for row in rows if arm <= row["_dt"] <= first_alert]
    start_overlap = float(arm_row.get("body_in_bed_ratio") or 0.0)
    end_overlap = float(first_alert_row.get("body_in_bed_ratio") or 0.0)
    elapsed = (first_alert - arm).total_seconds()
    peak_tcn = max(float(row.get("tcn_fall_probability") or 0.0) for row in event_rows)
    track_id = arm_row.get("primary_track_id")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "fall_event_evidence_dashboard.png"
    svg_path = args.output_dir / "fall_event_evidence_dashboard.svg"
    json_path = args.output_dir / "evidence.json"
    frozen_log_path = args.output_dir / "event_log_rows.jsonl"

    # The daily JSONL is append-only while the server is running. Freeze only
    # this event window so the report checksum stays reproducible.
    with frozen_log_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            frozen = {key: value for key, value in row.items() if key != "_dt"}
            handle.write(json.dumps(frozen, ensure_ascii=False, sort_keys=True) + "\n")

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(font_path)
        ).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
    grid = fig.add_gridspec(3, 3, height_ratios=(1.9, 0.14, 1.2), hspace=0.40, wspace=0.16)
    ax = fig.add_subplot(grid[0, :])
    ax.set_facecolor("#151923")

    times = [row["_dt"] for row in rows]
    tcn = [float(row.get("tcn_fall_probability") or 0.0) for row in rows]
    overlap = [float(row.get("body_in_bed_ratio") or 0.0) if row.get("primary_track_observed") else np.nan for row in rows]
    ax.plot(times, tcn, color="#ffcc66", linewidth=2.4, marker="o", markersize=3.5,
            label="TCN 낙상 확률")
    ax.plot(times, overlap, color="#47b5ff", linewidth=2.4, marker="o", markersize=3.5,
            label="신체-침대 겹침률")
    ax.axhline(TCN_THRESHOLD, color="#ffcc66", linestyle="--", alpha=0.55,
               label=f"TCN 임계값 {TCN_THRESHOLD:.4f}")
    ax.axhline(IN_BED_THRESHOLD, color="#68d391", linestyle=":", alpha=0.65,
               label=f"침대 안 기준 {IN_BED_THRESHOLD:.2f}")
    ax.axhline(OUTSIDE_THRESHOLD, color="#ff6b6b", linestyle=":", alpha=0.8,
               label=f"침대 밖 확인 {OUTSIDE_THRESHOLD:.2f}")

    for i in range(len(rows) - 1):
        if rows[i].get("burst_active"):
            ax.axvspan(rows[i]["_dt"], rows[i + 1]["_dt"], color="#f97316", alpha=0.11)
    ax.axvline(arm, color="#a78bfa", linewidth=2, linestyle="--")
    ax.axvline(first_alert, color="#ff4d6d", linewidth=2.5)
    ax.annotate("v6 직접 급속 이탈 후보 형성", (arm, 1.02), xytext=(8, 8),
                textcoords="offset points", color="#d8c7ff", fontsize=10, weight="bold")
    ax.annotate("최초 SHADOW_ALERT", (first_alert, 1.02), xytext=(8, -24),
                textcoords="offset points", color="#ff8fa3", fontsize=10, weight="bold")
    ax.set_ylim(-0.03, 1.11)
    ax.set_ylabel("정규화 값 (0–1)", color="#e6edf3")
    ax.set_xlabel("UTC 시각 (KST = UTC+9)", color="#e6edf3")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    ax.grid(color="#30363d", alpha=0.55, linewidth=0.7)
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    legend = ax.legend(loc="lower left", ncols=3, fontsize=8.7, framealpha=0.85)
    legend.get_frame().set_facecolor("#0d1117")
    for item in legend.get_texts():
        item.set_color("#e6edf3")
    ax.set_title(
        "bed_161 실제 낙상 시뮬레이션 — 로그 + 영상 유래 스켈레톤 통합 근거",
        color="white", fontsize=17, weight="bold", pad=15,
    )

    # Thin phase/evidence band.
    band = fig.add_subplot(grid[1, :], sharex=ax)
    band.set_facecolor("#151923")
    phase_colors = {
        "SAFE": "#2ea043", "WARMING": "#6e7681", "TCN_NOT_READY": "#8b949e",
        "INSUFFICIENT": "#484f58", "CANDIDATE": "#a371f7", "VERIFY": "#db6d28",
        "SHADOW_ALERT": "#f85149", "NO_PERSON": "#30363d",
    }
    for i, item in enumerate(replay[:-1]):
        band.axvspan(times[i], times[i + 1], color=phase_colors[item["phase"]], alpha=0.95)
    band.set_yticks([])
    band.set_ylabel("v6", rotation=0, labelpad=18, color="#e6edf3", va="center")
    band.tick_params(axis="x", labelbottom=False)
    for spine in band.spines.values():
        spine.set_visible(False)

    features = archive["features"]
    quality = archive["pose_quality"]
    for col, ((label, dt), index) in enumerate(zip(snapshots, feature_indices)):
        row_index = min(range(len(rows)), key=lambda i: abs((rows[i]["_dt"] - dt).total_seconds()))
        row = rows[row_index]
        subtitle = (
            f"{dt.strftime('%H:%M:%S.%f')[:12]} UTC | track {track_id} | "
            f"overlap={float(row.get('body_in_bed_ratio') or 0):.3f} | "
            f"TCN={float(row.get('tcn_fall_probability') or 0):.4f}"
        )
        skel_ax = fig.add_subplot(grid[2, col])
        draw_skeleton(skel_ax, features[index], float(quality[index]), label, subtitle)

    summary = (
        f"검증 핵심  •  동일 track {track_id}  •  겹침률 {start_overlap:.3f} → {end_overlap:.3f} "
        f"({elapsed:.3f}s)  •  TCN peak {peak_tcn:.4f}  •  최초 v6 경보 {first_alert.strftime('%H:%M:%S.%f')[:12]} UTC\n"
        "스켈레톤은 당시 영상에서 저장된 109D 정규화 Pose feature로 복원했습니다. "
        "RGB/원본 픽셀 위치는 개인정보 최소화 정책으로 저장되지 않아 포함하지 않았습니다."
    )
    fig.text(0.5, 0.015, summary, ha="center", va="bottom", color="#c9d1d9", fontsize=9.5)
    fig.savefig(png_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(svg_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    evidence = {
        "schema_version": "fall_event_evidence_v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "event": {
            "session_id": manifest["session_id"],
            "camera_id": manifest["camera_id"],
            "track_id": track_id,
            "label": "controlled_safe_fall_simulation_user_confirmed",
            "arm_timestamp_utc": arm.isoformat().replace("+00:00", "Z"),
            "first_v6_shadow_alert_utc": first_alert.isoformat().replace("+00:00", "Z"),
            "confirmation_latency_from_arm_sec": elapsed,
            "bed_overlap_start": start_overlap,
            "bed_overlap_at_alert": end_overlap,
            "tcn_peak_probability": peak_tcn,
            "v6_first_alert_risk": replay[first_alert_idx]["risk"],
            "v6_first_alert_evidence": replay[first_alert_idx]["evidence"],
        },
        "thresholds": {
            "tcn": TCN_THRESHOLD,
            "in_bed": IN_BED_THRESHOLD,
            "outside_confirm": OUTSIDE_THRESHOLD,
            "direct_rapid_departure_sec": 2.0,
        },
        "source": {
            "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "features": str(features_path.relative_to(PROJECT_ROOT)),
            "log": str(args.log.relative_to(PROJECT_ROOT)),
            "frozen_event_log": str(frozen_log_path.relative_to(PROJECT_ROOT)),
            "manifest_sha256": sha256(manifest_path),
            "features_sha256": sha256(features_path),
            "daily_log_sha256_at_generation": sha256(args.log),
            "frozen_event_log_sha256": sha256(frozen_log_path),
            "log_rows_in_session": len(rows),
            "feature_samples": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
        },
        "limitations": [
            "The event was a controlled safe simulation, not an uncontrolled clinical fall.",
            "RGB/video frames were not retained for this session.",
            "Skeleton panels use normalized 109D pose coordinates and do not show pixel-space bed position.",
            "v6 result is an offline replay of the current shadow policy over the original v5 log inputs.",
            "Shadow alert is not a production alarm promotion.",
        ],
        "artifacts": {
            "png": png_path.name,
            "svg": svg_path.name,
        },
    }
    json_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
