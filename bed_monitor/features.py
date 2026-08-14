from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MotionState:
    prev_center: tuple[float, float] | None = None
    prev_v: tuple[float, float] = (0.0, 0.0)
    last_seen_t: float = 0.0


@dataclass
class MotionResult:
    center_x: float | None
    center_y: float | None
    center_vx: float | None
    center_vy: float | None
    center_speed: float | None
    center_accel_like: float | None


def update_motion(
    state: MotionState,
    center: tuple[float, float] | None,
    t_sec: float,
    *,
    ema_alpha: float = 0.3,
    hold_sec: float = 0.5,
) -> MotionResult:
    if center is None:
        if state.prev_center is not None and (t_sec - state.last_seen_t) <= hold_sec:
            vx, vy = state.prev_v
            speed = (vx * vx + vy * vy) ** 0.5
            return MotionResult(None, None, vx, vy, speed, None)
        return MotionResult(None, None, None, None, None, None)

    cx, cy = center
    dt = t_sec - state.last_seen_t if state.last_seen_t > 0 else None
    vx = vy = speed = accel = None

    if state.prev_center is not None and dt and dt > 1e-6:
        inst_vx = (cx - state.prev_center[0]) / dt
        inst_vy = (cy - state.prev_center[1]) / dt
        pvx, pvy = state.prev_v
        vx = ema_alpha * inst_vx + (1 - ema_alpha) * pvx
        vy = ema_alpha * inst_vy + (1 - ema_alpha) * pvy
        speed = (vx * vx + vy * vy) ** 0.5
        accel = abs(((vx - pvx) ** 2 + (vy - pvy) ** 2) ** 0.5 / dt)

    state.prev_center = (cx, cy)
    state.last_seen_t = t_sec
    if vx is not None and vy is not None:
        state.prev_v = (vx, vy)

    return MotionResult(cx, cy, vx, vy, speed, accel)
