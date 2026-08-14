from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FusionPhase(str, Enum):
    NO_PERSON = "NO_PERSON"
    INSUFFICIENT = "INSUFFICIENT"
    WARMING = "WARMING"
    TCN_NOT_READY = "TCN_NOT_READY"
    SAFE = "SAFE"
    CANDIDATE = "CANDIDATE"
    VERIFY = "VERIFY"
    SHADOW_ALERT = "SHADOW_ALERT"


@dataclass(frozen=True)
class FusionInput:
    timestamp: float
    track_id: int | None
    primary_observed: bool
    bed_roi_ready: bool
    body_in_bed_ratio: float
    pose_class: str
    pose_confidence: float
    legacy_fall_score: float
    rapid_motion: bool
    motion_ratio: float
    tcn_ready: bool
    tcn_probability: float
    tcn_threshold: float
    tcn_candidate: bool
    missing_samples: int = 0


@dataclass(frozen=True)
class FusionResult:
    phase: FusionPhase
    risk: float
    evidence: tuple[str, ...]
    safe_evidence: tuple[str, ...]
    candidate_age_sec: float
    quality: float
    track_id: int | None


class HybridFusion:
    """Stateful shadow fusion. It never emits the production ALERT action."""

    LYING = {"front_lying", "prone_back", "side_near", "side_far"}

    def __init__(self, *, verify_after_sec: float = 0.5, alert_after_sec: float = 1.5,
                 alert_hold_sec: float = 3.0, motion_ratio_threshold: float = 0.018,
                 pose_dropout_grace_sec: float = 1.5,
                 rapid_bed_departure_sec: float = 3.5,
                 lying_departure_arm_sec: float = 4.5,
                 lying_departure_confirm_sec: float = 2.0,
                 direct_rapid_departure_sec: float = 2.0):
        self.verify_after_sec = float(verify_after_sec)
        self.alert_after_sec = float(alert_after_sec)
        self.alert_hold_sec = float(alert_hold_sec)
        self.motion_ratio_threshold = max(1e-6, float(motion_ratio_threshold))
        self.pose_dropout_grace_sec = max(0.0, float(pose_dropout_grace_sec))
        self.rapid_bed_departure_sec = max(0.0, float(rapid_bed_departure_sec))
        self.lying_departure_arm_sec = max(0.0, float(lying_departure_arm_sec))
        self.lying_departure_confirm_sec = max(
            0.0, float(lying_departure_confirm_sec)
        )
        self.direct_rapid_departure_sec = max(
            0.0, float(direct_rapid_departure_sec)
        )
        self.track_id: int | None = None
        self.candidate_since: float | None = None
        self.alert_until: float = 0.0
        self.dropout_armed = False
        self.dropout_started_at: float | None = None
        self.retained_risk = 0.0
        self.retained_evidence: tuple[str, ...] = ()
        self.last_in_bed_at: float | None = None
        self.lying_departure_started_at: float | None = None
        self.direct_rapid_departure_started_at: float | None = None

    def reset(self, track_id: int | None = None) -> None:
        self.track_id = track_id
        self.candidate_since = None
        self.alert_until = 0.0
        self.dropout_armed = False
        self.dropout_started_at = None
        self.retained_risk = 0.0
        self.retained_evidence = ()
        self.last_in_bed_at = None
        self.lying_departure_started_at = None
        self.direct_rapid_departure_started_at = None

    @staticmethod
    def _clip(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def update(self, x: FusionInput) -> FusionResult:
        now = float(x.timestamp)
        if x.track_id != self.track_id:
            self.reset(x.track_id)

        if x.track_id is None:
            return FusionResult(FusionPhase.NO_PERSON, 0.0, (), (), 0.0, 0.0, None)

        pose_quality = self._clip(x.pose_confidence / 0.6)
        missing_quality = self._clip(1.0 - x.missing_samples / 15.0)
        quality = pose_quality * missing_quality if x.primary_observed else 0.0
        if not x.primary_observed:
            if self.lying_departure_started_at is not None:
                transition_age = max(0.0, now - self.lying_departure_started_at)
                if transition_age <= self.lying_departure_confirm_sec:
                    return FusionResult(
                        FusionPhase.VERIFY, self.retained_risk,
                        ("bed_departure_lying_transition", "pose_dropout_grace"),
                        (), transition_age, quality, x.track_id,
                    )
                self.lying_departure_started_at = None
            if self.dropout_armed and self.candidate_since is not None:
                if self.dropout_started_at is None:
                    self.dropout_started_at = now
                dropout_age = max(0.0, now - self.dropout_started_at)
                if dropout_age <= self.pose_dropout_grace_sec:
                    evidence = tuple(dict.fromkeys(
                        (*self.retained_evidence, "pose_dropout_grace")
                    ))
                    return FusionResult(
                        FusionPhase.VERIFY, self.retained_risk, evidence, (),
                        max(0.0, now - self.candidate_since), quality, x.track_id,
                    )
            self.candidate_since = None
            self.dropout_armed = False
            self.dropout_started_at = None
            self.retained_risk = 0.0
            self.retained_evidence = ()
            return FusionResult(
                FusionPhase.INSUFFICIENT, 0.0, ("primary_unobserved",), (),
                0.0, quality, x.track_id,
            )

        kinematic = self._clip(x.legacy_fall_score / 100.0)
        temporal = self._clip(x.tcn_probability) if x.tcn_ready else 0.0
        motion = 1.0 if x.rapid_motion else self._clip(
            x.motion_ratio / self.motion_ratio_threshold
        )
        outside = x.bed_roi_ready and x.body_in_bed_ratio < 0.35
        in_bed = x.bed_roi_ready and x.body_in_bed_ratio >= 0.80
        if in_bed:
            self.last_in_bed_at = now
            if x.tcn_candidate and x.rapid_motion:
                self.direct_rapid_departure_started_at = now
        lying = x.pose_class in self.LYING and x.pose_confidence >= 0.35
        unsafe_posture = lying and outside

        # Fall evidence often lands in adjacent frames: a low/lying body
        # crossing the bed edge, followed by an outside-bed observation.
        # Preserve that ordered transition instead of requiring both facts in
        # one frame. TCN is required to arm it, so a seated exit cannot.
        lying_departure_onset = (
            x.tcn_candidate and lying and x.bed_roi_ready and
            0.35 <= x.body_in_bed_ratio < 0.80 and
            self.last_in_bed_at is not None and
            now - self.last_in_bed_at <= self.lying_departure_arm_sec
        )
        if lying_departure_onset:
            self.lying_departure_started_at = now
            self.retained_risk = max(self.retained_risk, 0.5)

        lying_departure_age = (
            max(0.0, now - self.lying_departure_started_at)
            if self.lying_departure_started_at is not None else None
        )
        lying_departure_active = (
            lying_departure_age is not None and
            lying_departure_age <= self.lying_departure_confirm_sec
        )
        lying_departure_confirmed = (
            lying_departure_active and x.bed_roi_ready and
            x.body_in_bed_ratio <= 0.25
        )
        if self.lying_departure_started_at is not None and not lying_departure_active:
            self.lying_departure_started_at = None

        direct_rapid_departure_age = (
            max(0.0, now - self.direct_rapid_departure_started_at)
            if self.direct_rapid_departure_started_at is not None else None
        )
        direct_rapid_departure_active = (
            direct_rapid_departure_age is not None and
            direct_rapid_departure_age <= self.direct_rapid_departure_sec
        )
        direct_rapid_departure_confirmed = (
            direct_rapid_departure_active and x.tcn_candidate and
            x.rapid_motion and x.bed_roi_ready and
            x.body_in_bed_ratio <= 0.25
        )
        if (
            self.direct_rapid_departure_started_at is not None and
            not direct_rapid_departure_active
        ):
            self.direct_rapid_departure_started_at = None

        posture = 0.85 if unsafe_posture else (0.25 if x.pose_class == "sitting_edge" else 0.0)
        spatial = 0.8 if outside else (0.3 if x.bed_roi_ready and x.body_in_bed_ratio < 0.60 else 0.0)
        risk = 0.40 * temporal + 0.25 * kinematic + 0.15 * posture + 0.10 * spatial + 0.10 * motion

        evidence: list[str] = []
        safe: list[str] = []
        if x.rapid_motion:
            evidence.append("rapid_motion")
        if kinematic >= 0.35:
            evidence.append("kinematic_risk")
        if x.tcn_candidate:
            evidence.append("tcn_persistent")
        if unsafe_posture:
            evidence.append("outside_bed_lying")
        if outside:
            evidence.append("outside_bed")
        if lying_departure_active:
            evidence.append("bed_departure_lying_transition")
        if lying_departure_confirmed:
            evidence.append("bed_departure_transition_confirmed")
        if direct_rapid_departure_active:
            evidence.append("direct_rapid_bed_departure")
        if direct_rapid_departure_confirmed:
            evidence.append("direct_rapid_bed_departure_confirmed")
        incident_without_tcn = (
            not x.tcn_ready and (
                x.rapid_motion or kinematic >= 0.35 or unsafe_posture
            )
        )
        if incident_without_tcn:
            evidence.append("tcn_not_ready")

        strong_temporal = x.tcn_candidate and (
            kinematic >= 0.35 or x.rapid_motion or unsafe_posture
        )
        strong_fallback = (
            not x.tcn_ready and kinematic >= 0.65 and x.rapid_motion and
            (unsafe_posture or x.pose_class == "sitting_edge")
        )
        # Fast and slow falls have separate confirmation routes. Motion is a
        # wake-up/candidate signal, not a universal requirement. TCN alone, or
        # TCN + kinematic evidence without a post-event low/lying result, can
        # reach VERIFY but cannot escalate to SHADOW_ALERT.
        structural_confirmation = unsafe_posture or kinematic >= 0.35
        post_event_low = lying and (outside or not x.bed_roi_ready)
        fast_alert_confirmed = (
            x.tcn_candidate and x.rapid_motion and structural_confirmation
        )
        slow_alert_confirmed = (
            x.tcn_candidate and kinematic >= 0.35 and post_event_low
        )
        alert_confirmed = (
            fast_alert_confirmed or slow_alert_confirmed or strong_fallback
        )
        candidate = quality >= 0.35 and (strong_temporal or strong_fallback)
        if lying_departure_active:
            candidate = quality >= 0.35
            alert_confirmed = lying_departure_confirmed
        if direct_rapid_departure_active:
            candidate = quality >= 0.35
            alert_confirmed = direct_rapid_departure_confirmed

        # A verified temporal+motion incident may survive a very short pose
        # dropout. The observed-only TCN buffer still resets independently.
        # Retained evidence cannot alert while pose is absent and requires
        # fresh structural/kinematic confirmation after reacquisition.
        retained_after_dropout = (
            self.dropout_started_at is not None and
            now - self.dropout_started_at <= self.pose_dropout_grace_sec
        )
        rapid_bed_departure = (
            retained_after_dropout and
            self.last_in_bed_at is not None and
            now - self.last_in_bed_at <= self.rapid_bed_departure_sec and
            x.bed_roi_ready and x.body_in_bed_ratio <= 0.25
        )
        if retained_after_dropout:
            evidence.append("retained_temporal_context")
            evidence.append("pose_reacquired")
            candidate = quality >= 0.35
            if rapid_bed_departure:
                evidence.append("rapid_bed_departure")
            alert_confirmed = structural_confirmation or rapid_bed_departure

        calm_in_bed = (
            in_bed and lying and not x.rapid_motion and
            kinematic < 0.30 and not x.tcn_candidate
        )
        if calm_in_bed:
            safe.append("stable_in_bed_posture")
            risk = max(0.0, risk - 0.20)
        if not x.bed_roi_ready:
            safe.append("bed_context_unknown")

        risk = self._clip(risk)
        if candidate:
            if self.candidate_since is None:
                self.candidate_since = now
            age = max(0.0, now - self.candidate_since)
            if lying_departure_confirmed or direct_rapid_departure_confirmed:
                # Completing this ordered sequence is the persistence check.
                phase = FusionPhase.SHADOW_ALERT
                self.alert_until = now + self.alert_hold_sec
                self.lying_departure_started_at = None
                self.direct_rapid_departure_started_at = None
            elif age >= self.alert_after_sec and alert_confirmed:
                phase = FusionPhase.SHADOW_ALERT
                self.alert_until = now + self.alert_hold_sec
            elif now < self.alert_until:
                phase = FusionPhase.SHADOW_ALERT
            elif age >= self.verify_after_sec:
                phase = FusionPhase.VERIFY
            else:
                phase = FusionPhase.CANDIDATE

            if (
                phase in {FusionPhase.VERIFY, FusionPhase.SHADOW_ALERT} and
                x.tcn_candidate and x.rapid_motion
            ):
                self.dropout_armed = True
                self.retained_risk = risk
                self.retained_evidence = tuple(evidence)
            if retained_after_dropout and phase == FusionPhase.SHADOW_ALERT:
                self.dropout_started_at = None
                self.dropout_armed = False
        else:
            age = 0.0
            self.candidate_since = None
            if now < self.alert_until:
                phase = FusionPhase.SHADOW_ALERT
            elif incident_without_tcn:
                # Never call an incident-looking frame SAFE merely because the
                # 30-observation temporal context is unavailable.
                phase = FusionPhase.TCN_NOT_READY
            elif not x.tcn_ready:
                phase = FusionPhase.WARMING
            else:
                phase = FusionPhase.SAFE

            if self.dropout_started_at is not None:
                self.dropout_started_at = None
                self.dropout_armed = False
                self.retained_risk = 0.0
                self.retained_evidence = ()

        return FusionResult(
            phase, risk, tuple(evidence), tuple(safe), age, quality, x.track_id
        )
