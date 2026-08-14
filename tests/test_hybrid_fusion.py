import unittest

from hybrid_fusion import FusionInput, FusionPhase, HybridFusion


def sample(ts=0.0, **overrides):
    values = dict(
        timestamp=ts, track_id=7, primary_observed=True, bed_roi_ready=True,
        body_in_bed_ratio=0.1, pose_class="front_lying", pose_confidence=0.9,
        legacy_fall_score=60.0, rapid_motion=True, motion_ratio=0.04,
        tcn_ready=True, tcn_probability=0.9, tcn_threshold=0.55,
        tcn_candidate=True, missing_samples=0,
    )
    values.update(overrides)
    return FusionInput(**values)


class HybridFusionTests(unittest.TestCase):
    def test_requires_persistent_hybrid_evidence(self):
        fusion = HybridFusion(verify_after_sec=0.5, alert_after_sec=1.5)
        self.assertEqual(fusion.update(sample(0.0)).phase, FusionPhase.CANDIDATE)
        self.assertEqual(fusion.update(sample(0.6)).phase, FusionPhase.VERIFY)
        result = fusion.update(sample(1.6))
        self.assertEqual(result.phase, FusionPhase.SHADOW_ALERT)
        self.assertIn("tcn_persistent", result.evidence)

    def test_tcn_alone_does_not_alert(self):
        fusion = HybridFusion()
        result = fusion.update(sample(
            0.0, legacy_fall_score=5.0, rapid_motion=False,
            motion_ratio=0.0, body_in_bed_ratio=0.95,
            pose_class="front_lying",
        ))
        self.assertEqual(result.phase, FusionPhase.SAFE)

    def test_tcn_plus_motion_without_structure_stays_verify(self):
        fusion = HybridFusion(verify_after_sec=0.5, alert_after_sec=1.5)
        inputs = dict(
            legacy_fall_score=0.0, body_in_bed_ratio=0.70,
            pose_class="sitting_center", pose_confidence=0.95, rapid_motion=True,
        )
        self.assertEqual(fusion.update(sample(0.0, **inputs)).phase, FusionPhase.CANDIDATE)
        self.assertEqual(fusion.update(sample(0.6, **inputs)).phase, FusionPhase.VERIFY)
        self.assertEqual(fusion.update(sample(2.0, **inputs)).phase, FusionPhase.VERIFY)

    def test_structure_can_confirm_existing_temporal_candidate(self):
        fusion = HybridFusion(verify_after_sec=0.5, alert_after_sec=1.5)
        inputs = dict(
            legacy_fall_score=0.0, body_in_bed_ratio=0.70,
            pose_class="sitting_center", pose_confidence=0.95, rapid_motion=True,
        )
        fusion.update(sample(0.0, **inputs))
        fusion.update(sample(1.0, **inputs))
        result = fusion.update(sample(
            1.6, **dict(inputs, body_in_bed_ratio=0.10, pose_class="side_near")
        ))
        self.assertEqual(result.phase, FusionPhase.SHADOW_ALERT)
        self.assertIn("outside_bed_lying", result.evidence)

    def test_slow_fall_can_alert_without_rapid_motion_after_low_posture(self):
        fusion = HybridFusion(verify_after_sec=0.5, alert_after_sec=1.5)
        inputs = dict(
            rapid_motion=False, motion_ratio=0.005,
            legacy_fall_score=55.0, body_in_bed_ratio=0.10,
            pose_class="side_near", pose_confidence=0.9,
        )
        self.assertEqual(fusion.update(sample(0.0, **inputs)).phase, FusionPhase.CANDIDATE)
        self.assertEqual(fusion.update(sample(0.6, **inputs)).phase, FusionPhase.VERIFY)
        self.assertEqual(fusion.update(sample(1.6, **inputs)).phase, FusionPhase.SHADOW_ALERT)

    def test_slow_tcn_plus_kinematic_without_low_posture_stays_verify(self):
        fusion = HybridFusion(verify_after_sec=0.5, alert_after_sec=1.5)
        inputs = dict(
            rapid_motion=False, motion_ratio=0.005,
            legacy_fall_score=55.0, body_in_bed_ratio=0.55,
            pose_class="sitting_center", pose_confidence=0.9,
        )
        fusion.update(sample(0.0, **inputs))
        fusion.update(sample(0.6, **inputs))
        self.assertEqual(fusion.update(sample(2.0, **inputs)).phase, FusionPhase.VERIFY)

    def test_in_bed_posture_is_soft_safety_evidence(self):
        fusion = HybridFusion()
        result = fusion.update(sample(
            0.0, tcn_candidate=False, tcn_probability=0.1,
            legacy_fall_score=5.0, rapid_motion=False, motion_ratio=0.0,
            body_in_bed_ratio=0.95,
        ))
        self.assertIn("stable_in_bed_posture", result.safe_evidence)
        self.assertEqual(result.phase, FusionPhase.SAFE)

    def test_unobserved_primary_is_insufficient_not_safe(self):
        fusion = HybridFusion()
        result = fusion.update(sample(0.0, primary_observed=False))
        self.assertEqual(result.phase, FusionPhase.INSUFFICIENT)

    def test_verified_incident_survives_short_pose_dropout(self):
        fusion = HybridFusion(
            verify_after_sec=0.5, alert_after_sec=1.5,
            pose_dropout_grace_sec=1.5,
        )
        fusion.update(sample(0.0))
        self.assertEqual(fusion.update(sample(0.6)).phase, FusionPhase.VERIFY)
        missing = fusion.update(sample(0.8, primary_observed=False))
        self.assertEqual(missing.phase, FusionPhase.VERIFY)
        self.assertIn("pose_dropout_grace", missing.evidence)

    def test_pose_dropout_never_alerts_while_unobserved(self):
        fusion = HybridFusion(
            verify_after_sec=0.5, alert_after_sec=1.5,
            pose_dropout_grace_sec=1.5,
        )
        fusion.update(sample(0.0))
        fusion.update(sample(0.6))
        self.assertEqual(
            fusion.update(sample(1.8, primary_observed=False)).phase,
            FusionPhase.VERIFY,
        )

    def test_reacquired_structure_can_confirm_retained_incident(self):
        fusion = HybridFusion(
            verify_after_sec=0.5, alert_after_sec=1.5,
            pose_dropout_grace_sec=1.5,
        )
        fusion.update(sample(0.0))
        fusion.update(sample(0.6))
        fusion.update(sample(0.8, primary_observed=False))
        result = fusion.update(sample(
            1.6, tcn_ready=False, tcn_candidate=False,
            tcn_probability=0.0, rapid_motion=False,
            pose_class="side_near", body_in_bed_ratio=0.10,
            legacy_fall_score=55.0,
        ))
        self.assertEqual(result.phase, FusionPhase.SHADOW_ALERT)
        self.assertIn("retained_temporal_context", result.evidence)
        self.assertIn("pose_reacquired", result.evidence)

    def test_rapid_bed_departure_can_confirm_without_pose_class(self):
        fusion = HybridFusion(
            verify_after_sec=0.5, alert_after_sec=1.5,
            pose_dropout_grace_sec=1.5, rapid_bed_departure_sec=3.5,
        )
        in_bed = dict(
            body_in_bed_ratio=0.90, pose_class="sitting_edge",
            legacy_fall_score=0.0, rapid_motion=True,
        )
        fusion.update(sample(0.0, **in_bed))
        self.assertEqual(fusion.update(sample(0.6, **in_bed)).phase, FusionPhase.VERIFY)
        fusion.update(sample(0.8, primary_observed=False, **in_bed))
        result = fusion.update(sample(
            1.6, body_in_bed_ratio=0.10, pose_class="sitting_edge",
            legacy_fall_score=0.0, tcn_ready=False,
            tcn_candidate=False, tcn_probability=0.0, rapid_motion=False,
        ))
        self.assertEqual(result.phase, FusionPhase.SHADOW_ALERT)
        self.assertIn("rapid_bed_departure", result.evidence)

    def test_normal_bed_departure_without_dropout_does_not_alert(self):
        fusion = HybridFusion(
            verify_after_sec=0.5, alert_after_sec=1.5,
            rapid_bed_departure_sec=3.5,
        )
        in_bed = dict(
            body_in_bed_ratio=0.90, pose_class="sitting_edge",
            legacy_fall_score=0.0, rapid_motion=False,
        )
        fusion.update(sample(0.0, **in_bed))
        fusion.update(sample(0.6, **in_bed))
        result = fusion.update(sample(
            1.6, body_in_bed_ratio=0.10, pose_class="sitting_edge",
            legacy_fall_score=0.0, rapid_motion=False,
        ))
        self.assertEqual(result.phase, FusionPhase.SAFE)
        self.assertNotIn("rapid_bed_departure", result.evidence)

    def test_edge_lying_then_outside_is_one_fall_transition(self):
        fusion = HybridFusion(
            lying_departure_arm_sec=4.5,
            lying_departure_confirm_sec=2.0,
        )
        fusion.update(sample(
            0.0, body_in_bed_ratio=0.90, pose_class="sitting_edge",
            legacy_fall_score=0.0, rapid_motion=False,
        ))
        armed = fusion.update(sample(
            3.2, body_in_bed_ratio=0.53, pose_class="prone_back",
            legacy_fall_score=0.0, rapid_motion=False,
        ))
        self.assertEqual(armed.phase, FusionPhase.CANDIDATE)
        self.assertIn("bed_departure_lying_transition", armed.evidence)
        result = fusion.update(sample(
            4.7, body_in_bed_ratio=0.24, pose_class="sitting_edge",
            legacy_fall_score=0.0, rapid_motion=False,
            tcn_ready=False, tcn_candidate=False, tcn_probability=0.0,
        ))
        self.assertEqual(result.phase, FusionPhase.SHADOW_ALERT)
        self.assertIn("bed_departure_transition_confirmed", result.evidence)

    def test_edge_lying_transition_survives_short_pose_dropout(self):
        fusion = HybridFusion(
            lying_departure_arm_sec=4.5,
            lying_departure_confirm_sec=2.0,
        )
        fusion.update(sample(
            0.0, body_in_bed_ratio=0.90, pose_class="sitting_edge",
            legacy_fall_score=0.0, rapid_motion=False,
        ))
        fusion.update(sample(
            3.2, body_in_bed_ratio=0.53, pose_class="prone_back",
            legacy_fall_score=0.0, rapid_motion=False,
        ))
        missing = fusion.update(sample(3.6, primary_observed=False))
        self.assertEqual(missing.phase, FusionPhase.VERIFY)
        result = fusion.update(sample(
            4.7, body_in_bed_ratio=0.24, pose_class="sitting_edge",
            legacy_fall_score=0.0, rapid_motion=False,
            tcn_ready=False, tcn_candidate=False, tcn_probability=0.0,
        ))
        self.assertEqual(result.phase, FusionPhase.SHADOW_ALERT)

    def test_seated_bed_exit_does_not_arm_lying_transition(self):
        fusion = HybridFusion()
        fusion.update(sample(
            0.0, body_in_bed_ratio=0.90, pose_class="sitting_edge",
            legacy_fall_score=0.0, rapid_motion=False,
        ))
        fusion.update(sample(
            3.2, body_in_bed_ratio=0.53, pose_class="sitting_center",
            legacy_fall_score=0.0, rapid_motion=False,
        ))
        result = fusion.update(sample(
            4.0, body_in_bed_ratio=0.20, pose_class="sitting_edge",
            legacy_fall_score=0.0, rapid_motion=False,
        ))
        self.assertNotEqual(result.phase, FusionPhase.SHADOW_ALERT)
        self.assertNotIn("bed_departure_lying_transition", result.evidence)

    def test_direct_rapid_bed_departure_can_alert_without_pose_class(self):
        fusion = HybridFusion(direct_rapid_departure_sec=2.0)
        in_bed = dict(
            body_in_bed_ratio=0.83, pose_class="sitting_edge",
            legacy_fall_score=0.0, rapid_motion=True,
        )
        self.assertEqual(
            fusion.update(sample(0.0, **in_bed)).phase,
            FusionPhase.CANDIDATE,
        )
        result = fusion.update(sample(
            1.2, body_in_bed_ratio=0.24, pose_class="sitting_edge",
            legacy_fall_score=0.0, rapid_motion=True,
        ))
        self.assertEqual(result.phase, FusionPhase.SHADOW_ALERT)
        self.assertIn("direct_rapid_bed_departure_confirmed", result.evidence)

    def test_calm_seated_bed_departure_does_not_use_direct_rapid_route(self):
        fusion = HybridFusion(direct_rapid_departure_sec=2.0)
        fusion.update(sample(
            0.0, body_in_bed_ratio=0.90, pose_class="sitting_edge",
            legacy_fall_score=0.0, rapid_motion=False,
        ))
        result = fusion.update(sample(
            1.2, body_in_bed_ratio=0.20, pose_class="sitting_edge",
            legacy_fall_score=0.0, rapid_motion=False,
        ))
        self.assertNotEqual(result.phase, FusionPhase.SHADOW_ALERT)
        self.assertNotIn("direct_rapid_bed_departure", result.evidence)

    def test_dropout_outside_bed_without_recent_in_bed_does_not_alert(self):
        fusion = HybridFusion(
            verify_after_sec=0.5, alert_after_sec=1.5,
            pose_dropout_grace_sec=1.5,
        )
        outside = dict(
            body_in_bed_ratio=0.10, pose_class="sitting_edge",
            legacy_fall_score=0.0,
        )
        fusion.update(sample(0.0, **outside))
        fusion.update(sample(0.6, **outside))
        fusion.update(sample(0.8, primary_observed=False, **outside))
        result = fusion.update(sample(
            1.6, tcn_ready=False, tcn_candidate=False,
            tcn_probability=0.0, rapid_motion=False, **outside,
        ))
        self.assertEqual(result.phase, FusionPhase.VERIFY)
        self.assertNotIn("rapid_bed_departure", result.evidence)

    def test_unverified_dropout_is_not_retained(self):
        fusion = HybridFusion(
            verify_after_sec=0.5, pose_dropout_grace_sec=1.5,
        )
        self.assertEqual(fusion.update(sample(0.0)).phase, FusionPhase.CANDIDATE)
        result = fusion.update(sample(0.2, primary_observed=False))
        self.assertEqual(result.phase, FusionPhase.INSUFFICIENT)

    def test_dropout_grace_expires(self):
        fusion = HybridFusion(
            verify_after_sec=0.5, pose_dropout_grace_sec=1.0,
        )
        fusion.update(sample(0.0))
        fusion.update(sample(0.6))
        fusion.update(sample(0.8, primary_observed=False))
        result = fusion.update(sample(1.9, primary_observed=False))
        self.assertEqual(result.phase, FusionPhase.INSUFFICIENT)

    def test_track_change_during_dropout_resets_retained_incident(self):
        fusion = HybridFusion(verify_after_sec=0.5)
        fusion.update(sample(0.0))
        fusion.update(sample(0.6))
        fusion.update(sample(0.8, primary_observed=False))
        result = fusion.update(sample(0.9, track_id=8, primary_observed=False))
        self.assertEqual(result.phase, FusionPhase.INSUFFICIENT)

    def test_track_change_resets_candidate_timer(self):
        fusion = HybridFusion(verify_after_sec=0.5, alert_after_sec=1.5)
        fusion.update(sample(0.0))
        self.assertEqual(fusion.update(sample(1.0)).phase, FusionPhase.VERIFY)
        result = fusion.update(sample(1.1, track_id=8))
        self.assertEqual(result.phase, FusionPhase.CANDIDATE)
        self.assertEqual(result.candidate_age_sec, 0.0)

    def test_warmup_fallback_needs_three_signals(self):
        fusion = HybridFusion()
        result = fusion.update(sample(
            0.0, tcn_ready=False, tcn_candidate=False,
            tcn_probability=0.0, legacy_fall_score=80.0,
        ))
        self.assertEqual(result.phase, FusionPhase.CANDIDATE)

    def test_incident_during_tcn_warmup_is_explicitly_not_ready(self):
        fusion = HybridFusion()
        result = fusion.update(sample(
            0.0, tcn_ready=False, tcn_candidate=False,
            tcn_probability=0.0, legacy_fall_score=20.0,
            rapid_motion=True, pose_class="sitting_center",
            body_in_bed_ratio=0.7,
        ))
        self.assertEqual(result.phase, FusionPhase.TCN_NOT_READY)
        self.assertIn("tcn_not_ready", result.evidence)

    def test_calm_tcn_warmup_remains_warming(self):
        fusion = HybridFusion()
        result = fusion.update(sample(
            0.0, tcn_ready=False, tcn_candidate=False,
            tcn_probability=0.0, legacy_fall_score=5.0,
            rapid_motion=False, motion_ratio=0.0,
            pose_class="sitting_center", body_in_bed_ratio=0.9,
        ))
        self.assertEqual(result.phase, FusionPhase.WARMING)


if __name__ == "__main__":
    unittest.main()
