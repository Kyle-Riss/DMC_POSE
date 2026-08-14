import unittest
from unittest.mock import patch

from analysis_state_machine import AnalysisState, CameraAnalysisStateMachine


class AnalysisStateMachineTests(unittest.TestCase):
    def test_analyzing_lost_person_returns_to_idle_after_grace(self):
        machine = CameraAnalysisStateMachine(
            buffer_ready_threshold=1,
            idle_timeout_sec=2.0,
        )
        with patch("analysis_state_machine.time.time", return_value=100.0):
            machine.state = AnalysisState.ANALYZING
            machine.last_person_detected_time = 100.0
            self.assertEqual(
                machine.update(False, False),
                AnalysisState.ANALYZING,
            )
        with patch("analysis_state_machine.time.time", return_value=102.1):
            self.assertEqual(machine.update(False, False), AnalysisState.IDLE)

    def test_analyzing_observation_still_enters_tracking(self):
        machine = CameraAnalysisStateMachine()
        machine.state = AnalysisState.ANALYZING
        self.assertEqual(machine.update(True, True), AnalysisState.TRACKING)


if __name__ == "__main__":
    unittest.main()
