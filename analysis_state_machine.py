"""
State machine for per-camera analysis pipeline.
Manages transitions: idle -> detecting -> analyzing -> tracking
"""
import enum
import time
from typing import Optional

class AnalysisState(enum.Enum):
    IDLE = "idle"              # No person detected, seg only
    DETECTING = "detecting"    # Person detected, collecting buffer
    ANALYZING = "analyzing"    # Buffer ready, running pose + 6-class
    TRACKING = "tracking"      # Continuous tracking, optimized FPS

class CameraAnalysisStateMachine:
    """State machine to manage analysis pipeline per camera.
    
    Reduces GPU/CPU load by:
    - Running seg only in IDLE
    - Buffering frames in DETECTING (CPU)
    - Running pose + 6-class only when buffer ready
    - Adapting FPS based on state
    """
    
    def __init__(self, buffer_ready_threshold: int = 3, idle_timeout_sec: float = 2.0):
        self.state = AnalysisState.IDLE
        self.buffer_ready_threshold = buffer_ready_threshold
        self.idle_timeout_sec = idle_timeout_sec
        self.state_enter_time = time.time()
        self.last_person_detected_time = time.time()
        self.frame_counter = 0
    
    def update(self, person_detected: bool, buffer_is_ready: bool) -> AnalysisState:
        """Update state based on person detection and buffer status.
        
        Returns current state after transition.
        """
        self.frame_counter += 1
        now = time.time()
        
        # IDLE -> DETECTING
        if self.state == AnalysisState.IDLE and person_detected:
            self.state = AnalysisState.DETECTING
            self.state_enter_time = now
            self.last_person_detected_time = now
            return self.state
        
        # IDLE: stay idle if no person, run low-freq seg
        if self.state == AnalysisState.IDLE and not person_detected:
            return self.state
        
        # DETECTING -> ANALYZING
        if self.state == AnalysisState.DETECTING and person_detected:
            self.last_person_detected_time = now

        if self.state == AnalysisState.DETECTING and buffer_is_ready:
            self.state = AnalysisState.ANALYZING
            self.state_enter_time = now
            self.last_person_detected_time = now
            return self.state
        
        # DETECTING: timeout if person disappears
        if self.state == AnalysisState.DETECTING and not person_detected:
            if now - self.last_person_detected_time > self.idle_timeout_sec:
                self.state = AnalysisState.IDLE
                return self.state
        
        # ANALYZING -> TRACKING (once buffer is full and pose consistently detected)
        if self.state == AnalysisState.ANALYZING and person_detected:
            self.state = AnalysisState.TRACKING
            self.last_person_detected_time = now
            return self.state

        # ANALYZING may be entered from a short detection buffer and then
        # lose the person before the next confirmed observation.  Previously
        # this state had no negative-observation branch and could remain
        # "analyzing" forever with person_count=0.  Apply the same bounded
        # dropout grace used by TRACKING, then return to the cheap IDLE mode.
        if self.state == AnalysisState.ANALYZING and not person_detected:
            if now - self.last_person_detected_time > self.idle_timeout_sec:
                self.state = AnalysisState.IDLE
            return self.state
        
        # TRACKING -> ANALYZING or IDLE (person lost)
        if self.state == AnalysisState.TRACKING:
            if person_detected:
                self.last_person_detected_time = now
                return self.state
            elif now - self.last_person_detected_time > self.idle_timeout_sec:
                self.state = AnalysisState.IDLE
                return self.state
            else:
                # Timeout grace period: stay in TRACKING momentarily
                return self.state
        
        return self.state
    
    def get_fps_target(self) -> int:
        """Return target FPS based on current state.
        
        - IDLE: 3 FPS (seg only, low load)
        - DETECTING: 10 FPS (collecting buffer)
        - ANALYZING: 20 FPS (pose + 6-class)
        - TRACKING: 15 FPS (continuous, optimized)
        """
        if self.state == AnalysisState.IDLE:
            return 3
        elif self.state == AnalysisState.DETECTING:
            return 10
        elif self.state == AnalysisState.ANALYZING:
            return 20
        else:  # TRACKING
            return 15
    
    def should_run_pose(self) -> bool:
        """Whether to run YOLO pose in this state."""
        return self.state in (AnalysisState.DETECTING, AnalysisState.ANALYZING, AnalysisState.TRACKING)
    
    def should_run_6class(self) -> bool:
        """Whether to run 6-class model in this state."""
        return self.state in (AnalysisState.ANALYZING, AnalysisState.TRACKING)
    
    def state_name(self) -> str:
        """Human-readable state name."""
        return self.state.value
    
    def time_in_state(self) -> float:
        """Seconds spent in current state."""
        return time.time() - self.state_enter_time
