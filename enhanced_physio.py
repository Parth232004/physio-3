"""
Exercise Type Enumeration

Defines supported exercise types for multi-exercise continuity tracking.
"""
from enum import Enum


class ExerciseType(Enum):
    """Supported exercise types for physio rehabilitation."""
    ELBOW_FLEXION = "elbow_flexion"
    SHOULDER_ABDUCTION = "shoulder_abduction"
    SHOULDER_FLEXION = "shoulder_flexion"
    WRIST_FLEXION = "wrist_flexion"
    WRIST_EXTENSION = "wrist_extension"
    KNEE_FLEXION = "knee_flexion"
    KNEE_EXTENSION = "knee_extension"
    HIP_FLEXION = "hip_flexion"
    HIP_EXTENSION = "hip_extension"
    UNKNOWN = "unknown"


"""
Medical Thresholds Module

Defines safe/unsafe medical thresholds for neuro-reflex and rehab constraints.
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class MedicalThresholds:
    """Medical thresholds for safe exercise execution."""
    
    # Velocity thresholds (degrees per frame)
    max_safe_velocity: float = 15.0
    min_movement_velocity: float = 0.5
    
    # Angle thresholds (degrees) - exercise specific
    min_angle: float = 0.0
    max_angle: float = 180.0
    
    # Hold phase duration thresholds (frames)
    min_hold_duration: int = 10  # Minimum frames for valid hold
    max_hold_duration: int = 300  # Maximum frames before fatigue warning
    
    # Rest period between repetitions (frames)
    min_rest_period: int = 5
    
    # Spasticity detection thresholds
    spasticity_velocity_spike: float = 30.0  # Sudden velocity spike
    spasticity_angle_change: float = 45.0  # Rapid angle change
    
    # Tremor detection (noise threshold)
    tremor_threshold: float = 2.0  # Degrees of oscillation
    
    # Range of motion percentage thresholds
    min_rom_percentage: float = 20.0  # Minimum ROM required
    max_rom_percentage: float = 95.0  # Maximum ROM cap for safety


class MedicalThresholdValidator:
    """Validates movement against medical safety thresholds."""
    
    def __init__(self, thresholds: Optional[MedicalThresholds] = None):
        """Initialize validator with medical thresholds.
        
        Args:
            thresholds: Custom medical thresholds. Uses defaults if None.
        """
        self.thresholds = thresholds or MedicalThresholds()
    
    def validate_velocity(self, velocity: float, phase: str) -> Tuple[bool, str]:
        """Validate movement velocity against safe thresholds.
        
        Args:
            velocity: Current velocity in degrees per frame.
            phase: Current movement phase.
            
        Returns:
            Tuple of (is_safe, warning_message).
        """
        if abs(velocity) > self.thresholds.max_safe_velocity:
            return False, f"Velocity too high ({abs(velocity):.1f}°/frame). Slow down movement."
        
        if phase == 'raise' or phase == 'lower':
            if abs(velocity) < self.thresholds.min_movement_velocity:
                return False, "Movement too slow. Increase pace."
        
        return True, ""
    
    def validate_angle(self, angle: float, exercise_type: str) -> Tuple[bool, str]:
        """Validate joint angle against safe range.
        
        Args:
            angle: Current joint angle in degrees.
            exercise_type: Type of exercise being performed.
            
        Returns:
            Tuple of (is_safe, warning_message).
        """
        if angle < self.thresholds.min_angle:
            return False, f"Angle below minimum ({self.thresholds.min_angle}°). Check joint limit."
        
        if angle > self.thresholds.max_angle:
            return False, f"Angle exceeds maximum ({self.thresholds.max_angle}°). Reduce range."
        
        return True, ""
    
    def detect_spasticity(self, velocity_history: list) -> Tuple[bool, str]:
        """Detect potential spasticity from velocity patterns.
        
        Args:
            velocity_history: Recent velocity measurements.
            
        Returns:
            Tuple of (is_spastic, warning_message).
        """
        if len(velocity_history) < 3:
            return False, ""
        
        recent_velocities = velocity_history[-3:]
        velocity_spikes = sum(1 for v in recent_velocities 
                             if abs(v) > self.thresholds.spasticity_velocity_spike)
        
        if velocity_spikes >= 2:
            return True, "Warning: Rapid velocity changes detected. Possible spasticity."
        
        return False, ""
    
    def validate_hold_duration(self, hold_frames: int) -> Tuple[bool, str]:
        """Validate hold phase duration.
        
        Args:
            hold_frames: Number of frames in hold phase.
            
        Returns:
            Tuple of (is_valid, warning_message).
        """
        if hold_frames < self.thresholds.min_hold_duration:
            return False, f"Hold too short ({hold_frames} frames). Maintain for at least {self.thresholds.min_hold_duration}."
        
        if hold_frames > self.thresholds.max_hold_duration:
            return False, f"Hold too long ({hold_frames} frames). Risk of fatigue. Rest recommended."
        
        return True, ""
    
    def calculate_rom_percentage(self, min_angle: float, max_angle: float, 
                                  exercise_type: str) -> float:
        """Calculate range of motion as percentage of normal.
        
        Args:
            min_angle: Minimum angle achieved.
            max_angle: Maximum angle achieved.
            exercise_type: Type of exercise.
            
        Returns:
            ROM percentage (0-100).
        """
        normal_rom = self._get_normal_rom(exercise_type)
        achieved_rom = max_angle - min_angle
        
        rom_percentage = (achieved_rom / normal_rom) * 100
        return min(100.0, rom_percentage)
    
    def _get_normal_rom(self, exercise_type: str) -> float:
        """Get normal range of motion for exercise type."""
        normal_roms = {
            'elbow_flexion': 150.0,
            'shoulder_abduction': 180.0,
            'shoulder_flexion': 180.0,
            'wrist_flexion': 80.0,
            'wrist_extension': 70.0,
            'knee_flexion': 135.0,
            'knee_extension': 0.0,
            'hip_flexion': 120.0,
            'hip_extension': 30.0,
        }
        return normal_roms.get(exercise_type, 150.0)


"""
Structured Output Interface Module

Provides JSON output format for downstream consumers (VR / Unreal adapter).
"""
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from enum import Enum


class OutputFormat(Enum):
    """Supported output formats."""
    JSON = "json"
    DICT = "dict"


@dataclass
class PhaseData:
    """Phase detection data."""
    phase: str
    confidence: float
    velocity: float
    timestamp: float


@dataclass
class SafetyAlert:
    """Safety alert data."""
    alert_type: str
    severity: str
    message: str
    timestamp: float


@dataclass
class ExerciseSession:
    """Complete exercise session data."""
    session_id: str
    exercise_type: str
    start_time: float
    end_time: Optional[float]
    repetition_count: int
    phases: List[Dict]
    safety_alerts: List[Dict]
    scores: Dict[str, float]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'session_id': self.session_id,
            'exercise_type': self.exercise_type,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'repetition_count': self.repetition_count,
            'phases': self.phases,
            'safety_alerts': self.safety_alerts,
            'scores': self.scores,
            'metadata': self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class StructuredOutputInterface:
    """Structured output interface for downstream consumers."""
    
    def __init__(self, output_format: OutputFormat = OutputFormat.JSON):
        """Initialize interface.
        
        Args:
            output_format: Desired output format.
        """
        self.output_format = output_format
        self.session_id = f"session_{int(time.time())}"
        self.sessions: List[ExerciseSession] = []
        self.current_session: Optional[ExerciseSession] = None
    
    def start_session(self, exercise_type: str, metadata: Optional[Dict] = None):
        """Start a new exercise session.
        
        Args:
            exercise_type: Type of exercise being performed.
            metadata: Optional session metadata.
        """
        self.current_session = ExerciseSession(
            session_id=self.session_id,
            exercise_type=exercise_type,
            start_time=time.time(),
            end_time=None,
            repetition_count=0,
            phases=[],
            safety_alerts=[],
            scores={},
            metadata=metadata or {}
        )
    
    def record_phase(self, phase_data: PhaseData, scores: Dict[str, float]):
        """Record a phase update.
        
        Args:
            phase_data: Current phase data.
            scores: Current session scores.
        """
        if not self.current_session:
            return
        
        phase_dict = {
            'phase': phase_data.phase,
            'confidence': phase_data.confidence,
            'velocity': phase_data.velocity,
            'timestamp': phase_data.timestamp
        }
        self.current_session.phases.append(phase_dict)
        self.current_session.scores = scores
        
        # Count repetitions (raise->hold->lower cycle)
        if phase_data.phase == 'lower':
            self.current_session.repetition_count += 1
    
    def record_safety_alert(self, alert: SafetyAlert):
        """Record a safety alert.
        
        Args:
            alert: Safety alert data.
        """
        if not self.current_session:
            return
        
        alert_dict = {
            'alert_type': alert.alert_type,
            'severity': alert.severity,
            'message': alert.message,
            'timestamp': alert.timestamp
        }
        self.current_session.safety_alerts.append(alert_dict)
    
    def end_session(self) -> ExerciseSession:
        """End current session and return data.
        
        Returns:
            Completed exercise session data.
        """
        if self.current_session:
            self.current_session.end_time = time.time()
            self.sessions.append(self.current_session)
            session = self.current_session
            self.current_session = None
            return session
        return None
    
    def export_current_session(self) -> Any:
        """Export current session data in specified format."""
        session = self.current_session or self.sessions[-1] if self.sessions else None
        if not session:
            return None
        
        if self.output_format == OutputFormat.JSON:
            return session.to_json()
        else:
            return session.to_dict()
    
    def get_vr_adapter_data(self) -> Dict:
        """Get data formatted for VR/Unreal Engine adapter.
        
        Returns:
            Dictionary optimized for VR/Unreal consumption.
        """
        if not self.current_session:
            return {}
        
        return {
            'session_id': self.current_session.session_id,
            'exercise_type': self.current_session.exercise_type,
            'repetition_count': self.current_session.repetition_count,
            'current_phase': self.current_session.phases[-1]['phase'] if self.current_session.phases else None,
            'scores': self.current_session.scores,
            'alerts': [{'type': a['alert_type'], 'severity': a['severity']} 
                      for a in self.current_session.safety_alerts],
            'timestamp': time.time()
        }


"""
Error Handling Module

Handles pose dropouts and noisy landmark frames.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from collections import deque


@dataclass
class LandmarkQuality:
    """Landmark quality assessment."""
    visibility: float
    confidence: float
    is_valid: bool
    warning: Optional[str]


class LandmarkNoiseFilter:
    """Filters noisy landmark measurements."""
    
    def __init__(self, window_size: int = 5, noise_threshold: float = 5.0):
        """Initialize noise filter.
        
        Args:
            window_size: Number of frames for smoothing window.
            noise_threshold: Maximum allowed deviation from median.
        """
        self.window_size = window_size
        self.noise_threshold = noise_threshold
        self.position_buffer = deque(maxlen=window_size * 2)
    
    def filter_position(self, x: float, y: float, 
                       visibility: float) -> Tuple[float, float, LandmarkQuality]:
        """Filter landmark position using median smoothing.
        
        Args:
            x: Raw x coordinate.
            y: Raw y coordinate.
            visibility: Landmark visibility score (0-1).
            
        Returns:
            Tuple of (filtered_x, filtered_y, quality assessment).
        """
        # Check visibility
        if visibility < 0.3:
            return x, y, LandmarkQuality(
                visibility=visibility,
                confidence=visibility,
                is_valid=False,
                warning="Low visibility landmark"
            )
        
        # Add to buffer
        self.position_buffer.append((x, y))
        
        if len(self.position_buffer) < 3:
            return x, y, LandmarkQuality(
                visibility=visibility,
                confidence=visibility,
                is_valid=True,
                warning=None
            )
        
        # Calculate median filter
        positions = list(self.position_buffer)
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        
        x_median = np.median(x_coords)
        y_median = np.median(y_coords)
        
        # Check deviation from median
        x_deviation = abs(x - x_median)
        y_deviation = abs(y - y_median)
        
        if x_deviation > self.noise_threshold or y_deviation > self.noise_threshold:
            # Return median instead of noisy value
            return x_median, y_median, LandmarkQuality(
                visibility=visibility,
                confidence=0.5,
                is_valid=True,
                warning="Outlier filtered"
            )
        
        return x, y, LandmarkQuality(
            visibility=visibility,
            confidence=visibility,
            is_valid=True,
            warning=None
        )


class PoseDropoutHandler:
    """Handles pose detection dropouts."""
    
    def __init__(self, max_consecutive_dropouts: int = 5,
                 recovery_frames: int = 3):
        """Initialize dropout handler.
        
        Args:
            max_consecutive_dropouts: Maximum dropouts before marking as lost.
            recovery_frames: Frames required to confirm recovery.
        """
        self.max_consecutive_dropouts = max_consecutive_dropouts
        self.recovery_frames = recovery_frames
        self.consecutive_dropouts = 0
        self.recovery_counter = 0
        self.is_tracking = True
        self.dropout_history: List[bool] = []
    
    def register_detection(self) -> Tuple[bool, str]:
        """Register successful detection.
        
        Returns:
            Tuple of (is_tracking, status_message).
        """
        if not self.is_tracking:
            self.recovery_counter += 1
            if self.recovery_counter >= self.recovery_frames:
                self.is_tracking = True
                self.recovery_counter = 0
                return True, "Tracking recovered"
        
        self.consecutive_dropouts = 0
        self.dropout_history.append(True)
        return True, "Tracking active"
    
    def register_dropout(self) -> Tuple[bool, str]:
        """Register detection dropout.
        
        Returns:
            Tuple of (is_tracking, status_message).
        """
        self.consecutive_dropouts += 1
        self.dropout_history.append(False)
        
        if self.consecutive_dropouts >= self.max_consecutive_dropouts:
            self.is_tracking = False
            return False, f"Pose lost after {self.consecutive_dropouts} consecutive dropouts"
        
        return False, f"Transient dropout ({self.consecutive_dropouts}/{self.max_consecutive_dropouts})"
    
    def get_status(self) -> Dict:
        """Get current handler status.
        
        Returns:
            Status dictionary.
        """
        return {
            'is_tracking': self.is_tracking,
            'consecutive_dropouts': self.consecutive_dropouts,
            'recovery_counter': self.recovery_counter,
            'dropout_rate': sum(1 for d in self.dropout_history[-30:] if not d) / 30 if len(self.dropout_history) >= 30 else 0
        }


"""
Exercise Continuity Tracker

Provides multi-exercise awareness and continuity tracking.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict


class ContinuityState(Enum):
    """States for exercise continuity tracking."""
    IDLE = "idle"
    ACTIVE = "active"
    RESTING = "resting"
    COMPLETED = "completed"
    TRANSITIONING = "transitioning"


@dataclass
class ExerciseTransition:
    """Exercise transition data."""
    from_exercise: str
    to_exercise: str
    timestamp: float
    completed_repetitions: int
    transition_reason: str


class ExerciseContinuityTracker:
    """Tracks exercise continuity across multiple exercises."""
    
    def __init__(self):
        """Initialize continuity tracker."""
        self.current_exercise: Optional[ExerciseType] = None
        self.continuity_state = ContinuityState.IDLE
        self.exercise_history: Dict[ExerciseType, Dict] = defaultdict(lambda: {
            'start_time': None,
            'end_time': None,
            'repetitions': 0,
            'total_duration': 0,
            'transitions': []
        })
        self.transitions: List[ExerciseTransition] = []
        self.last_phase_time: Dict[str, float] = {}
        self.exercise_start_time: Optional[float] = None
    
    def start_exercise(self, exercise_type: ExerciseType):
        """Start a new exercise.
        
        Args:
            exercise_type: Type of exercise to start.
        """
        import time
        
        # Complete current exercise if any
        if self.current_exercise and self.current_exercise != exercise_type:
            self.end_exercise("transition")
        
        if self.current_exercise is None:
            self.exercise_start_time = time.time()
        
        self.current_exercise = exercise_type
        self.continuity_state = ContinuityState.ACTIVE
        
        if self.exercise_history[exercise_type]['start_time'] is None:
            self.exercise_history[exercise_type]['start_time'] = time.time()
    
    def end_exercise(self, reason: str = "completed"):
        """End current exercise.
        
        Args:
            reason: Reason for ending (transition, completed, interrupted).
        """
        import time
        
        if self.current_exercise:
            self.exercise_history[self.current_exercise]['end_time'] = time.time()
            self.exercise_history[self.current_exercise]['total_duration'] = (
                self.exercise_history[self.current_exercise]['end_time'] -
                self.exercise_history[self.current_exercise]['start_time']
            )
            self.current_exercise = None
            self.continuity_state = ContinuityState.RESTING
    
    def record_repetition(self, exercise_type: Optional[ExerciseType] = None):
        """Record a completed repetition.
        
        Args:
            exercise_type: Exercise type (uses current if None).
        """
        ex = exercise_type or self.current_exercise
        if ex:
            self.exercise_history[ex]['repetitions'] += 1
    
    def record_phase(self, phase: str, exercise_type: Optional[ExerciseType] = None):
        """Record a phase transition.
        
        Args:
            phase: Current phase name.
            exercise_type: Exercise type (uses current if None).
        """
        import time
        ex = exercise_type or self.current_exercise
        if ex:
            current_time = time.time()
            
            # Detect exercise type from phase patterns if needed
            if self.current_exercise is None:
                self.start_exercise(ex)
            
            self.last_phase_time[phase] = current_time
    
    def get_continuity_report(self) -> Dict:
        """Get continuity report for all exercises.
        
        Returns:
            Continuity report dictionary.
        """
        return {
            'current_exercise': self.current_exercise.value if self.current_exercise else None,
            'continuity_state': self.continuity_state.value,
            'exercise_history': {
                ex.value: {
                    'repetitions': data['repetitions'],
                    'duration': data['total_duration']
                } for ex, data in self.exercise_history.items()
            },
            'total_transitions': len(self.transitions),
            'session_duration': (
                self.exercise_history[self.current_exercise]['end_time'] - self.exercise_start_time
                if self.exercise_start_time and self.current_exercise else 0
            )
        }
    
    def detect_exercise_from_angles(self, elbow_angle: float, 
                                    shoulder_angle: float) -> ExerciseType:
        """Auto-detect exercise type from angle patterns.
        
        Args:
            elbow_angle: Current elbow angle.
            shoulder_angle: Current shoulder angle.
            
        Returns:
            Detected exercise type.
        """
        # Simple heuristics for exercise detection
        if shoulder_angle < 30 and elbow_angle > 150:
            return ExerciseType.SHOULDER_FLEXION
        elif shoulder_angle > 80 and shoulder_angle < 100:
            return ExerciseType.SHOULDER_ABDUCTION
        elif elbow_angle < 45:
            return ExerciseType.ELBOW_FLEXION
        else:
            return ExerciseType.ELBOW_FLEXION  # Default


"""
Enhanced Movement Phase Detector

Combines all modules for comprehensive movement detection.
"""
import time
import numpy as np
from collections import deque
from typing import Optional, Tuple


class EnhancedMovementPhaseDetector:
    """Enhanced movement phase detector with all missing features."""
    
    def __init__(self, window_size: int = 10, velocity_threshold: float = 5.0):
        """Initialize enhanced detector.
        
        Args:
            window_size: Rolling window size for analysis.
            velocity_threshold: Velocity threshold for phase detection.
        """
        # Core phase detection
        self.window_size = window_size
        self.velocity_threshold = velocity_threshold
        self.angle_buffer = deque(maxlen=window_size)
        self.velocity_buffer = deque(maxlen=window_size)
        self.previous_phase = None
        
        # Error handling
        self.noise_filter = LandmarkNoiseFilter()
        self.dropout_handler = PoseDropoutHandler()
        
        # Medical validation
        self.medical_validator = MedicalThresholdValidator()
        
        # Output interface
        self.output_interface = StructuredOutputInterface()
        
        # Continuity tracking
        self.continuity_tracker = ExerciseContinuityTracker()
        
        # Safety tracking
        self.safety_alerts: List[SafetyAlert] = []
        self.hold_frame_count = 0
        self.is_initialized = False
    
    def initialize_exercise(self, exercise_type: str):
        """Initialize for specific exercise.
        
        Args:
            exercise_type: Type of exercise.
        """
        # Handle string to ExerciseType conversion
        exercise_type_map = {
            'elbow_flexion': ExerciseType.ELBOW_FLEXION,
            'shoulder_abduction': ExerciseType.SHOULDER_ABDUCTION,
            'shoulder_flexion': ExerciseType.SHOULDER_FLEXION,
            'wrist_flexion': ExerciseType.WRIST_FLEXION,
            'wrist_extension': ExerciseType.WRIST_EXTENSION,
            'knee_flexion': ExerciseType.KNEE_FLEXION,
            'knee_extension': ExerciseType.KNEE_EXTENSION,
            'hip_flexion': ExerciseType.HIP_FLEXION,
            'hip_extension': ExerciseType.HIP_EXTENSION,
        }
        ex_type = exercise_type_map.get(exercise_type, ExerciseType.UNKNOWN)
        self.continuity_tracker.start_exercise(ex_type)
        self.output_interface.start_session(exercise_type)
        self.is_initialized = True
    
    def process_frame(self, landmarks: Dict, visibility_threshold: float = 0.3) -> Tuple[Optional[str], List[SafetyAlert]]:
        """Process a new frame with full error handling.
        
        Args:
            landmarks: Dictionary of landmark positions.
            visibility_threshold: Minimum visibility for valid detection.
            
        Returns:
            Tuple of (current_phase, safety_alerts).
        """
        alerts = []
        
        # Check for pose dropout
        is_tracking, status = self.dropout_handler.register_detection()
        if not is_tracking:
            alerts.append(SafetyAlert(
                alert_type="tracking_lost",
                severity="high",
                message=status,
                timestamp=time.time()
            ))
            return None, alerts
        
        # Filter noisy landmarks
        filtered_landmarks = {}
        for name, (x, y, vis) in landmarks.items():
            fx, fy, quality = self.noise_filter.filter_position(x, y, vis)
            filtered_landmarks[name] = (fx, fy)
            if quality.warning:
                alerts.append(SafetyAlert(
                    alert_type="noise_filtered",
                    severity="low",
                    message=f"{name}: {quality.warning}",
                    timestamp=time.time()
                ))
        
        # Calculate angle
        angle = self._calculate_angle_from_landmarks(filtered_landmarks)
        
        # Validate against medical thresholds
        is_safe, warning = self.medical_validator.validate_angle(angle, 
                                                                 self.output_interface.current_session.exercise_type 
                                                                 if self.output_interface.current_session else "unknown")
        if not is_safe:
            alerts.append(SafetyAlert(
                alert_type="unsafe_angle",
                severity="high",
                message=warning,
                timestamp=time.time()
            ))
        
        # Update phase detection
        self.angle_buffer.append(angle)
        phase = self._detect_phase()
        
        if phase:
            # Calculate velocity
            if len(self.angle_buffer) >= 2:
                velocity = list(self.angle_buffer)[-1] - list(self.angle_buffer)[-2]
                self.velocity_buffer.append(velocity)
                
                # Validate velocity
                is_safe, warning = self.medical_validator.validate_velocity(velocity, phase)
                if not is_safe:
                    alerts.append(SafetyAlert(
                        alert_type="unsafe_velocity",
                        severity="medium",
                        message=warning,
                        timestamp=time.time()
                    ))
            
            # Track hold duration
            if phase == 'hold':
                self.hold_frame_count += 1
                is_valid, warning = self.medical_validator.validate_hold_duration(self.hold_frame_count)
                if not is_valid:
                    alerts.append(SafetyAlert(
                        alert_type="hold_duration",
                        severity="low",
                        message=warning,
                        timestamp=time.time()
                    ))
            else:
                self.hold_frame_count = 0
            
            # Update continuity
            self.continuity_tracker.record_phase(phase)
            
            # Record phase in output interface
            phase_data = PhaseData(
                phase=phase,
                confidence=self._calculate_confidence(),
                velocity=velocity if self.velocity_buffer else 0,
                timestamp=time.time()
            )
            scores = self._calculate_current_scores()
            self.output_interface.record_phase(phase_data, scores)
        
        # Record all alerts
        for alert in alerts:
            self.output_interface.record_safety_alert(alert)
            self.safety_alerts.append(alert)
        
        return phase, alerts
    
    def _calculate_angle_from_landmarks(self, landmarks: Dict) -> float:
        """Calculate angle from landmark positions.
        
        Args:
            landmarks: Dictionary of landmark positions.
            
        Returns:
            Calculated angle in degrees.
        """
        # Simplified calculation - would use actual landmark indices
        if len(landmarks) >= 3:
            points = list(landmarks.values())
            v1 = np.array([points[0][0] - points[1][0], points[0][1] - points[1][1]])
            v2 = np.array([points[2][0] - points[1][0], points[2][1] - points[1][1]])
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        return 90.0  # Default
    
    def _detect_phase(self) -> Optional[str]:
        """Detect current phase."""
        if len(self.angle_buffer) < self.window_size:
            return None
        
        angles = list(self.angle_buffer)
        velocities = [angles[i] - angles[i-1] for i in range(1, len(angles))]
        avg_velocity = np.mean(velocities)
        
        if avg_velocity < -self.velocity_threshold:
            phase = 'raise'
        elif avg_velocity > self.velocity_threshold:
            phase = 'lower'
        else:
            phase = 'hold'
        
        self.previous_phase = phase
        return phase
    
    def _calculate_confidence(self) -> float:
        """Calculate detection confidence."""
        if len(self.angle_buffer) < self.window_size:
            return 0.0
        angles = list(self.angle_buffer)
        variance = np.var(angles)
        return max(0, min(1, 1 - (variance / 100)))
    
    def _calculate_current_scores(self) -> Dict[str, float]:
        """Calculate current session scores."""
        # Placeholder - would integrate with AdvancedSessionScorer
        return {'overall': 75.0, 'stability': 80.0, 'smoothness': 70.0}
    
    def get_session_data(self) -> Dict:
        """Get complete session data for export."""
        session = self.output_interface.end_session()
        if session:
            return session.to_dict()
        return {}
    
    def get_vr_adapter_output(self) -> Dict:
        """Get data formatted for VR adapter."""
        return self.output_interface.get_vr_adapter_data()
    
    def get_continuity_report(self) -> Dict:
        """Get exercise continuity report."""
        return self.continuity_tracker.get_continuity_report()
