# Physio-3 - Enhanced Rehabilitation Tracking System

## Overview
Physio-3 is an enhanced computer vision-based rehabilitation tracking system that builds upon Physio-2 with additional features for multi-exercise continuity, medical safety validation, structured output interfaces, and robust error handling.

## What's New (Missing Features Implemented)

### 1. Multi-Exercise Support with Exercise Awareness
- **Exercise Type Enum**: Supports 10+ exercise types (elbow flexion, shoulder abduction, wrist flexion/extension, knee flexion/extension, hip flexion/extension)
- **Exercise Continuity Tracker**: Tracks exercise transitions, repetitions, and session duration
- **Auto-detection**: Can detect exercise type from angle patterns

### 2. Medical Threshold Validation
- **MedicalThresholds Class**: Defines safe/unsafe thresholds for:
  - Velocity limits (max 15°/frame)
  - Hold phase duration (10-300 frames)
  - Range of motion percentages
  - Spasticity detection (velocity spikes >30°/frame)
  - Tremor detection (oscillation >2°)

- **MedicalThresholdValidator**: Validates movements against:
  - Safe velocity ranges
  - Joint angle limits
  - Hold duration requirements
  - ROM percentage calculations
  - Spasticity patterns

### 3. Structured Output Interface for VR/Unreal Adapter
- **OutputFormat Enum**: Supports JSON and dict output formats
- **StructuredOutputInterface Class**:
  - Session management (start/end sessions)
  - Phase recording with timestamps
  - Safety alert logging
  - VR/Unreal adapter output format (`get_vr_adapter_data()`)
  - JSON export (`to_json()`)

### 4. Error Handling for Pose Dropouts & Noisy Frames
- **LandmarkNoiseFilter**: Median-based noise filtering with configurable thresholds
- **PoseDropoutHandler**: Tracks consecutive dropouts with recovery detection
- **LandmarkQuality Assessment**: Visibility and confidence scoring

## File Structure

```
physio-3/
├── movement_phase.py           # Original phase detection
├── advanced_session_scoring.py  # Original scoring system
├── day3_demo.py               # Enhanced demo with all features
├── enhanced_physio.py         # NEW: All missing functionality
└── README.md                  # This file
```

## Usage

### Basic Demo
```bash
python day3_demo.py
```

### Key Classes (in enhanced_physio.py)

```python
from enhanced_physio import (
    ExerciseType,              # Exercise type enumeration
    MedicalThresholds,         # Medical safety thresholds
    MedicalThresholdValidator, # Threshold validation
    StructuredOutputInterface, # Output interface for VR
    LandmarkNoiseFilter,       # Noise filtering
    PoseDropoutHandler,        # Dropout handling
    ExerciseContinuityTracker  # Multi-exercise continuity
)
```

### Example: Medical Validation
```python
validator = MedicalThresholdValidator()
is_safe, warning = validator.validate_velocity(velocity, phase)
is_safe, warning = validator.validate_angle(angle, "elbow_flexion")
```

### Example: Structured Output
```python
output = StructuredOutputInterface(OutputFormat.JSON)
output.start_session("elbow_flexion")
output.record_phase(phase_data, scores)
json_output = output.export_current_session()
vr_data = output.get_vr_adapter_data()
```

### Example: Error Handling
```python
noise_filter = LandmarkNoiseFilter(window_size=5, noise_threshold=5.0)
dropout_handler = PoseDropoutHandler()

# Process frame
filtered_x, filtered_y, quality = noise_filter.filter_position(x, y, visibility)
is_tracking, status = dropout_handler.register_detection()
```

## Medical Thresholds Reference

| Parameter | Value | Description |
|-----------|-------|-------------|
| max_safe_velocity | 15.0 °/frame | Maximum safe movement velocity |
| min_movement_velocity | 0.5 °/frame | Minimum velocity to detect movement |
| min_hold_duration | 10 frames | Minimum valid hold phase |
| max_hold_duration | 300 frames | Maximum hold before fatigue warning |
| spasticity_velocity_spike | 30.0 °/frame | Velocity spike threshold for spasticity |
| tremor_threshold | 2.0 ° | Tremor detection threshold |

## Exercise Types Supported

1. ELBOW_FLEXION
2. SHOULDER_ABDUCTION
3. SHOULDER_FLEXION
4. WRIST_FLEXION
5. WRIST_EXTENSION
6. KNEE_FLEXION
7. KNEE_EXTENSION
8. HIP_FLEXION
9. HIP_EXTENSION
10. UNKNOWN

## Safety Alerts

The system generates safety alerts for:
- **tracking_lost**: Pose detection failure
- **noise_filtered**: Landmark outliers filtered
- **unsafe_angle**: Angle outside safe range
- **unsafe_velocity**: Movement too fast
- **hold_duration**: Hold too short/long

## VR/Unreal Adapter Output Format

```json
{
  "session_id": "session_1234567890",
  "exercise_type": "elbow_flexion",
  "repetition_count": 5,
  "current_phase": "hold",
  "scores": {
    "overall": 82.5,
    "stability": 85.0,
    "smoothness": 80.0
  },
  "alerts": [
    {"type": "unsafe_velocity", "severity": "medium"}
  ],
  "timestamp": 1234567890.123
}
```

## Dependencies
- OpenCV
- MediaPipe
- NumPy
- Python 3.7+

## License
MIT License
