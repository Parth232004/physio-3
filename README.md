# Physio 3: Physiotherapy Movement Analysis

This project provides tools for analyzing physiotherapy sessions, focusing on joint movement phases and session scoring.

## Components

### MovementPhaseDetector (`movement_phase.py`)
A class for detecting movement phases ('raise', 'hold', 'lower') from joint angle data using a rolling window analysis.

- **Key Methods**:
  - `calculate_angle(landmark1, landmark2, landmark3)`: Calculates the angle at a joint.
  - `update_angle(angle)`: Updates the angle buffer.
  - `detect_phase()`: Detects the current phase based on velocity.

### AdvancedSessionScorer (`advanced_session_scoring.py`)
A class for scoring physiotherapy sessions based on stability, smoothness, and completion accuracy.

- **Key Methods**:
  - `update(angle, phase)`: Updates with new data.
  - `get_scores()`: Returns a dictionary of scores (stability, smoothness, completion_accuracy, overall).

## Demo

The `day3_demo.py` script provides a real-time physiotherapy session demo using webcam input.

### Running the Demo
1. Ensure Python 3, NumPy, OpenCV, and MediaPipe are installed.
2. Run: `python day3_demo.py`

This will open the webcam, detect pose, calculate elbow angles, detect movement phases, provide live feedback (phase, correction, severity, score delta), and display the final session scores. Press 'q' to quit.

## Requirements
- Python 3.x
- NumPy
- OpenCV
- MediaPipe

Install dependencies: `pip install numpy opencv-python mediapipe`

## Project Structure
- `movement_phase.py`: Movement phase detection logic.
- `advanced_session_scoring.py`: Session scoring logic.
- `day3_demo.py`: Demonstration script.
- `.gitignore`: Ignores temporary files.