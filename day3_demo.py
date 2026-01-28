import numpy as np
from movement_phase import MovementPhaseDetector
from advanced_session_scoring import AdvancedSessionScorer

def simulate_session():
    """
    Simulate a physiotherapy session with joint angle data.
    Returns a list of angles representing the movement over time.
    """
    angles = []

    # Raise phase: from 0 to 90 degrees over 20 frames
    for i in range(20):
        angle = (i / 19) * 90
        angles.append(angle)

    # Hold phase: stay at 90 degrees for 15 frames
    for _ in range(15):
        angles.append(90)

    # Lower phase: from 90 to 0 degrees over 20 frames
    for i in range(20):
        angle = 90 - (i / 19) * 90
        angles.append(angle)

    return angles

def main():
    # Initialize detectors
    phase_detector = MovementPhaseDetector(window_size=10, velocity_threshold=2.0)
    scorer = AdvancedSessionScorer()

    # Simulate session
    angles = simulate_session()

    detected_phases = []

    # Process each angle
    for angle in angles:
        phase_detector.update_angle(angle)
        phase = phase_detector.detect_phase()
        if phase:
            detected_phases.append(phase)
            scorer.update(angle, phase)

    # Get scores
    scores = scorer.get_scores()

    print("Session Simulation Complete")
    print(f"Total frames: {len(angles)}")
    print(f"Detected phases: {detected_phases[:10]}...")  # Show first 10
    print("Scores:")
    for key, value in scores.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()