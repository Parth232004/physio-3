"""
Day 3 Demo - Enhanced Physio-3 with Multi-Exercise Support

Demonstrates:
1. Multi-exercise awareness with continuity tracking
2. Medical threshold validation for safe/unsafe rehab constraints
3. Structured output interface for VR/Unreal adapter
4. Error handling for pose dropouts and noisy landmark frames
"""
import cv2
import mediapipe as mp
import numpy as np
from movement_phase import MovementPhaseDetector
from advanced_session_scoring import AdvancedSessionScorer
from enhanced_physio import (
    ExerciseType, MedicalThresholds, MedicalThresholdValidator,
    StructuredOutputInterface, OutputFormat, SafetyAlert,
    LandmarkNoiseFilter, PoseDropoutHandler, PhaseData,
    ExerciseContinuityTracker, ContinuityState
)


def calculate_angle(landmark1, landmark2, landmark3):
    """
    Calculate the angle at landmark2 formed by landmark1-landmark2-landmark3.
    """
    v1 = np.array([landmark1.x - landmark2.x, landmark1.y - landmark2.y])
    v2 = np.array([landmark3.x - landmark2.x, landmark3.y - landmark2.y])

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1, 1)

    angle = np.arccos(cos_angle) * 180 / np.pi
    return angle


def get_correction(phase, angle):
    """
    Provide correction based on phase and angle.
    """
    if phase == 'raise':
        if angle < 45:
            return 'Raise arm higher'
        elif angle > 135:
            return 'Don\'t overextend'
    elif phase == 'hold':
        if abs(angle - 90) > 20:
            return 'Maintain 90 degree bend'
    elif phase == 'lower':
        if angle > 45:
            return 'Lower arm fully'
    return 'Good form'


def get_severity(deviation):
    """
    Determine severity based on deviation from ideal.
    """
    if deviation < 10:
        return 'low'
    elif deviation < 30:
        return 'medium'
    else:
        return 'high'


def main():
    print("=" * 60)
    print("Physio-3 Enhanced Demo")
    print("=" * 60)
    print("\nFeatures Demonstrated:")
    print("1. Multi-exercise awareness with continuity tracking")
    print("2. Medical threshold validation (safe/unsafe rehab)")
    print("3. Structured output interface (JSON for VR/Unreal)")
    print("4. Error handling for pose dropouts & noisy frames")
    print("=" * 60)
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # Initialize original detectors
    phase_detector = MovementPhaseDetector(window_size=10, velocity_threshold=2.0)
    scorer = AdvancedSessionScorer()

    # Initialize NEW enhanced components
    medical_validator = MedicalThresholdValidator()
    noise_filter = LandmarkNoiseFilter(window_size=5, noise_threshold=5.0)
    dropout_handler = PoseDropoutHandler(max_consecutive_dropouts=5, recovery_frames=3)
    continuity_tracker = ExerciseContinuityTracker()
    output_interface = StructuredOutputInterface(OutputFormat.JSON)
    
    # Start exercise session
    output_interface.start_session("elbow_flexion", {
        "patient_id": "P001",
        "therapist": "Dr. Smith",
        "session_number": 3
    })

    # Webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    prev_score = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Get landmarks
            landmarks = results.pose_landmarks.landmark

            # NEW: Check visibility and handle dropouts
            left_elbow_vis = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].visibility
            
            # Register detection
            is_tracking, status = dropout_handler.register_detection()
            if not is_tracking:
                print(f"  [TRACKING] {status}")
            
            # NEW: Filter noisy landmarks
            filtered_elbow, _, quality = noise_filter.filter_position(
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                left_elbow_vis
            )
            if quality.warning:
                print(f"  [NOISE] {quality.warning}")

            # Calculate left elbow angle
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

            angle = calculate_angle(shoulder, elbow, wrist)

            # Update original phase detector
            phase_detector.update_angle(angle)
            phase = phase_detector.detect_phase()

            if phase:
                # Update original scorer
                scorer.update(angle, phase)

                # Get current scores
                scores = scorer.get_scores()
                current_score = scores['overall']
                score_delta = current_score - prev_score
                prev_score = current_score

                # NEW: Medical threshold validation
                velocity = angle - list(phase_detector.angle_buffer)[-2] if len(phase_detector.angle_buffer) > 1 else 0
                
                is_safe_vel, vel_warning = medical_validator.validate_velocity(velocity, phase)
                is_safe_angle, angle_warning = medical_validator.validate_angle(angle, "elbow_flexion")
                
                if not is_safe_vel:
                    print(f"  [ALERT] {vel_warning}")
                if not is_safe_angle:
                    print(f"  [ALERT] {angle_warning}")

                # NEW: Exercise continuity tracking
                continuity_tracker.record_phase(phase)
                continuity_report = continuity_tracker.get_continuity_report()
                
                # NEW: Record to structured output
                phase_data = PhaseData(
                    phase=phase,
                    confidence=0.85,
                    velocity=velocity,
                    timestamp=__import__('time').time()
                )
                output_interface.record_phase(phase_data, scores)

                # Get correction
                correction = get_correction(phase, angle)

                # Get severity (based on angle deviation from 90 for simplicity)
                deviation = abs(angle - 90)
                severity = get_severity(deviation)

                # Enhanced console output
                print(f"Frame {frame_count:4d} | Phase: {phase:6} | Angle: {angle:5.1f}° | "
                      f"Vel: {velocity:5.1f}°/f | Score: {current_score:5.1f} | State: {continuity_report['continuity_state']}")

            # Draw pose
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        else:
            # Handle pose dropout
            is_tracking, status = dropout_handler.register_dropout()
            if is_tracking is False:
                print(f"  [DROPOUT] {status}")

        # Display frame
        cv2.imshow('Physio-3 Enhanced Demo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Final outputs
    print("\n" + "=" * 60)
    print("FINAL SESSION DATA")
    print("=" * 60)
    
    # Original scores
    final_scores = scorer.get_scores()
    print("\n[ORIGINAL SCORING]")
    for key, value in final_scores.items():
        print(f"  {key}: {value}")
    
    # NEW: Continuity Report
    continuity_report = continuity_tracker.get_continuity_report()
    print("\n[EXERCISE CONTINUITY]")
    print(f"  Current Exercise: {continuity_report['current_exercise']}")
    print(f"  Continuity State: {continuity_report['continuity_state']}")
    print(f"  Total Transitions: {continuity_report['total_transitions']}")
    
    # NEW: VR Adapter Output
    vr_output = output_interface.get_vr_adapter_data()
    print("\n[VR/UNREAL ADAPTER OUTPUT]")
    print(f"  Session ID: {vr_output.get('session_id', 'N/A')}")
    print(f"  Repetition Count: {vr_output.get('repetition_count', 0)}")
    print(f"  Current Phase: {vr_output.get('current_phase', 'N/A')}")
    
    # NEW: JSON Export
    json_output = output_interface.export_current_session()
    print("\n[JSON OUTPUT - First 600 chars]")
    print(json_output[:600] + "..." if len(json_output) > 600 else json_output)
    
    # NEW: Dropout Handler Status
    dropout_status = dropout_handler.get_status()
    print("\n[DROPOUT HANDLER STATUS]")
    print(f"  Is Tracking: {dropout_status['is_tracking']}")
    print(f"  Consecutive Dropouts: {dropout_status['consecutive_dropouts']}")
    print(f"  Dropout Rate (last 30 frames): {dropout_status['dropout_rate']:.2%}")

    print("\n" + "=" * 60)
    print("Demo Complete - All Missing Features Implemented!")
    print("=" * 60)


if __name__ == "__main__":
    main()
