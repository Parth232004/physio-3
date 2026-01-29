import cv2
import mediapipe as mp
import numpy as np
from movement_phase import MovementPhaseDetector
from advanced_session_scoring import AdvancedSessionScorer

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
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Initialize detectors
    phase_detector = MovementPhaseDetector(window_size=10, velocity_threshold=2.0)
    scorer = AdvancedSessionScorer()

    # Webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    prev_score = 0

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

            # Calculate left elbow angle
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

            angle = calculate_angle(shoulder, elbow, wrist)

            # Update phase detector
            phase_detector.update_angle(angle)
            phase = phase_detector.detect_phase()

            if phase:
                # Update scorer
                scorer.update(angle, phase)

                # Get current scores
                scores = scorer.get_scores()
                current_score = scores['overall']
                score_delta = current_score - prev_score
                prev_score = current_score

                # Get correction
                correction = get_correction(phase, angle)

                # Get severity (based on angle deviation from 90 for simplicity)
                deviation = abs(angle - 90)
                severity = get_severity(deviation)

                # Console output
                print(f"Phase: {phase}, Angle: {angle:.1f}, Correction: {correction}, Severity: {severity}, Score Delta: {score_delta:.2f}")

            # Draw pose
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display frame
        cv2.imshow('Physio Demo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Final scores
    final_scores = scorer.get_scores()
    print("Final Scores:")
    for key, value in final_scores.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()