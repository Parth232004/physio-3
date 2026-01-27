import numpy as np
from collections import deque

class MovementPhaseDetector:
    def __init__(self, window_size=10, velocity_threshold=5.0):
        """
        Initialize the movement phase detector.

        Args:
            window_size (int): Number of frames to use for rolling window analysis.
            velocity_threshold (float): Threshold for detecting phase changes (degrees per frame).
        """
        self.window_size = window_size
        self.velocity_threshold = velocity_threshold
        self.angle_buffer = deque(maxlen=window_size)
        self.previous_phase = None

    def calculate_angle(self, landmark1, landmark2, landmark3):
        """
        Calculate the angle at landmark2 formed by landmark1-landmark2-landmark3.

        Args:
            landmark1, landmark2, landmark3: Tuples of (x, y) coordinates.

        Returns:
            float: Angle in degrees.
        """
        # Vectors
        v1 = np.array([landmark1[0] - landmark2[0], landmark1[1] - landmark2[1]])
        v2 = np.array([landmark3[0] - landmark2[0], landmark3[1] - landmark2[1]])

        # Cosine of angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)  # Avoid numerical errors

        angle = np.arccos(cos_angle) * 180 / np.pi
        return angle

    def update_angle(self, angle):
        """
        Update the angle buffer with a new angle measurement.

        Args:
            angle (float): New angle measurement.
        """
        self.angle_buffer.append(angle)

    def detect_phase(self):
        """
        Detect the current movement phase based on the rolling window of angles.

        Returns:
            str: Current phase ('raise', 'hold', 'lower', or None if insufficient data).
        """
        if len(self.angle_buffer) < self.window_size:
            return None

        # Calculate velocities (differences)
        angles = list(self.angle_buffer)
        velocities = [angles[i] - angles[i-1] for i in range(1, len(angles))]

        # Average velocity over the window
        avg_velocity = np.mean(velocities)

        # Determine phase
        if avg_velocity < -self.velocity_threshold:
            phase = 'raise'
        elif avg_velocity > self.velocity_threshold:
            phase = 'lower'
        else:
            phase = 'hold'

        self.previous_phase = phase
        return phase

    def get_angle_history(self):
        """
        Get the current angle history.

        Returns:
            list: List of angles in the buffer.
        """
        return list(self.angle_buffer)