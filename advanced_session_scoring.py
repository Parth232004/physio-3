import numpy as np
from collections import defaultdict

class AdvancedSessionScorer:
    def __init__(self, stability_weight=0.3, smoothness_weight=0.4, completion_weight=0.3):
        """
        Initialize the advanced session scorer.

        Args:
            stability_weight (float): Weight for stability score.
            smoothness_weight (float): Weight for smoothness score.
            completion_weight (float): Weight for completion accuracy score.
        """
        self.stability_weight = stability_weight
        self.smoothness_weight = smoothness_weight
        self.completion_weight = completion_weight

        # Data storage
        self.angles = []
        self.phases = []
        self.velocities = []

    def update(self, angle, phase):
        """
        Update the scorer with new angle and phase data.

        Args:
            angle (float): Current joint angle.
            phase (str): Current phase ('raise', 'hold', 'lower').
        """
        self.angles.append(angle)
        self.phases.append(phase)

        if len(self.angles) > 1:
            velocity = self.angles[-1] - self.angles[-2]
            self.velocities.append(velocity)

    def calculate_stability(self):
        """
        Calculate stability score based on angle variance during hold phases.

        Returns:
            float: Stability score (0-100, higher is better).
        """
        if not self.angles:
            return 0.0

        # Get angles during hold phases
        hold_angles = [self.angles[i] for i in range(len(self.phases)) if self.phases[i] == 'hold']

        if not hold_angles:
            return 50.0  # Neutral if no hold phases

        # Stability: lower variance is better
        variance = np.var(hold_angles)
        # Normalize: assume max variance is 100 deg^2, score = 100 - (variance / max_variance * 100)
        max_variance = 100.0
        stability_score = max(0, 100 - (variance / max_variance * 100))
        return stability_score

    def calculate_smoothness(self):
        """
        Calculate smoothness score based on velocity consistency and jerkiness.

        Returns:
            float: Smoothness score (0-100, higher is better).
        """
        if len(self.velocities) < 2:
            return 50.0

        # Smoothness: lower std dev of velocities and lower max jerk is better
        vel_std = np.std(self.velocities)
        jerks = [self.velocities[i] - self.velocities[i-1] for i in range(1, len(self.velocities))]
        max_jerk = max(abs(j) for j in jerks) if jerks else 0

        # Penalize high vel_std and high max_jerk
        vel_penalty = min(50, vel_std * 2)  # Assume 25 deg/frame std is 50 penalty
        jerk_penalty = min(50, max_jerk * 5)  # Assume 10 deg/frame^2 jerk is 50 penalty

        smoothness_score = 100 - vel_penalty - jerk_penalty
        return max(0, smoothness_score)

    def calculate_completion_accuracy(self):
        """
        Calculate completion accuracy based on phase sequence and duration.

        Returns:
            float: Completion accuracy score (0-100, higher is better).
        """
        if not self.phases:
            return 0.0

        # Check for proper sequence: raise -> hold -> lower (at least once)
        phase_counts = defaultdict(int)
        for phase in self.phases:
            phase_counts[phase] += 1

        # Require at least one of each phase
        has_raise = phase_counts['raise'] > 0
        has_hold = phase_counts['hold'] > 0
        has_lower = phase_counts['lower'] > 0

        if not (has_raise and has_hold and has_lower):
            return 20.0  # Low score if missing phases

        # Check sequence: look for raise-hold-lower pattern
        sequence_score = 0
        prev_phase = None
        for phase in self.phases:
            if prev_phase == 'raise' and phase == 'hold':
                sequence_score += 25
            elif prev_phase == 'hold' and phase == 'lower':
                sequence_score += 25
            prev_phase = phase

        # Cap at 50 for sequence
        sequence_score = min(50, sequence_score)

        # Duration balance: similar counts
        total_phases = len(self.phases)
        ideal_each = total_phases / 3
        duration_penalty = sum(abs(phase_counts[p] - ideal_each) for p in ['raise', 'hold', 'lower']) / total_phases * 100
        duration_score = max(0, 50 - duration_penalty)

        return sequence_score + duration_score

    def get_scores(self):
        """
        Get the current session scores.

        Returns:
            dict: Scores for stability, smoothness, completion_accuracy, and overall.
        """
        stability = self.calculate_stability()
        smoothness = self.calculate_smoothness()
        completion = self.calculate_completion_accuracy()

        overall = (self.stability_weight * stability +
                   self.smoothness_weight * smoothness +
                   self.completion_weight * completion)

        return {
            'stability': round(stability, 2),
            'smoothness': round(smoothness, 2),
            'completion_accuracy': round(completion, 2),
            'overall': round(overall, 2)
        }

    def reset(self):
        """
        Reset the scorer for a new session.
        """
        self.angles = []
        self.phases = []
        self.velocities = []