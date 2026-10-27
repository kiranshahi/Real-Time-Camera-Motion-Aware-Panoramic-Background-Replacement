"""Pose-provider interface, initially controlled with UI trackbars."""
from dataclasses import dataclass


@dataclass
class Pose:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    confidence: float = 1.0


class ManualPoseProvider:
    def update(self, frame, timestamp_seconds: float, *, controls=None) -> Pose:
        if controls is None:
            return Pose()
        return Pose(*controls)


# Later implement VisualOdometryPoseProvider with the same update signature.
# Keep the output as absolute orientation relative to the initial camera frame.
