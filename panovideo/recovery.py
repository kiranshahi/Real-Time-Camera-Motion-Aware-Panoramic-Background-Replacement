"""T09: explicit recovery state, safe tracking reseed and pose-preserving mode switches."""
from dataclasses import dataclass
from enum import Enum
import numpy as np
from .geometry import rotation_matrix


class RecoveryState(str, Enum):
    TRACKING = 'TRACKING'
    HOLD = 'HOLD'
    RESEED = 'RESEED'
    WARMUP = 'WARMUP'


@dataclass(frozen=True)
class RecoveryConfig:
    reseed_after: int = 3
    warmup_frames: int = 1


class TrackingRecovery:
    def __init__(self, tracker, config=None):
        self.tracker = tracker
        self.config = config or RecoveryConfig()
        if self.config.reseed_after < 1 or self.config.warmup_frames < 0:
            raise ValueError('Invalid recovery configuration')
        self.state = RecoveryState.WARMUP
        self.failures = 0
        self.warmup_left = self.config.warmup_frames
        self.mode = 'AUTO'

    def _reseed(self, frame, foreground_mask=None):
        """Refresh optical-flow correspondences without clearing absolute orientation."""
        self.tracker.reinitialize_features(frame, foreground_mask)
        self.state = RecoveryState.WARMUP
        self.warmup_left = self.config.warmup_frames
        self.failures = 0

    def update(self, frame, foreground_mask, decision):
        """Call once per AUTO frame *after* validation. Returns recovery state.

        Rejected estimates do not update rotation in the existing T04 policy.
        A reseed takes effect on the next frame (no old-new bridge across gaps).
        """
        if self.mode != 'AUTO':
            return self.state
        if self.state == RecoveryState.WARMUP:
            if self.warmup_left > 0:
                self.warmup_left -= 1
                return self.state
            self.state = RecoveryState.HOLD
        if decision is not None and decision.accepted:
            self.failures = 0
            self.state = RecoveryState.TRACKING
        else:
            self.failures += 1
            self.state = RecoveryState.HOLD
            if self.failures >= self.config.reseed_after:
                self.state = RecoveryState.RESEED
                self._reseed(frame, foreground_mask)
        return self.state

    def set_mode(self, mode, displayed_ypr, frame, foreground_mask=None):
        """Keep the *displayed* pose continuous on both AUTO/MANUAL transitions.

        Rebase AUTO's camera-to-world matrix from the visible Euler pose and
        restart optical flow to prevent stale-frame motion at the transition.
        """
        mode = str(mode).upper()
        if mode not in ('AUTO', 'MANUAL'):
            raise ValueError('mode must be AUTO or MANUAL')
        if mode == self.mode:
            return tuple(float(v) for v in displayed_ypr)
        ypr = tuple(float(v) for v in displayed_ypr)
        if not np.all(np.isfinite(ypr)):
            raise ValueError('Nonfinite pose')
        self.tracker.set_orientation_anchor(ypr)
        self._reseed(frame, foreground_mask)
        self.mode = mode
        return ypr
