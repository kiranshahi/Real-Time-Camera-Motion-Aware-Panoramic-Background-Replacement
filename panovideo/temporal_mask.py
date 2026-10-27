"""Conservative foreground exclusion propagation during short detection dropouts.

This mask is for *feature tracking*, never for foreground compositing.
"""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class TemporalMaskConfig:
    max_missing_frames: int = 2
    fallback_dilation_px: int = 8
    min_consistent_fraction: float = 0.55
    fb_tolerance_px: float = 2.0


class TemporalForegroundMask:
    def __init__(self, config=None):
        self.config = config or TemporalMaskConfig()
        self.previous_gray = None
        self.last_mask = None
        self.missing = 0
        self.last_status = 'EMPTY'

    def reset(self):
        self.previous_gray = None
        self.last_mask = None
        self.missing = 0
        self.last_status = 'EMPTY'

    def update(self, frame, current_mask, *, detected):
        """Return uint8 exclusion mask; detected means a real positive segmentation.

        current_mask already contains tracking-specific dilation. A blank detected
        mask explicitly clears stale state; a blank non-detected mask is a dropout.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if current_mask.shape != gray.shape:
            raise ValueError('Invalid exclusion mask dimensions')
        if self.previous_gray is not None and self.previous_gray.shape != gray.shape:
            self.reset()
        mask = np.where(np.asarray(current_mask) != 0, 255, 0).astype(np.uint8)
        if detected and np.any(mask):
            self.last_mask = mask.copy()
            self.missing = 0
            self.last_status = 'DETECTED'
        elif detected:  # Explicit validated empty scene
            self.last_mask = None
            self.missing = 0
            self.last_status = 'EMPTY'
        else:
            self.missing += 1
            if self.last_mask is None or self.missing > self.config.max_missing_frames:
                self.last_mask = None
                self.last_status = 'EXPIRED' if self.missing > self.config.max_missing_frames else 'EMPTY'
            else:
                propagated = self._warp_consistent(self.previous_gray, gray, self.last_mask)
                if propagated is None:
                    # Conservative fallback for ONE dropout only. Avoid persistent ghost masks.
                    if self.missing == 1:
                        n = self.config.fallback_dilation_px
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*n+1, 2*n+1))
                        self.last_mask = cv2.dilate(self.last_mask, kernel)
                        self.last_status = 'FALLBACK'
                    else:
                        self.last_mask = None
                        self.last_status = 'UNRELIABLE'
                else:
                    self.last_mask = propagated
                    self.last_status = 'WARPED'
        self.previous_gray = gray.copy()
        return (self.last_mask.copy() if self.last_mask is not None
                else np.zeros_like(gray, dtype=np.uint8))

    def _warp_consistent(self, prev_gray, gray, mask):
        if prev_gray is None or mask is None:
            return None
        try:
            params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            forward = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **params)
            backward = cv2.calcOpticalFlowFarneback(gray, prev_gray, None, **params)
            h, w = gray.shape
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            src_x = xx + backward[..., 0]
            src_y = yy + backward[..., 1]
            valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
            fwd_at_src = cv2.remap(forward, src_x, src_y, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
            consistency = np.linalg.norm(backward + fwd_at_src, axis=2)
            warped = cv2.remap(mask, src_x, src_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            fg = warped != 0
            if fg.sum() == 0 or np.mean((valid & (consistency <= self.config.fb_tolerance_px))[fg]) < self.config.min_consistent_fraction:
                return None
            # Do not leave holes in uncertain pixels within silhouette: dilate conservatively.
            good = valid & (consistency <= self.config.fb_tolerance_px)
            output = np.where(fg & good, 255, 0).astype(np.uint8)
            output = cv2.dilate(output, np.ones((3, 3), dtype=np.uint8))
            if np.count_nonzero(output) < 0.25 * np.count_nonzero(mask):
                return None
            return output
        except cv2.error:
            return None
