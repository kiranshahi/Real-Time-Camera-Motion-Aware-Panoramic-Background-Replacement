"""Shared pinhole camera calibration and optional OpenCV distortion rectification."""
from dataclasses import dataclass
import json
import math
import numpy as np
import cv2

@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple = ()

    def __post_init__(self):
        if self.width < 1 or self.height < 1:
            raise ValueError("Invalid camera image dimensions")
        if not all(math.isfinite(float(x)) for x in (self.fx, self.fy, self.cx, self.cy)) or self.fx <= 0 or self.fy <= 0:
            raise ValueError("Camera intrinsic parameters must be finite and focal lengths positive")
        if len(self.distortion) not in (0, 4, 5, 8, 12, 14) or not all(math.isfinite(float(x)) for x in self.distortion):
            raise ValueError("Distortion must contain 4, 5, 8, 12, or 14 finite coefficients")

    @classmethod
    def from_hfov(cls, width, height, hfov_degrees=70.0):
        if not 0 < hfov_degrees < 180:
            raise ValueError("HFOV must be between 0 and 180 degrees")
        f = width / (2 * math.tan(math.radians(hfov_degrees) / 2))
        # Preserve T01–T09 fallback principal-point convention.
        return cls(int(width), int(height), f, f, width / 2, height / 2)

    @property
    def matrix(self):
        return np.array([[self.fx, 0., self.cx], [0., self.fy, self.cy], [0., 0., 1.]], dtype=np.float64)

    @property
    def coefficients(self):
        return np.array(self.distortion if self.distortion else [0., 0., 0., 0., 0.], dtype=np.float64)

    def rectification_maps(self):
        """Same K for rectified output; no crop or newCameraMatrix change."""
        return cv2.initUndistortRectifyMap(self.matrix, self.coefficients, None, self.matrix, (self.width, self.height), cv2.CV_32FC1)

    def rectify(self, frame):
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError("Frame size disagrees with calibration")
        if not self.distortion or not np.any(self.coefficients):
            return frame
        return cv2.undistort(frame, self.matrix, self.coefficients, None, self.matrix)

    def pixel_rays(self):
        x, y = np.meshgrid(np.arange(self.width, dtype=np.float64), np.arange(self.height, dtype=np.float64))
        rays = np.stack(((x-self.cx)/self.fx, (y-self.cy)/self.fy, np.ones_like(x)), axis=-1)
        return rays / np.linalg.norm(rays, axis=-1, keepdims=True)

    @classmethod
    def from_json(cls, path, width=None, height=None):
        with open(path, encoding="utf8") as f:
            v = json.load(f)
        obj = cls(int(v['width']), int(v['height']), float(v['fx']), float(v['fy']), float(v['cx']), float(v['cy']), tuple(float(x) for x in v.get('distortion', [])))
        if width is not None and (width, height) != (obj.width, obj.height):
            raise ValueError("Calibration resolution does not match actual camera frame; recalibrate or rescale explicitly")
        return obj
