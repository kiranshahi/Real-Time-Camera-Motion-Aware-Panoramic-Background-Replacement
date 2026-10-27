from dataclasses import dataclass, replace
from enum import Enum

class MotionState(str, Enum):
    VALID = "VALID"
    UNCERTAIN = "UNCERTAIN"
    INVALID = "INVALID"

@dataclass(frozen=True)
class AcceptanceConfig:
    min_inliers: int = 20
    min_inlier_ratio: float = 0.60
    max_frame_rotation_deg: float = 10.0
    rotation_rmse_valid_px: float = 2.0  # illustrative, not calibrated
    rotation_rmse_invalid_px: float = 5.0
    min_occupied_cells: int = 5  # 4x4 image grid, homography inliers
    min_coverage_fraction: float = 0.18  # bbox / image area
    max_innovation_deg: float = 5.0  # difference from previous valid increment
    invalid_after_failures: int = 3

@dataclass(frozen=True)
class MotionDecision:
    state: MotionState
    reason: str
    accepted: bool
    inliers: int = 0
    inlier_ratio: float = 0.0
    rotation_rmse_px: float = float('inf')
    occupied_cells: int = 0
    coverage_fraction: float = 0.0
    innovation_deg: float = float('nan')
    confidence: float = 0.0  # T08 heuristic, not a calibrated probability
    mask_reliability: float = 1.0

import cv2
import numpy as np
from .calibration import CameraCalibration

class BackgroundFeatureTracker:

    def __init__(self):

        self.feature_params = {
            "maxCorners": 300,
            "qualityLevel": 0.01,
            "minDistance": 10,
            "blockSize": 7
        }

        
        self.lk_params = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        }

        self.previous_gray = None
        self.previous_points = None

        self.camera_matrix = None
        self.rotation_matrix = np.eye(3, dtype=np.float64)

        self.min_matches = 12
        self.ransac_threshold = 3.0
        
        self.fb_threshold = 1.5
        self.min_tracked_features = 80
        self.feature_grid = (4, 4)
        self.feature_spacing_px = 10.0

        self.smoothed_rotation = np.eye(3, dtype=np.float64)
        self.smoothing_alpha = 0.25
        self.last_valid_increment = None
        self.consecutive_failures = 0

    
    def _background_mask(self, shape, foreground_mask):
        """OpenCV mask: 255=allowed, 0=excluded. Treat any nonzero foreground as excluded."""
        if foreground_mask is None:
            return np.full(shape, 255, dtype=np.uint8)
        if foreground_mask.shape != shape:
            raise ValueError("Invalid exclusion mask dimensions")
        return np.where(foreground_mask != 0, 0, 255).astype(np.uint8)

    def _select_distributed(self, candidates, existing, shape, allowed_mask, target):
        """Deterministically select candidates round-robin over image cells.

        Uses order of candidates as detector's strength order within each cell.
        Enforces spacing to BOTH existing tracks and newly accepted points.
        """
        h, w = shape
        if target <= 0:
            return np.empty((0, 1, 2), dtype=np.float32)
        rows, cols = self.feature_grid
        buckets = [[] for _ in range(rows * cols)]
        for point in np.asarray(candidates, dtype=np.float32).reshape(-1, 2):
            x, y = map(float, point)
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            ix, iy = int(round(x)), int(round(y))
            if not (0 <= ix < w and 0 <= iy < h) or allowed_mask[iy, ix] == 0:
                continue
            col = min(int(x * cols / w), cols - 1)
            row = min(int(y * rows / h), rows - 1)
            buckets[row * cols + col].append(point)

        points = [p.copy() for p in np.asarray(existing, dtype=np.float32).reshape(-1, 2)]
        selected = []
        threshold_sq = self.feature_spacing_px ** 2
        offsets = [0] * len(buckets)
        while len(selected) < target:
            progress = False
            for k, bucket in enumerate(buckets):
                while offsets[k] < len(bucket):
                    candidate = bucket[offsets[k]]
                    offsets[k] += 1
                    if all(float(np.sum((candidate - p) ** 2)) >= threshold_sq for p in points):
                        selected.append(candidate)
                        points.append(candidate)
                        progress = True
                        break
                if len(selected) >= target:
                    break
            if not progress:
                break
        return np.asarray(selected, dtype=np.float32).reshape(-1, 1, 2)

    def detect_features(self, frame, foreground_mask=None, existing_points=None, target=None):
        """Find spatially balanced Shi-Tomasi corners; all points obey foreground and spacing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        allowed = self._background_mask(gray.shape, foreground_mask)
        existing = (np.empty((0, 2), np.float32) if existing_points is None
                    else np.asarray(existing_points, np.float32).reshape(-1, 2))
        limit = int(self.feature_params['maxCorners'])
        count = min(max(limit - len(existing), 0),
                    limit if target is None else max(int(target), 0))
        if count == 0:
            return None
        # Suppress near existing points *before* detection, to avoid starving the candidate pool.
        radius = max(1, int(np.ceil(self.feature_spacing_px)))
        for x, y in existing:
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(allowed, (int(round(float(x))), int(round(float(y)))),
                           radius, 0, -1)
        # Per-cell feature detection avoids one high-texture patch dominating the result.
        rows, cols = self.feature_grid
        candidates = []
        params = dict(self.feature_params)
        params['maxCorners'] = limit
        for r in range(rows):
            y0, y1 = r * gray.shape[0] // rows, (r+1) * gray.shape[0] // rows
            for c in range(cols):
                x0, x1 = c * gray.shape[1] // cols, (c+1) * gray.shape[1] // cols
                cell = gray[y0:y1, x0:x1]
                mask = allowed[y0:y1, x0:x1]
                if cell.size == 0 or not np.any(mask):
                    continue
                detected = cv2.goodFeaturesToTrack(cell, mask=mask, **params)
                if detected is not None:
                    coords = detected.reshape(-1, 2) + np.array([x0, y0], np.float32)
                    candidates.extend(coords)
        if not candidates:
            return None
        result = self._select_distributed(candidates, existing, gray.shape,
                                           self._background_mask(gray.shape, foreground_mask), count)
        return result if len(result) else None

    def draw_features(self, frame, features):
        output = frame.copy();

        if features is None:
            return output

        for point in features:
            x, y = point.ravel()
            cv2.circle(output, (int(x), int(y)), 3, (0, 255, 0), -1)

        return output

    def track_features(self, frame, foreground_mask=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # First frame: initial feature points
        if self.previous_gray is None:
            self.previous_gray = gray.copy()
            self.previous_points = self.detect_features(frame, foreground_mask)
            return None, None

        if (self.previous_points is None or len(self.previous_points) < 12):
            self.previous_gray = gray.copy()
            self.previous_points = self.detect_features(frame, foreground_mask)
            return None, None

        next_points, status, error = (cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, self.previous_points, None, **self.lk_params))

        if next_points is None or status is None:
            self.previous_gray = gray.copy()
            self.previous_points = self.detect_features(frame, foreground_mask)
            return None, None

        # Track new feature locations backward
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, next_points, None, **self.lk_params)
        if back_points is None or back_status is None:
            self.previous_gray = gray.copy()
            self.previous_points = self.detect_features(frame, foreground_mask)
            return None, None

        # Compute forward-backward tracking error
        fb_error = np.linalg.norm(self.previous_points.reshape(-1, 2) - back_points.reshape(-1, 2), axis=1)


        valid = ((status.reshape(-1) == 1) & (back_status.reshape(-1) == 1) & np.isfinite(fb_error) & (fb_error < self.fb_threshold))
        old_points = self.previous_points[valid].reshape(-1, 2)
        new_points = next_points[valid].reshape(-1, 2)

        # Reject features now lying on the foreground
        if foreground_mask is not None and len(new_points) > 0:
            coords = np.rint(new_points).astype(int)
            h, w = gray.shape

            inside = ((coords[:, 0] >= 0) & (coords[:, 0] < w) & (coords[:, 1] >= 0) & (coords[:, 1] < h))

            keep = np.zeros(len(coords), dtype=bool)
            indices = np.flatnonzero(inside)

            keep[indices] = (foreground_mask[coords[indices, 1], coords[indices, 0]] == 0)

            old_points = old_points[keep]
            new_points = new_points[keep]

        # Keep the valid correspondences for this frame
        tracked_points = new_points.reshape(-1, 1, 2).astype(np.float32)

        # Replenish only up to target, ensuring spacing against old AND fresh points.
        if len(tracked_points) < self.min_tracked_features:
            fresh_points = self.detect_features(
                frame, foreground_mask, existing_points=tracked_points,
                target=self.min_tracked_features - len(tracked_points))
            if fresh_points is not None:
                tracked_points = np.concatenate([tracked_points, fresh_points], axis=0)

        self.previous_gray = gray.copy()
        self.previous_points = (tracked_points if len(tracked_points) > 0 else None)

        return old_points, new_points

    
    def draw_tracking(self, frame, old_points, new_points):

        output = frame.copy()

        if old_points is None or new_points is None:
            return output

        for old, new in zip(old_points, new_points):

            x0, y0 = old.astype(int)
            x1, y1 = new.astype(int)

            cv2.line(output, (x0, y0), (x1, y1), (0, 255, 0), 1)
            cv2.circle(output, (x1, y1), 3, (0, 0, 255), -1)

        return output

    def set_calibration(self, calibration: CameraCalibration):
        """Use the shared intrinsics of RECTIFIED images for geometric estimation."""
        self.calibration = calibration
        self.camera_matrix = calibration.matrix
        return self.camera_matrix

    def set_camera_intrinsics(self, width, height, fov_degrees=70.0):
        """Backwards-compatible HFOV setup for older tests and callers."""
        return self.set_calibration(CameraCalibration.from_hfov(width, height, fov_degrees))

    def estimate_rotation(self, old_points, new_points):
        """
        Estimate relative camera rotation from
        tracked background feature correspondences.
        """

        if old_points is None or new_points is None:
            return None, 0

        if len(old_points) < self.min_matches:
            return None, 0

        if self.camera_matrix is None:
            raise RuntimeError("Camera intrinsics are not initialized.")

        # Step 1: Estimate homography with RANSAC
        H, inlier_mask = cv2.findHomography(old_points, new_points, cv2.RANSAC, self.ransac_threshold)

        if H is None or inlier_mask is None:
            return None, 0

        inlier_count = int(inlier_mask.sum())

        if inlier_count < self.min_matches:
            return None, inlier_count

        # Step 2: Remove camera intrinsics
        K = self.camera_matrix
        K_inv = np.linalg.inv(K)

        R_approx = K_inv @ H @ K

        if not np.all(np.isfinite(R_approx)):
            return None, inlier_count

        # Step 3: Normalize the matrix scale
        determinant = np.linalg.det(R_approx)

        if not np.isfinite(determinant) or determinant <= 0:
            return None, inlier_count

        R_approx /= np.cbrt(determinant)

        # Step 4: Project onto a valid rotation matrix
        U, _, Vt = np.linalg.svd(R_approx)

        R = U @ Vt

        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        return R, inlier_count

    def update_rotation(self, relative_rotation):
        """
        Accumulate frame-to-frame camera rotation.

        relative_rotation maps points from the previous
        camera coordinate system into the current one.
        """

        if relative_rotation is None:
            return self.rotation_matrix

        self.rotation_matrix = (
            relative_rotation @ self.rotation_matrix
        )

        # Prevent numerical errors from accumulating
        U, _, Vt = np.linalg.svd(self.rotation_matrix)

        R = U @ Vt

        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        self.rotation_matrix = R

        return self.rotation_matrix

    def get_camera_orientation(self):
        """
        Return camera orientation relative to initial frame.

        The accumulated matrix maps initial coordinates
        into current camera coordinates, so transpose it
        to obtain camera-to-world rotation.
        """
        R = self.smoothed_rotation.T

        # Extract pitch
        pitch = np.arcsin(np.clip(-R[1, 2], -1.0, 1.0))

        # Extract yaw and roll
        if abs(np.cos(pitch)) > 1e-6:
            yaw = np.arctan2(R[0, 2], R[2, 2])
            roll = np.arctan2(R[1, 0], R[1, 1])
        else:
            # Gimbal-lock fallback
            yaw = np.arctan2(-R[2, 0], R[0, 0])
            roll = 0.0

        return tuple(np.rad2deg([yaw, pitch, roll]))

    
    def reset_rotation(self):
        """
        Reset the estimated camera orientation
        and optical-flow tracking state.
        """

        self.rotation_matrix = np.eye(3, dtype=np.float64)
        self.smoothed_rotation = np.eye(3, dtype=np.float64)

        self.previous_gray = None
        self.previous_points = None
        self.last_valid_increment = None
        self.consecutive_failures = 0

    def reinitialize_features(self, frame, foreground_mask=None):
        """Seed LK features on the current frame without changing camera orientation."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.previous_gray = gray.copy()
        self.previous_points = self.detect_features(frame, foreground_mask)
        self.last_valid_increment = None
        self.consecutive_failures = 0

    def set_orientation_anchor(self, yaw_pitch_roll):
        """Rebase pose to currently displayed yaw/pitch/roll (no visual jump)."""
        from .geometry import rotation_matrix
        ypr = np.asarray(yaw_pitch_roll, dtype=np.float64)
        if ypr.shape != (3,) or not np.all(np.isfinite(ypr)):
            raise ValueError('Expected three finite angles')
        R = rotation_matrix(*ypr).astype(np.float64).T
        self.rotation_matrix = R.copy()
        self.smoothed_rotation = R.copy()
        self.last_valid_increment = None
        self.consecutive_failures = 0

    def smooth_rotation(self):
        """
        Smooth accumulated rotation using
        the SO(3) logarithm and exponential maps.
        """

        target = self.rotation_matrix
        current = self.smoothed_rotation

        # Rotation error between current and target
        delta = target @ current.T

        # Logarithm map: rotation matrix to rotation vector
        rotation_vector, _ = cv2.Rodrigues(delta)

        # Apply a fraction of the rotational difference
        step_vector = (self.smoothing_alpha * rotation_vector)

        # Exponential map: rotation vector to matrix
        step_matrix, _ = cv2.Rodrigues(step_vector)

        self.smoothed_rotation = (step_matrix @ current)

        return self.smoothed_rotation


    def diagnose_rotation_fit(self, old_points, new_points):
        """Compare general homography and SO(3)-constrained rotation on the SAME RANSAC inliers.

        This is a diagnostic, not a translation detector. In particular, uniform image
        translation and planar motion can sometimes be approximated by rotation.
        Returns a dict of errors in pixels, masks, and the estimated old-to-new R.
        Existing estimate_rotation() behavior is intentionally unchanged for T02.
        """
        if old_points is None or new_points is None:
            return None
        old = np.asarray(old_points, dtype=np.float64).reshape(-1, 2)
        new = np.asarray(new_points, dtype=np.float64).reshape(-1, 2)
        if len(old) != len(new) or len(old) < self.min_matches:
            return None
        if not np.isfinite(old).all() or not np.isfinite(new).all():
            return None
        if self.camera_matrix is None:
            raise RuntimeError('Camera intrinsics are not initialized.')

        H, mask = cv2.findHomography(old, new, cv2.RANSAC, self.ransac_threshold)
        if H is None or mask is None:
            return None
        inlier = mask.reshape(-1).astype(bool)
        if int(inlier.sum()) < self.min_matches:
            return None
        K = self.camera_matrix
        approx = np.linalg.inv(K) @ H @ K
        determinant = np.linalg.det(approx)
        if not np.isfinite(approx).all() or not np.isfinite(determinant) or determinant <= 0:
            return None
        approx = approx / np.cbrt(determinant)
        U, _, Vt = np.linalg.svd(approx)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        def errors(transform):
            pts = np.column_stack((old, np.ones(len(old)))) @ transform.T
            z = pts[:, 2]
            e = np.full(len(old), np.inf, dtype=np.float64)
            valid = np.isfinite(pts).all(axis=1) & (np.abs(z) > 1e-9)
            e[valid] = np.linalg.norm(pts[valid, :2] / z[valid, None] - new[valid], axis=1)
            return e

        h_error = errors(H)
        r_error = errors(K @ R @ np.linalg.inv(K))

        def stats(e):
            # Metrics are computed on the HOMOGRAPHY INLIER SET for fair comparison.
            v = e[inlier]
            return dict(rmse_px=float(np.sqrt(np.mean(v ** 2))),
                        median_px=float(np.median(v)), p95_px=float(np.percentile(v, 95)),
                        mean_px=float(np.mean(v)))

        hs, rs = stats(h_error), stats(r_error)
        return {
            'rotation': R,
            'homography': H,
            'inlier_mask': inlier,
            'matches': len(old),
            'inliers': int(inlier.sum()),
            'inlier_ratio': float(inlier.mean()),
            'homography_fit': hs,
            'rotation_fit': rs,
            'rmse_gap_px': rs['rmse_px'] - hs['rmse_px'],
            'homography_errors_px': h_error,
            'rotation_errors_px': r_error,
        }

    @staticmethod
    def _rotation_angle_deg(R):
        c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.rad2deg(np.arccos(c)))

    def _coverage(self, points, inliers, frame_size):
        width, height = frame_size
        if width <= 0 or height <= 0:
            return 0, 0.0
        xy = np.asarray(points).reshape(-1, 2)[inliers]
        if not len(xy):
            return 0, 0.0
        good = np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        xy = xy[good]
        if not len(xy):
            return 0, 0.0
        cells = np.floor(xy / [width / 4.0, height / 4.0]).astype(int)
        cells = np.clip(cells, 0, 3)
        occupied = len(np.unique(cells[:, 1] * 4 + cells[:, 0]))
        extents = np.ptp(xy, axis=0)
        return occupied, float(np.prod(extents) / (width * height))

    def _validate_rotation_policy(self, old_points, new_points, *, frame_size, config=None):
        """Evaluate and (if valid) accumulate relative pose. No smoothing here.

        This replaces the main loop's old 3-threshold decision, not the T01-T03
        estimate_rotation() baseline. UNCERTAIN/INVALID both HOLD orientation.
        This method uses a fresh T02 RANSAC diagnostic once per frame.
        """
        cfg = config or AcceptanceConfig()
        diag = self.diagnose_rotation_fit(old_points, new_points)
        if diag is None:
            return self._reject(MotionState.INVALID, 'insufficient_or_degenerate_correspondences', cfg)
        R = diag['rotation']
        inliers = diag['inliers']
        ratio = diag['inlier_ratio']
        residual = diag['rotation_fit']['rmse_px']
        cells, coverage = self._coverage(old_points, diag['inlier_mask'], frame_size)
        angle = self._rotation_angle_deg(R)
        innovation = (self._rotation_angle_deg(R @ self.last_valid_increment.T)
                      if self.last_valid_increment is not None and self.consecutive_failures == 0
                      else float('nan'))
        data = dict(inliers=inliers, inlier_ratio=ratio, rotation_rmse_px=residual,
                    occupied_cells=cells, coverage_fraction=coverage, innovation_deg=innovation)
        if not np.isfinite(residual) or angle > cfg.max_frame_rotation_deg or residual > cfg.rotation_rmse_invalid_px:
            return self._reject(MotionState.INVALID, 'nonphysical_or_large_rotation_residual', cfg, **data)
        if inliers < cfg.min_inliers or ratio < cfg.min_inlier_ratio:
            return self._reject(MotionState.UNCERTAIN, 'weak_inlier_support', cfg, **data)
        if residual > cfg.rotation_rmse_valid_px:
            return self._reject(MotionState.UNCERTAIN, 'rotation_residual_exceeds_valid_threshold', cfg, **data)
        if cells < cfg.min_occupied_cells or coverage < cfg.min_coverage_fraction:
            return self._reject(MotionState.UNCERTAIN, 'poor_spatial_coverage', cfg, **data)
        if np.isfinite(innovation) and innovation > cfg.max_innovation_deg:
            return self._reject(MotionState.UNCERTAIN, 'temporal_innovation', cfg, **data)
        self.update_rotation(R)
        self.last_valid_increment = R.copy()
        self.consecutive_failures = 0
        return MotionDecision(MotionState.VALID, 'accepted', True, **data)

    def _reject(self, state, reason, cfg, **details):
        self.consecutive_failures += 1
        # A stale increment must not block recovery after a missed frame.
        if self.consecutive_failures >= cfg.invalid_after_failures:
            self.last_valid_increment = None
            # Preserve last reliable absolute orientation; feature tracks can
            # independently repopulate on subsequent frames.
        return MotionDecision(state, reason, False, **details)

    @staticmethod
    def _mask_reliability(status):
        """Conservative tracking-mask reliability factor (NOT segmentation accuracy)."""
        weights = {
            'DETECTED': 1.0, 'WARPED': 0.8, 'FALLBACK': 0.55,
            'UNRELIABLE': 0.35, 'EXPIRED': 0.35, 'EMPTY': 0.65,
            'DISABLED': 0.65,
        }
        return float(weights.get(str(status).upper(), 0.35))

    @staticmethod
    def compute_tracking_confidence(decision, *, mask_status='DISABLED', config=None):
        """Deterministic 0..1 reliability score, NOT a calibrated probability.

        Only comparable within the same configuration and dataset. The policy's
        state is authoritative: scoring cannot turn a rejection into acceptance.
        """
        cfg = config or AcceptanceConfig()
        if decision.state == MotionState.INVALID:
            return 0.0
        if not np.isfinite(decision.rotation_rmse_px):
            return 0.0
        # Bounded, dimensionless individual contributions.
        inlier = np.clip(decision.inlier_ratio / max(cfg.min_inlier_ratio, 1e-9), 0, 1)
        support = np.clip(decision.inliers / max(cfg.min_inliers * 2, 1), 0, 1)
        residual = np.clip(1.0 - decision.rotation_rmse_px / max(cfg.rotation_rmse_invalid_px, 1e-9), 0, 1)
        spatial_cells = np.clip(decision.occupied_cells / 16.0, 0, 1)
        spatial_area = np.clip(decision.coverage_fraction / 0.5, 0, 1)
        spatial = 0.5 * (spatial_cells + spatial_area)
        if np.isfinite(decision.innovation_deg):
            temporal = np.clip(1.0 - decision.innovation_deg / max(cfg.max_innovation_deg * 2, 1e-9), 0, 1)
        else:
            temporal = 0.75  # unknown, not perfect consistency
        weighted = (0.2 * inlier + 0.15 * support + 0.3 * residual
                    + 0.2 * spatial + 0.15 * temporal)
        reliability = BackgroundFeatureTracker._mask_reliability(mask_status)
        score = float(np.clip(weighted * (0.75 + 0.25 * reliability), 0, 1))
        if decision.state == MotionState.UNCERTAIN:
            return min(score, 0.49)
        return max(0.5, score)  # validated rotation; NOT guaranteed correct

    def validate_rotation(self, old_points, new_points, *, frame_size,
                          config=None, mask_status='DISABLED'):
        """T04 authoritative policy + T08 descriptive confidence diagnostics."""
        cfg = config or AcceptanceConfig()
        decision = self._validate_rotation_policy(
            old_points, new_points, frame_size=frame_size, config=cfg)
        return replace(
            decision,
            confidence=self.compute_tracking_confidence(
                decision, mask_status=mask_status, config=cfg),
            mask_reliability=self._mask_reliability(mask_status),
        )
