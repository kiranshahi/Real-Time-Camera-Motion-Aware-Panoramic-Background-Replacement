"""Deterministic single-primary-person association, independent of YOLO."""
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass(frozen=True)
class SubjectConfig:
    max_missing: int = 3
    min_similarity: float = 0.35
    box_weight: float = 0.35
    mask_weight: float = 0.40
    centre_weight: float = 0.25
    max_centre_distance: float = 0.30  # normalized by image diagonal
    minimum_overlap: float = 0.05


@dataclass
class PersonDetection:
    box: np.ndarray
    mask: np.ndarray
    confidence: float = 1.0


class SubjectTracker:
    """Maintains one target. On loss, holds its identity for max_missing frames.

    After expiry, a new target is selected near the centre. This is not re-identification.
    During ambiguous association, outputs no person rather than switching identity.
    """
    def __init__(self, config=None):
        self.config = config or SubjectConfig()
        if self.config.max_missing < 0:
            raise ValueError('max_missing must be nonnegative')
        self.reset()

    def reset(self):
        self.target = None
        self.missing = 0
        self.last_status = 'EMPTY'
        self.last_score = 0.0
        self.last_index = None

    @staticmethod
    def box_iou(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
        area_a = max(0., a[2]-a[0])*max(0., a[3]-a[1])
        area_b = max(0., b[2]-b[0])*max(0., b[3]-b[1])
        return float(intersection/max(area_a+area_b-intersection, 1e-8))

    @staticmethod
    def mask_iou(a, b):
        a = np.asarray(a) >= .5
        b = np.asarray(b) >= .5
        intersection = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(intersection/union) if union else 0.0

    def _score(self, previous, detection, frame_shape):
        h, w = frame_shape[:2]
        overlap_box = self.box_iou(previous.box, detection.box)
        overlap_mask = self.mask_iou(previous.mask, detection.mask)
        p = (previous.box[:2]+previous.box[2:])*.5
        q = (detection.box[:2]+detection.box[2:])*.5
        distance = float(np.linalg.norm(p-q)/np.hypot(h, w))
        if distance > self.config.max_centre_distance:
            return None
        if max(overlap_box, overlap_mask) < self.config.minimum_overlap:
            return None
        similarity = (self.config.box_weight*overlap_box + self.config.mask_weight*overlap_mask
                      + self.config.centre_weight*max(0., 1.-distance/max(self.config.max_centre_distance, 1e-9)))
        return similarity if similarity >= self.config.min_similarity else None

    def update(self, detections, frame_shape):
        """Return selected detection or None. Empty on ambiguous/missing frames.

        Caller supplies person detections only, all masks matching frame dimensions.
        """
        h, w = frame_shape[:2]
        for detection in detections:
            if detection.mask.shape != (h, w):
                raise ValueError('Detection mask dimensions must match frame')
            if not np.isfinite(detection.box).all() or np.asarray(detection.box).shape != (4,):
                raise ValueError('Invalid bounding box')
        self.last_index = None
        self.last_score = 0.0
        if self.target is None:
            if not detections:
                self.last_status = 'EMPTY'
                return None
            centre = np.array([w/2., h/2.])
            distances = [np.linalg.norm((d.box[:2]+d.box[2:])*.5-centre) for d in detections]
            idx = int(np.argmin(distances))
            self.target = PersonDetection(detections[idx].box.copy(), detections[idx].mask.copy(), detections[idx].confidence)
            self.missing = 0
            self.last_index = idx
            self.last_score = 1.
            self.last_status = 'ACQUIRED'
            return detections[idx]
        matches = [(score, i) for i, d in enumerate(detections) if (score := self._score(self.target, d, frame_shape)) is not None]
        matches.sort(key=lambda t: (-t[0], t[1]))
        # Prefer no selection over switching between equally plausible detections.
        if matches and (len(matches) == 1 or matches[0][0]-matches[1][0] > 0.05):
            score, idx = matches[0]
            d = detections[idx]
            self.target = PersonDetection(d.box.copy(), d.mask.copy(), d.confidence)
            self.missing = 0
            self.last_status = 'TRACKED'
            self.last_score = score
            self.last_index = idx
            return d
        self.missing += 1
        self.last_status = 'AMBIGUOUS' if matches else 'MISSING'
        if self.missing > self.config.max_missing:
            self.target = None
            self.last_status = 'EXPIRED'
        return None
