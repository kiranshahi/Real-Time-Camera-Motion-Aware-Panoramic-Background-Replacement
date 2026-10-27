"""T06 deterministic replenishment tests, no camera/model required."""
import cv2
import numpy as np
import pytest

from panovideo.camera_tracking import BackgroundFeatureTracker


def make_frame(width=320, height=240):
    # Corner-rich synthetic tiles, seeded deterministically.
    rng = np.random.default_rng(2026)
    gray = rng.integers(0, 256, (height, width), dtype=np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def minimum_distance(points):
    pts = np.asarray(points).reshape(-1, 2)
    if len(pts) < 2:
        return float('inf')
    delta = pts[:, None, :] - pts[None, :, :]
    d = np.sum(delta ** 2, axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.sqrt(np.min(d)))


def occupied_cells(points, shape, rows=4, cols=4):
    h, w = shape
    pts = np.asarray(points).reshape(-1, 2)
    return {(min(int(y * rows / h), rows - 1), min(int(x * cols / w), cols - 1))
            for x, y in pts}


def test_initial_detection_enforces_spacing_and_spread():
    tracker = BackgroundFeatureTracker()
    frame = make_frame()
    pts = tracker.detect_features(frame)
    assert pts is not None
    assert pts.dtype == np.float32 and pts.shape[1:] == (1, 2)
    assert len(pts) <= 300
    assert minimum_distance(pts) >= tracker.feature_spacing_px - 1e-4
    assert len(occupied_cells(pts, frame.shape[:2])) == 16


def test_mask_excludes_top_half_and_preserves_spacing():
    tracker = BackgroundFeatureTracker()
    frame = make_frame()
    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[:120] = 255
    pts = tracker.detect_features(frame, mask)
    assert pts is not None
    assert np.all(pts.reshape(-1, 2)[:, 1] >= 120)
    assert len(occupied_cells(pts, frame.shape[:2])) == 8
    assert minimum_distance(pts) >= 10 - 1e-4


def test_any_nonzero_foreground_is_excluded():
    tracker = BackgroundFeatureTracker()
    frame = make_frame()
    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[:, :160] = 1
    pts = tracker.detect_features(frame, mask)
    assert pts is not None
    assert np.all(pts.reshape(-1, 2)[:, 0] >= 160)


def test_existing_points_and_new_points_are_spaced():
    tracker = BackgroundFeatureTracker()
    frame = make_frame()
    existing = np.array([[40., 40.], [160., 120.], [280., 200.]], np.float32).reshape(-1, 1, 2)
    fresh = tracker.detect_features(frame, existing_points=existing, target=35)
    assert fresh is not None and len(fresh) == 35
    combined = np.concatenate([existing, fresh])
    assert minimum_distance(combined) >= 10 - 1e-4


def test_replenishment_stops_exactly_at_target_and_preserves_tracks(monkeypatch):
    tracker = BackgroundFeatureTracker()
    frame = make_frame()
    base = tracker.detect_features(frame, target=20)
    assert len(base) == 20
    tracker.previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracker.previous_points = base.copy()
    # Mock LK with perfect forward/backward correspondence; deterministic, no numerical jitter.
    calls = [0]
    def fake_lk(prev, current, points, next_points, **kwargs):
        calls[0] += 1
        status = np.ones((len(points), 1), dtype=np.uint8)
        return points.copy(), status, np.zeros((len(points), 1), np.float32)
    monkeypatch.setattr(cv2, 'calcOpticalFlowPyrLK', fake_lk)
    old, new = tracker.track_features(frame)
    assert calls[0] == 2
    assert len(old) == len(new) == 20
    assert len(tracker.previous_points) == tracker.min_tracked_features == 80
    np.testing.assert_array_equal(tracker.previous_points[:20].reshape(-1, 2), new)
    assert minimum_distance(tracker.previous_points) >= 10 - 1e-4
    assert len(occupied_cells(tracker.previous_points, frame.shape[:2])) >= 12


def test_replenishment_respects_new_foreground(monkeypatch):
    tracker = BackgroundFeatureTracker()
    frame = make_frame()
    tracker.previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracker.previous_points = np.array([[[20., 20.]], [[40., 40.]]] * 6, np.float32)
    def fake_lk(prev, current, points, next_points, **kwargs):
        return points.copy(), np.ones((len(points), 1), np.uint8), None
    monkeypatch.setattr(cv2, 'calcOpticalFlowPyrLK', fake_lk)
    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[:120] = 255
    old, new = tracker.track_features(frame, mask)
    assert len(old) == 0  # all existing tracks in foreground are discarded
    assert tracker.previous_points is not None
    assert np.all(tracker.previous_points.reshape(-1, 2)[:, 1] >= 120)
    assert len(tracker.previous_points) == 80


def test_candidate_selection_balances_cells_and_checks_new_to_new_spacing():
    tracker = BackgroundFeatureTracker()
    mask = np.full((80, 80), 255, np.uint8)
    # deliberately interleave stronger points from the first cell before others
    candidates = np.array([[5, 5], [5.5, 5.5], [6, 6], [25, 5],
                           [5, 25], [25, 25], [7, 7]], np.float32)
    selected = tracker._select_distributed(candidates, np.empty((0, 2)),
                                            (80, 80), mask, 4)
    pts = selected.reshape(-1, 2)
    assert len(pts) == 4
    assert minimum_distance(pts) >= 10
    assert len(occupied_cells(pts, (80, 80))) == 4


def test_candidate_selection_rejects_foreground_and_invalid_points():
    tracker = BackgroundFeatureTracker()
    mask = np.full((80, 80), 255, np.uint8)
    mask[25, 25] = 0
    candidates = np.array([[25, 25], [-1, 3], [np.nan, 3], [12, 12], [65, 65]], np.float32)
    pts = tracker._select_distributed(candidates, np.empty((0, 2)), (80, 80), mask, 10)
    assert len(pts) == 2
    assert np.isfinite(pts).all()


def test_no_features_is_handled_cleanly():
    tracker = BackgroundFeatureTracker()
    frame = np.zeros((120, 160, 3), np.uint8)
    assert tracker.detect_features(frame) is None
    assert tracker.detect_features(make_frame(), np.full((240, 320), 255, np.uint8)) is None


def test_exclusion_mask_shape_validation():
    tracker = BackgroundFeatureTracker()
    with pytest.raises(ValueError, match='dimensions'):
        tracker.detect_features(make_frame(), np.zeros((5, 5), np.uint8))


def test_deterministic_repeated_detection():
    tracker = BackgroundFeatureTracker()
    frame = make_frame()
    np.testing.assert_array_equal(tracker.detect_features(frame), tracker.detect_features(frame))
