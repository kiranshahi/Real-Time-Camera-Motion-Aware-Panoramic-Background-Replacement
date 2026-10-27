from __future__ import annotations
import csv
from pathlib import Path
import cv2
import numpy as np
import pytest
from panovideo.camera_tracking import BackgroundFeatureTracker
from panovideo.geometry import rotation_matrix

WIDTH, HEIGHT, HFOV = 640, 360, 70.0
ROTATION_TOLERANCE_DEG = 0.5
RNG_SEED = 20261025
CASES = [
    ("identity", 0, 0, 0),
    ("yaw_positive", 5, 0, 0),
    ("yaw_negative", -5, 0, 0),
    ("pitch", 0, 4, 0),
    ("roll", 0, 0, 5),
    ("combined", 4, -3, 2),
]


def geodesic_error_deg(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Shortest angular distance between two SO(3) rotation matrices."""
    c = np.clip((np.trace(estimate @ truth.T) - 1.0) / 2.0, -1, 1)
    return float(np.rad2deg(np.arccos(c)))


def tracker() -> BackgroundFeatureTracker:
    instance = BackgroundFeatureTracker()
    instance.set_camera_intrinsics(WIDTH, HEIGHT, HFOV)
    return instance


def camera_grid(K: np.ndarray) -> np.ndarray:
    """Spread 192 stable image features over the image, away from image boundaries."""
    xs, ys = np.meshgrid(np.linspace(90, WIDTH - 90, 16), np.linspace(65, HEIGHT - 65, 12))
    return np.column_stack((xs.ravel(), ys.ravel())).astype(np.float64)


def apply_rotation(old_pixels: np.ndarray, K: np.ndarray, R_old_to_new: np.ndarray):
    homogeneous = np.column_stack((old_pixels, np.ones(len(old_pixels))))
    rays = (np.linalg.inv(K) @ homogeneous.T).T
    rotated = (R_old_to_new @ rays.T).T
    projected = (K @ rotated.T).T
    valid = projected[:, 2] > 0
    next_pixels = projected[:, :2] / projected[:, 2:3]
    valid &= (next_pixels[:, 0] >= 0) & (next_pixels[:, 0] < WIDTH)
    valid &= (next_pixels[:, 1] >= 0) & (next_pixels[:, 1] < HEIGHT)
    return old_pixels[valid].astype(np.float32), next_pixels[valid].astype(np.float32)


def run_case(name, old, new, expected_rotation, output_dir: Path):
    instance = tracker()
    estimated, inliers = instance.estimate_rotation(old, new)
    n = len(old)
    inlier_ratio = inliers / n if n else 0
    error = geodesic_error_deg(estimated, expected_rotation) if estimated is not None else float('nan')
    estimated_angle = geodesic_error_deg(estimated, np.eye(3)) if estimated is not None else float('nan')
    # Mirror acceptance criteria from main(6).py (not part of estimate_rotation).
    accepted = (estimated is not None and inliers >= 20 and inlier_ratio >= .60 and estimated_angle <= 10.0)
    return dict(case=name, matches=n, inliers=inliers, inlier_ratio=round(inlier_ratio, 5),
                error_deg=round(error, 6), estimated_motion_deg=round(estimated_angle, 6),
                rotation_accepted=bool(accepted))


@pytest.mark.parametrize("name,yaw,pitch,roll", CASES)
def test_known_rotations(name, yaw, pitch, roll, records, output_dir):
    instance = tracker()
    # geometry returns a camera-to-world matrix. Its transpose maps old-camera rays
    # into new-camera coordinates, which is what estimate_rotation estimates.
    camera_to_world = rotation_matrix(yaw, pitch, roll).astype(np.float64)
    expected = camera_to_world.T
    old, new = apply_rotation(camera_grid(instance.camera_matrix), instance.camera_matrix, expected)
    result = run_case(name, old, new, expected, output_dir)
    records.append(result)
    assert result['matches'] >= 20
    assert result['rotation_accepted'], f"Tracking rejected a valid camera rotation: {result}"
    assert result['error_deg'] <= ROTATION_TOLERANCE_DEG, result


@pytest.mark.parametrize('name,dx,dy', [
    ('translation_horizontal_10px', 10., 0.),
    ('translation_vertical_10px', 0., 10.),
    ('translation_diagonal_20_15px', 20., 15.),
])
def test_translation_only_diagnostics(name, dx, dy, records, output_dir):
    """Characterisation, not a pass/fail assertion that the existing estimator is robust.

    Uniform pixel translation is an image-space surrogate (not a metric 3D camera
    translation with depth/parallax). Ground-truth camera rotation is identity.
    """
    instance = tracker()
    old = camera_grid(instance.camera_matrix).astype(np.float32)
    new = old + np.array([dx, dy], dtype=np.float32)
    result = run_case(name, old, new, np.eye(3), output_dir)
    result['false_positive_rotation'] = bool(result['rotation_accepted'] and result['estimated_motion_deg'] > 0.1)
    records.append(result)
    assert result['matches'] == len(old)
    assert result['inliers'] >= 0
    # Intentional: don't assert that it rejects translation in baseline T01.


@pytest.fixture(scope='module')
def output_dir():
    directory = Path('experiments')
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture(scope='module')
def records():
    rows = []
    yield rows
    if rows:
        output = Path('experiments') / 'synthetic_baseline.csv'
        fields = ['case', 'matches', 'inliers', 'inlier_ratio', 'error_deg', 'estimated_motion_deg', 'rotation_accepted', 'false_positive_rotation']
        with output.open('w', newline='', encoding='utf-8') as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nT01 baseline results written to {output}')
        for row in rows:
            print(row)
