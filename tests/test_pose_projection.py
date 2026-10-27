"""T05 end-to-end SO(3) and panoramic projection convention tests.

Rotation convention: x right, y down, z forward; R_cw = Ry(yaw) Rx(pitch) Rz(roll).
The tracker accumulates R_initial_to_current = R_cw.T; renderer expects R_cw.
Run: python -m pytest -q tests/test_pose_projection.py
"""
import numpy as np
import cv2
import pytest

from panovideo.camera_tracking import BackgroundFeatureTracker
from panovideo.geometry import PanoramaRenderer, rotation_matrix
from panovideo.pose import ManualPoseProvider


def angle_distance(a, b):
    c = np.clip((np.trace(a @ b.T) - 1) / 2, -1, 1)
    return float(np.rad2deg(np.arccos(c)))


@pytest.mark.parametrize('yaw,pitch,roll', [
    (0, 0, 0), (5, 0, 0), (-7, 0, 0), (0, 6, 0),
    (0, -6, 0), (0, 0, 8), (0, 0, -8), (4, -3, 2), (-9, 7, -5),
])
def test_tracker_orientation_to_renderer_matrix(yaw, pitch, roll):
    """Relative world-to-camera rotation must invert to renderer camera-to-world."""
    t = BackgroundFeatureTracker()
    truth = rotation_matrix(yaw, pitch, roll).astype(np.float64)
    t.update_rotation(truth.T)
    t.smoothing_alpha = 1.0
    t.smooth_rotation()
    recovered = t.get_camera_orientation()
    assert angle_distance(rotation_matrix(*recovered), truth) < 0.02
    assert np.allclose(recovered, (yaw, pitch, roll), atol=0.02)


@pytest.mark.parametrize('yaw,pitch,roll', [(0, 0, 0), (11, 0, 0), (0, 9, 0), (0, 0, -9), (8, -6, 4)])
def test_pose_provider_preserves_tracker_angles(yaw, pitch, roll):
    t = BackgroundFeatureTracker()
    t.update_rotation(rotation_matrix(yaw, pitch, roll).astype(np.float64).T)
    t.smoothing_alpha = 1.0
    t.smooth_rotation()
    p = ManualPoseProvider().update(None, 0.0, controls=t.get_camera_orientation())
    assert angle_distance(rotation_matrix(p.yaw, p.pitch, p.roll), rotation_matrix(yaw, pitch, roll)) < 0.02


def test_reset_clears_rotation_and_tracking_state():
    t = BackgroundFeatureTracker()
    t.update_rotation(rotation_matrix(9, 3, -2).astype(np.float64).T)
    t.smooth_rotation()
    t.previous_gray = np.zeros((4, 4), np.uint8)
    t.previous_points = np.zeros((4, 1, 2), np.float32)
    t.reset_rotation()
    assert np.allclose(t.rotation_matrix, np.eye(3), atol=1e-12)
    assert np.allclose(t.smoothed_rotation, np.eye(3), atol=1e-12)
    assert t.previous_gray is None and t.previous_points is None
    assert np.allclose(t.get_camera_orientation(), (0, 0, 0), atol=1e-8)


def test_smoothing_follows_so3_geodesic_and_converges():
    t = BackgroundFeatureTracker()
    target = rotation_matrix(14, -8, 5).astype(np.float64).T
    t.smoothing_alpha = 0.25
    t.update_rotation(target)
    initial = angle_distance(t.smoothed_rotation, target)
    t.smooth_rotation()
    first = angle_distance(t.smoothed_rotation, target)
    assert first == pytest.approx(initial * 0.75, abs=0.002)
    for _ in range(35):
        t.smooth_rotation()
    assert angle_distance(t.smoothed_rotation, target) < 0.02
    assert np.allclose(t.smoothed_rotation.T @ t.smoothed_rotation, np.eye(3), atol=1e-9)
    assert np.linalg.det(t.smoothed_rotation) == pytest.approx(1.0, abs=1e-9)


def test_relative_rotations_compose_in_correct_order():
    t = BackgroundFeatureTracker()
    old_world_to_camera = np.eye(3)
    for pose in [(4, -2, 1), (9, -1, 3), (-3, 5, 2)]:
        cw = rotation_matrix(*pose).astype(np.float64)
        new_world_to_camera = cw.T
        relative_old_to_new = new_world_to_camera @ old_world_to_camera.T
        t.update_rotation(relative_old_to_new)
        old_world_to_camera = new_world_to_camera
    t.smoothing_alpha = 1.0
    t.smooth_rotation()
    assert angle_distance(rotation_matrix(*t.get_camera_orientation()), cw) < 0.02


def make_panorama(w=720, h=360):
    # A periodic gradient in longitude; red is sin(lon), green is cos(lon).
    lon = (np.arange(w) / w - 0.5) * 2 * np.pi
    panorama = np.zeros((h, w, 3), np.uint8)
    panorama[:, :, 2] = np.rint(128 + 110 * np.sin(lon)).astype(np.uint8)
    panorama[:, :, 1] = np.rint(128 + 110 * np.cos(lon)).astype(np.uint8)
    return panorama


def center_longitude_of_render(yaw, pitch, roll):
    renderer = PanoramaRenderer(make_panorama(), 101, 61, 70)
    output = renderer.render(yaw, pitch, roll)
    center = output[30, 50].astype(float)
    return np.rad2deg(np.arctan2(center[2] - 128, center[1] - 128))


@pytest.mark.parametrize('yaw', [-30, -15, 0, 15, 30])
def test_center_pixel_looks_toward_yaw(yaw):
    # The analytical centre ray is (0,0,1) on odd-sized images.
    actual = center_longitude_of_render(yaw, 0, 0)
    assert actual == pytest.approx(yaw, abs=0.6)


def test_pitch_changes_center_latitude_and_roll_does_not():
    renderer = PanoramaRenderer(make_panorama(), 101, 61, 70)
    ray = renderer.rays[(61 // 2) * 101 + 101 // 2]
    for yaw, pitch, roll in [(0, 10, 0), (0, -10, 0), (0, 0, 15), (8, 6, -7)]:
        world = ray @ rotation_matrix(yaw, pitch, roll).T
        latitude = np.rad2deg(np.arctan2(-world[1], np.hypot(world[0], world[2])))
        assert latitude == pytest.approx(pitch, abs=0.1)


def test_renderer_matches_analytic_spherical_sampling_interior():
    panorama = make_panorama(1440, 720)
    renderer = PanoramaRenderer(panorama, 101, 61, 70)
    for pose in [(0, 0, 0), (17, -11, 8), (-28, 7, -4)]:
        output = renderer.render(*pose)
        R = rotation_matrix(*pose)
        for y, x in [(30, 50), (20, 30), (42, 70)]:
            ray = renderer.rays[y * 101 + x]
            world = ray @ R.T
            lon = np.arctan2(world[0], world[2])
            lat = np.arctan2(-world[1], np.hypot(world[0], world[2]))
            mx = (lon / (2 * np.pi) + 0.5) * panorama.shape[1]
            my = (0.5 - lat / np.pi) * (panorama.shape[0] - 1)
            # Bilinear channel error is at most approximately one quantisation step.
            expected = cv2.remap(panorama, np.array([[mx]], np.float32), np.array([[my]], np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)[0, 0]
            np.testing.assert_allclose(output[y, x].astype(float), expected.astype(float), atol=1.0)
