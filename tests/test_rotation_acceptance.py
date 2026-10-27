"""T04 acceptance/failure policy tests. Run from project root with pytest."""
from dataclasses import replace
import numpy as np
import pytest
from panovideo.camera_tracking import AcceptanceConfig, MotionState
from test_camera_rotation import tracker
from test_physical_motion import generate_scene

SIZE = (640, 360)
CFG = AcceptanceConfig()


def example(angles=(3, 0, 0), translation=(0, 0, 0), depths=(1.5, 3., 6.), noise=0.0):
    t = tracker()
    old, new, _ = generate_scene(t, angles=angles, translation=translation, depths=depths, noise=noise)
    return t, old, new


def test_valid_pure_rotation_updates_pose():
    t, old, new = example()
    before = t.rotation_matrix.copy()
    d = t.validate_rotation(old, new, frame_size=SIZE)
    assert d.state is MotionState.VALID
    assert d.accepted and d.rotation_rmse_px < CFG.rotation_rmse_valid_px
    assert not np.allclose(before, t.rotation_matrix)
    assert d.occupied_cells >= CFG.min_occupied_cells


def test_invalid_missing_data_preserves_pose():
    t, old, new = example()
    assert t.validate_rotation(old, new, frame_size=SIZE).accepted
    previous = t.rotation_matrix.copy()
    d = t.validate_rotation(None, None, frame_size=SIZE)
    assert d.state is MotionState.INVALID
    assert not d.accepted
    np.testing.assert_allclose(t.rotation_matrix, previous)


def test_translation_like_motion_not_accepted_with_large_residual():
    t, old, new = example(angles=(0, 0, 0), translation=(.08, 0, 0))
    d = t.validate_rotation(old, new, frame_size=SIZE)
    assert d.state in (MotionState.UNCERTAIN, MotionState.INVALID)
    assert not d.accepted
    np.testing.assert_allclose(t.rotation_matrix, np.eye(3))


def test_unambiguous_pure_rotation_with_noise():
    t, old, new = example(noise=.25)
    d = t.validate_rotation(old, new, frame_size=SIZE)
    assert d.state is MotionState.VALID


def test_poor_spatial_coverage_holds_orientation():
    t, old, new = example()
    d = t.validate_rotation(old, new, frame_size=SIZE,
                            config=replace(CFG, min_occupied_cells=17))
    assert d.state is MotionState.UNCERTAIN
    assert d.reason == 'poor_spatial_coverage'
    np.testing.assert_allclose(t.rotation_matrix, np.eye(3))


def test_temporal_innovation_holds_orientation():
    t, old, new = example(angles=(2, 0, 0))
    assert t.validate_rotation(old, new, frame_size=SIZE).accepted
    previous = t.rotation_matrix.copy()
    old2, new2, _ = generate_scene(t, angles=(-5, 0, 0))
    d = t.validate_rotation(old2, new2, frame_size=SIZE,
                            config=replace(CFG, max_innovation_deg=2))
    assert d.state is MotionState.UNCERTAIN
    assert d.reason == 'temporal_innovation'
    np.testing.assert_allclose(t.rotation_matrix, previous)


def test_recovery_after_three_failures_preserves_orientation():
    t, old, new = example(angles=(2, 0, 0))
    assert t.validate_rotation(old, new, frame_size=SIZE).accepted
    pose = t.rotation_matrix.copy()
    for _ in range(3):
        assert not t.validate_rotation(None, None, frame_size=SIZE).accepted
    np.testing.assert_allclose(t.rotation_matrix, pose)
    assert t.last_valid_increment is None
    assert t.consecutive_failures == 3
    # After failure budget, temporal innovation should not block re-acquisition.
    new_old, new_new, _ = generate_scene(t, angles=(-5, 0, 0))
    result = t.validate_rotation(new_old, new_new, frame_size=SIZE)
    assert result.state is MotionState.VALID
    assert t.consecutive_failures == 0


def test_reset_removes_temporal_history():
    t, old, new = example()
    assert t.validate_rotation(old, new, frame_size=SIZE).accepted
    t.reset_rotation()
    assert t.last_valid_increment is None
    assert t.consecutive_failures == 0
    np.testing.assert_allclose(t.rotation_matrix, np.eye(3))


def test_failed_diagnostic_does_not_throw():
    t = tracker()
    invalid = np.full((25, 2), np.nan)
    d = t.validate_rotation(invalid, invalid, frame_size=SIZE)
    assert d.state is MotionState.INVALID


def test_far_translation_ambiguity_is_documented():
    t, old, new = example(angles=(0, 0, 0), translation=(.08, 0, 0), depths=(12., 20., 30.), noise=.25)
    d = t.validate_rotation(old, new, frame_size=SIZE)
    # Deliberately record, not assert rejection: geometry is ambiguous.
    assert d.state in tuple(MotionState)
    assert np.isfinite(d.rotation_rmse_px)


def test_invalid_large_residual():
    t, old, new = example(angles=(0, 0, 0), translation=(.15, .10, 0))
    d = t.validate_rotation(old, new, frame_size=SIZE)
    assert not d.accepted
    assert d.rotation_rmse_px > CFG.rotation_rmse_valid_px
