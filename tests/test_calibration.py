"""T10: shared camera intrinsics and optional rectification, deterministic."""
import json
import cv2
import numpy as np
import pytest
from panovideo.calibration import CameraCalibration
from panovideo.geometry import PanoramaRenderer, rotation_matrix
from panovideo.camera_tracking import BackgroundFeatureTracker


def camera():
    return CameraCalibration(160, 120, 125.0, 143.0, 77.0, 58.0)


def test_shared_matrix():
    c = camera()
    t = BackgroundFeatureTracker()
    np.testing.assert_allclose(t.set_calibration(c), c.matrix)
    p = PanoramaRenderer(np.zeros((120, 240, 3), np.uint8), 160, 120, calibration=c)
    assert p.calibration is c
    np.testing.assert_allclose(p.rays[58 * 160 + 77], [0, 0, 1], atol=1e-8)


def test_anisotropic_focal_lengths():
    c = camera()
    p = PanoramaRenderer(np.zeros((120, 240, 3), np.uint8), 160, 120, calibration=c)
    ray = p.rays[88 * 160 + 102]
    np.testing.assert_allclose(ray[:2] / ray[2], [25 / 125, 30 / 143], atol=1e-7)


@pytest.mark.parametrize('pose', [(0,0,0), (5,0,0), (0,4,0), (0,0,-3), (4,-3,2)])
def test_rotation_reprojection_with_shared_K(pose):
    c = camera()
    xs, ys = np.meshgrid(np.arange(15, 145, 15), np.arange(15, 105, 15))
    xy = np.column_stack((xs.ravel(), ys.ravel())).astype(np.float64)
    homogeneous = np.column_stack((xy, np.ones(len(xy))))
    truth = rotation_matrix(*pose).astype(np.float64).T  # initial->current
    transformed = (c.matrix @ truth @ np.linalg.inv(c.matrix) @ homogeneous.T).T
    new = (transformed[:,:2] / transformed[:,2,None]).astype(np.float32)
    t = BackgroundFeatureTracker(); t.set_calibration(c)
    result, count = t.estimate_rotation(xy.astype(np.float32), new)
    assert result is not None and count >= 20
    err = cv2.Rodrigues(result @ truth.T)[0]
    assert np.linalg.norm(err) < np.deg2rad(.05)
    metrics = t.diagnose_rotation_fit(xy.astype(np.float32), new)
    assert metrics['rotation_fit']['rmse_px'] < .05


def test_distortion_round_trip():
    c = CameraCalibration(160,120,125,143,77,58,(.08,-.02,.001,-.002,.006))
    undistorted = np.array([[20, 25], [130, 90], [77, 58], [56, 72]], dtype=np.float64)
    rays = np.column_stack(((undistorted[:,0]-c.cx)/c.fx, (undistorted[:,1]-c.cy)/c.fy, np.ones(len(undistorted))))
    distorted, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), c.matrix, c.coefficients)
    restored = cv2.undistortPoints(distorted, c.matrix, c.coefficients, P=c.matrix).reshape(-1,2)
    np.testing.assert_allclose(restored, undistorted, atol=1e-3)


def test_frame_rectification_and_shape():
    c = CameraCalibration(160,120,125,143,77,58,(.08,-.02,.001,-.002,.006))
    image = np.zeros((120,160,3), np.uint8)
    assert c.rectify(image).shape == image.shape
    a,b = c.rectification_maps()
    assert a.shape == b.shape == (120,160)
    with pytest.raises(ValueError): c.rectify(image[:50])


def test_no_distortion_no_frame_change():
    c = camera(); image = np.zeros((120,160,3), np.uint8)
    assert c.rectify(image) is image


def test_json_roundtrip(tmp_path):
    f = tmp_path / 'calibration.json'
    f.write_text(json.dumps(dict(width=160,height=120,fx=125,fy=143,cx=77,cy=58,distortion=[.08,-.02,.001,-.002,.006])))
    c = CameraCalibration.from_json(f,160,120)
    assert c.fx == 125 and len(c.distortion) == 5
    with pytest.raises(ValueError,match='resolution'):
        CameraCalibration.from_json(f,640,360)


@pytest.mark.parametrize('bad', [(-1,100), (0,100), (100,0)])
def test_bad_focal(bad):
    with pytest.raises(ValueError):
        CameraCalibration(160,120,bad[0],bad[1],77,58)


def test_bad_distortion():
    with pytest.raises(ValueError):
        CameraCalibration(160,120,125,143,77,58,(.1,.2,.3))


def test_legacy_hfov_compatibility():
    t = BackgroundFeatureTracker(); K = t.set_camera_intrinsics(640,360,70)
    c = CameraCalibration.from_hfov(640,360,70)
    np.testing.assert_array_equal(K,c.matrix)
