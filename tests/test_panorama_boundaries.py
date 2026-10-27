"""T11: deterministic longitude seam and latitude pole regression checks."""
import cv2
import numpy as np
import pytest

from panovideo.calibration import CameraCalibration
from panovideo.geometry import PanoramaRenderer, rotation_matrix


def patterned_panorama(h=32, w=64):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = np.arange(w, dtype=np.uint8)[None, :] * 3
    img[:, :, 1] = np.arange(h, dtype=np.uint8)[:, None] * 5
    img[:, :, 2] = 100
    return img


def center_renderer(panorama, calibration=None):
    if calibration is None:
        calibration = CameraCalibration(1, 1, 1., 1., 0., 0.)
    return PanoramaRenderer(panorama, 1, 1, calibration=calibration)


def expected_sample(panorama, ray):
    """Independent bilinear equirectangular sampling; horizontal periodic only."""
    h, w = panorama.shape[:2]
    lon = np.arctan2(ray[0], ray[2]); lat = np.arctan2(-ray[1], np.hypot(ray[0], ray[2]))
    x = ((lon / (2 * np.pi) + .5) % 1.) * w
    y = np.clip((.5 - lat / np.pi) * (h - 1), 0., h - 1.)
    x0 = int(np.floor(x)) % w; x1 = (x0 + 1) % w
    y0 = int(np.floor(y)); y1 = min(y0 + 1, h - 1)
    ax = x - np.floor(x); ay = y - y0
    return ((1-ay)*((1-ax)*panorama[y0,x0]+ax*panorama[y0,x1])
            +ay*((1-ax)*panorama[y1,x0]+ax*panorama[y1,x1]))


@pytest.mark.parametrize('yaw', [-540., -360., -180., -179.9, 0., 179.9, 180., 360., 540.])
def test_yaw_wraparound_360(yaw):
    img = patterned_panorama()
    renderer = center_renderer(img)
    np.testing.assert_array_equal(renderer.render(yaw, 0, 0), renderer.render(yaw + 360., 0, 0))


@pytest.mark.parametrize('yaw', [179.99, -179.99, 180., -180.])
def test_seam_interpolation_matches_periodic_reference(yaw):
    img = patterned_panorama()
    renderer = center_renderer(img)
    ray = rotation_matrix(yaw, 0, 0).astype(np.float64) @ np.array([0., 0., 1.])
    got = renderer.render(yaw, 0, 0)[0,0].astype(float)
    np.testing.assert_allclose(got, expected_sample(img.astype(float), ray), atol=2.0)


def test_seam_smoothly_connects_last_and_first_column():
    # When sampling just across the seam, neither value should jump to the
    # distant centre of the panorama or default to an out-of-bounds colour.
    img = np.zeros((16, 64, 3), dtype=np.uint8)
    img[:, :1, 0] = 30
    img[:, -1:, 0] = 210
    r = center_renderer(img)
    left = int(r.render(179.99)[0,0,0]); right = int(r.render(-179.99)[0,0,0])
    assert abs(left-right) <= 3
    assert 29 <= left <= 31
    # From the last texel towards the first, the sampler blends through
    # the wrapped neighbour instead of wrapping vertically or clamping.
    edge = int(r.render(179.0)[0, 0, 0])
    assert 30 < edge < 210


@pytest.mark.parametrize('pitch,expected_row', [(90., 0), (-90., -1)])
def test_poles_use_respective_edge_row_without_vertical_wrap(pitch, expected_row):
    img = np.zeros((30, 60, 3), np.uint8)
    img[0,:,1] = 241
    img[-1,:,2] = 223
    img[1:-1,:,0] = 12
    r = center_renderer(img)
    pixel = r.render(0, pitch, 0)[0,0]
    np.testing.assert_array_equal(pixel, img[expected_row,30])


@pytest.mark.parametrize('pitch', [89.0, 89.9, 90.0, 90.1, -89.0, -89.9, -90.0, -90.1, 180.0, -180.0])
def test_extreme_pitch_is_finite_and_in_bounds(pitch):
    img = patterned_panorama(40, 80)
    r = PanoramaRenderer(img, 61, 39, calibration=CameraCalibration(61,39,30.,28.,30.,19.))
    output = r.render(15., pitch, 6.)
    assert output.shape == (39, 61, 3)
    assert output.dtype == np.uint8
    assert np.isfinite(output).all()


def test_calibrated_off_centre_principal_point_matches_reference():
    img = patterned_panorama(48,96)
    calib = CameraCalibration(17, 11, 28., 22., 5.3, 7.8)
    r = PanoramaRenderer(img, 17, 11, calibration=calib)
    yaw,pitch,roll=37.,-24.,13.
    out = r.render(yaw,pitch,roll)
    matrix = rotation_matrix(yaw,pitch,roll)
    for y,x in [(0,0),(4,7),(10,16),(9,2)]:
        ray = matrix.astype(float) @ r.rays[y*17+x].astype(float)
        np.testing.assert_allclose(out[y,x].astype(float), expected_sample(img.astype(float),ray),atol=2.)


def test_renderer_never_uses_vertical_wrap_border(monkeypatch):
    img=patterned_panorama()
    seen={}
    original=cv2.remap
    def spy(src,map_x,map_y,interpolation,**kwargs):
        seen['src']=src
        seen['borderMode']=kwargs['borderMode']
        seen['ymin']=float(np.min(map_y)); seen['ymax']=float(np.max(map_y))
        return original(src,map_x,map_y,interpolation,**kwargs)
    monkeypatch.setattr(cv2,'remap',spy)
    PanoramaRenderer(img,17,11).render(122,89.9,65)
    assert seen['borderMode']==cv2.BORDER_REPLICATE
    assert seen['src'].shape[0]==img.shape[0]
    assert seen['src'].shape[1]==img.shape[1]+2
    assert seen['ymin']>=0 and seen['ymax']<=img.shape[0]-1
