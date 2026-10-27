import csv
from pathlib import Path
import numpy as np
import pytest
from panovideo.geometry import rotation_matrix
from test_camera_rotation import WIDTH, HEIGHT, HFOV, camera_grid, apply_rotation, tracker, geodesic_error_deg

"""deterministic rotation-only fit diagnostics (includes noisy/outlier cases)."""

OUTPUT = Path('experiments/rotation_diagnostics.csv')
FIELDS = ['case', 'matches', 'inliers', 'inlier_ratio', 'rotation_error_deg',
          'homography_rmse_px', 'rotation_rmse_px', 'rmse_gap_px',
          'homography_p95_px', 'rotation_p95_px', 'rotation_accepted',
          'false_positive_rotation']


def synthetic(name, yaw=0, pitch=0, roll=0, noise=0, outlier_fraction=0, shift=None):
    t = tracker()
    old = camera_grid(t.camera_matrix)
    truth = rotation_matrix(yaw, pitch, roll).astype(np.float64).T
    if shift is None:
        old, new = apply_rotation(old, t.camera_matrix, truth)
    else:
        new = old + np.asarray(shift)
    rng = np.random.default_rng(93017)
    new = new.astype(np.float64)
    if noise:
        new += rng.normal(0, noise, new.shape)
    if outlier_fraction:
        n = round(outlier_fraction * len(new))
        indices = rng.choice(len(new), n, replace=False)
        new[indices] = np.column_stack((rng.uniform(0, WIDTH, n), rng.uniform(0, HEIGHT, n)))
    return t, old.astype(np.float32), new.astype(np.float32), truth


CASES = [
    ('identity', dict()),
    ('yaw_5', dict(yaw=5)),
    ('pitch_4', dict(pitch=4)),
    ('roll_5', dict(roll=5)),
    ('combined', dict(yaw=4, pitch=-3, roll=2)),
    ('yaw_noisy_0_5px', dict(yaw=5, noise=.5)),
    ('combined_noisy_1px', dict(yaw=4, pitch=-3, roll=2, noise=1.)),
    ('yaw_25pct_outliers', dict(yaw=5, noise=.5, outlier_fraction=.25)),
    ('combined_35pct_outliers', dict(yaw=4, pitch=-3, roll=2, noise=.75, outlier_fraction=.35)),
    ('translation_x_10px', dict(shift=(10, 0))),
    ('translation_y_10px', dict(shift=(0, 10))),
    ('translation_xy_20_15px', dict(shift=(20, 15))),
    ('translation_x_10px_noisy', dict(shift=(10, 0), noise=.5)),
]


@pytest.fixture(scope='module')
def records():
    rows = []
    yield rows
    OUTPUT.parent.mkdir(exist_ok=True, parents=True)
    with OUTPUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\ndiagnostic CSV: {OUTPUT}')


@pytest.mark.parametrize('case,params', CASES, ids=[name for name, _ in CASES])
def test_fit_comparison(case, params, records):
    t, old, new, truth = synthetic(case, **params)
    result = t.diagnose_rotation_fit(old, new)
    assert result is not None, case
    assert result['inliers'] >= 20
    H = result['homography_fit']
    R = result['rotation_fit']
    err = geodesic_error_deg(result['rotation'], truth)
    motion = geodesic_error_deg(result['rotation'], np.eye(3))
    accepted = result['inliers'] >= 20 and result['inlier_ratio'] >= .60 and motion <= 10.
    is_translation = params.get('shift') is not None
    records.append(dict(case=case, matches=result['matches'], inliers=result['inliers'],
                        inlier_ratio=round(result['inlier_ratio'], 5), rotation_error_deg=round(err, 6),
                        homography_rmse_px=round(H['rmse_px'], 6),
                        rotation_rmse_px=round(R['rmse_px'], 6), rmse_gap_px=round(result['rmse_gap_px'], 6),
                        homography_p95_px=round(H['p95_px'], 6), rotation_p95_px=round(R['p95_px'], 6),
                        rotation_accepted=accepted,
                        false_positive_rotation=bool(is_translation and accepted and motion > .1)))
    assert np.isfinite(R['rmse_px']) and np.isfinite(H['rmse_px'])
    if not is_translation:
        tolerance = 1.5 if params.get('outlier_fraction') else (1.0 if params.get('noise') else .5)
        assert err <= tolerance, (case, err)
        if not params.get('noise'):
            assert R['rmse_px'] < .1, (case, R)
    # Deliberately record translation results instead of asserting rejection.


def test_diagnostics_agrees_with_original_estimator():
    t, old, new, _ = synthetic('agreement', yaw=4, pitch=-3, roll=2)
    original, count = t.estimate_rotation(old, new)
    diagnostic = t.diagnose_rotation_fit(old, new)
    assert count == diagnostic['inliers']
    assert geodesic_error_deg(original, diagnostic['rotation']) < 1e-4


def test_insufficient_correspondences():
    t = tracker()
    assert t.diagnose_rotation_fit(np.zeros((5, 2)), np.zeros((5, 2))) is None
