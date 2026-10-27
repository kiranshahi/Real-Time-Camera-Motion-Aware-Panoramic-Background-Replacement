import csv
from pathlib import Path
import numpy as np
import pytest
from panovideo.geometry import rotation_matrix
from test_camera_rotation import tracker, geodesic_error_deg

"""
Multi-depth rigid 3D scene tests for conservative rotation gating.
Coordinate system: X right, Y down, Z forward; X2 = R_12 @ X1 + t_12.
Translation in METRES. Ground truth rotation is old-camera to new-camera.
"""


OUT = Path('experiments/physical_motion.csv')
FIELDS = ['case','category','depth_m','translation_m','noise_px','outlier_fraction','seed','matches','inliers','inlier_ratio',
          'true_rotation_deg','estimated_rotation_deg','rotation_error_deg','homography_rmse_px',
          'rotation_rmse_px','rmse_gap_px','old_accept','gated_accept','old_false_positive','gate_false_positive',
          'old_false_negative','gate_false_negative']
# Illustrative starting policy only: MUST calibrate on independent real footage.
RMSE_GATE_PX = 2.0


def project(K, xyz):
    p = xyz @ K.T
    return p[:, :2] / p[:, 2, None]


def generate_scene(t, angles=(0, 0, 0), translation=(0, 0, 0), depths=(1.5, 3., 6.),
                   noise=0., outliers=0., seed=101, n_per_plane=100):
    """3D points sampled across several FRONTOPARALLEL depth planes with shared rays.

    Independent sampling per depth: imperfectly planar scene; a real translation
    produces parallax as a function of inverse depth.
    """
    rng = np.random.default_rng(seed)
    K = t.camera_matrix
    R = rotation_matrix(*angles).astype(np.float64).T
    trans = np.array(translation, np.float64)
    xyz = []
    for z in depths:
        pixels = np.column_stack([rng.uniform(45, 595, n_per_plane),
                                  rng.uniform(35, 325, n_per_plane)])
        rays = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(K).T
        xyz.append(rays * z)
    xyz = np.concatenate(xyz)
    changed = xyz @ R.T + trans
    assert np.all(changed[:, 2] > 0)
    old = project(K, xyz)
    new = project(K, changed)
    if noise:
        old = old + rng.normal(0, noise, old.shape)
        new = new + rng.normal(0, noise, new.shape)
    if outliers:
        ids = rng.choice(len(new), round(len(new)*outliers), replace=False)
        new[ids] = np.column_stack([rng.uniform(0, 640, len(ids)), rng.uniform(0, 360, len(ids))])
    return old.astype('float32'), new.astype('float32'), R


def classify(t, old, new, truth, category):
    diag = t.diagnose_rotation_fit(old, new)
    assert diag is not None
    angle = geodesic_error_deg(diag['rotation'], np.eye(3))
    truth_angle = geodesic_error_deg(truth, np.eye(3))
    old_accept = (diag['inliers'] >= 20 and diag['inlier_ratio'] >= .6 and angle <= 10.)
    gated_accept = old_accept and diag['rotation_fit']['rmse_px'] <= RMSE_GATE_PX
    # Positive class: pure rotations (including identity); negative: translation-only.
    # Mixed motion is intentionally excluded from binary FPR/FNR and separately logged.
    positive = category == 'pure_rotation'
    negative = category == 'pure_translation'
    return diag, dict(matches=diag['matches'], inliers=diag['inliers'], inlier_ratio=diag['inlier_ratio'],
      true_rotation_deg=truth_angle, estimated_rotation_deg=angle,
      rotation_error_deg=geodesic_error_deg(diag['rotation'], truth),
      homography_rmse_px=diag['homography_fit']['rmse_px'],rotation_rmse_px=diag['rotation_fit']['rmse_px'],
      rmse_gap_px=diag['rmse_gap_px'], old_accept=int(old_accept),gated_accept=int(gated_accept),
      old_false_positive=int(negative and old_accept),gate_false_positive=int(negative and gated_accept),
      old_false_negative=int(positive and not old_accept),gate_false_negative=int(positive and not gated_accept))


# Distinct seeds and physically plausible camera-space translation in metres.
CASES = [
 ('identity','pure_rotation',(0,0,0),(0,0,0),0.,0.,(1.5,3.,6.)),
 ('yaw_4','pure_rotation',(4,0,0),(0,0,0),0.,0.,(1.5,3.,6.)),
 ('pitch_3','pure_rotation',(0,3,0),(0,0,0),.35,0.,(1.5,3.,6.)),
 ('roll_minus_4','pure_rotation',(0,0,-4),(0,0,0),.5,.2,(1.5,3.,6.)),
 ('combined','pure_rotation',(4,-3,2),(0,0,0),.5,.25,(1.5,3.,6.)),
 ('translate_x_2cm','pure_translation',(0,0,0),(.02,0,0),.25,0.,(1.5,3.,6.)),
 ('translate_x_8cm','pure_translation',(0,0,0),(.08,0,0),.25,0.,(1.5,3.,6.)),
 ('translate_y_8cm','pure_translation',(0,0,0),(0,.08,0),.25,0.,(1.5,3.,6.)),
 ('translate_z_10cm','pure_translation',(0,0,0),(0,0,.10),.25,0.,(1.5,3.,6.)),
 ('translate_xy_12cm','pure_translation',(0,0,0),(.09,.08,0),.25,.1,(1.5,3.,6.)),
 ('translate_x_far','pure_translation',(0,0,0),(.08,0,0),.25,0.,(12.,20.,30.)),
 ('translate_x_planar','pure_translation',(0,0,0),(.08,0,0),.25,0.,(4.,)),
 ('mixed_yaw_x','mixed',(4,0,0),(.08,0,0),.25,0.,(1.5,3.,6.)),
 ('mixed_yaw_z','mixed',(4,0,0),(0,0,.10),.25,.1,(1.5,3.,6.)),
 ('mixed_combined','mixed',(4,-3,2),(.05,.03,.07),.5,.2,(1.5,3.,6.)),
]

@pytest.mark.parametrize('case,category,angles,translation,noise,outliers,depths',CASES, ids=[x[0] for x in CASES])
def test_physical_motion(case,category,angles,translation,noise,outliers,depths, record):
    t=tracker()
    old,new,truth=generate_scene(t,angles,translation,depths,noise,outliers)
    diag,metrics=classify(t,old,new,truth,category)
    assert np.isfinite(metrics['rotation_rmse_px'])
    assert 0 <= metrics['inlier_ratio'] <= 1
    if category=='pure_rotation':
        assert metrics['rotation_error_deg'] < 1.0, (case,metrics)
    record.append(dict(case=case,category=category,depth_m=';'.join(map(str,depths)),
      translation_m=';'.join(map(str,translation)),noise_px=noise,outlier_fraction=outliers,seed=101,**metrics))

@pytest.fixture(scope='module')
def record():
    rows=[]
    yield rows
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader();writer.writerows(rows)
    print('\nT03 results written to',OUT)
    for cat in ('pure_rotation','pure_translation','mixed'):
        r=[x for x in rows if x['category']==cat]
        print(cat,len(r), 'accepted old:',sum(x['old_accept'] for x in r),'gate:',sum(x['gated_accept'] for x in r))


def test_depth_dependent_parallax():
    t=tracker()
    old,new,_=generate_scene(t,translation=(.10,0,0),depths=(1.5,6.),n_per_plane=100)
    # Approx horizontal image flow inversely proportional to depth
    near=np.mean(new[:100,0]-old[:100,0]);far=np.mean(new[100:,0]-old[100:,0])
    assert abs(near/far-4.) < .01


def test_rotation_is_depth_independent():
    t=tracker()
    old,new,_=generate_scene(t,angles=(4,-2,1),depths=(1.5,6.),n_per_plane=100)
    # A pure rotation follows the same rotation-induced homography at every depth.
    H=t.camera_matrix @ rotation_matrix(4,-2,1).astype('float64').T @ np.linalg.inv(t.camera_matrix)
    p=np.column_stack([old,np.ones(len(old))]) @ H.T
    xy=p[:,:2]/p[:,2,None]
    assert np.max(np.linalg.norm(xy-new,axis=1)) < 1e-3
