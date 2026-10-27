import numpy as np
import cv2
import pytest
from panovideo.camera_tracking import BackgroundFeatureTracker, MotionDecision, MotionState
from panovideo.recovery import TrackingRecovery, RecoveryState, RecoveryConfig
from panovideo.geometry import rotation_matrix


def frame(seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)


def decision(accept):
    return MotionDecision(MotionState.VALID if accept else MotionState.INVALID,
                          'accepted' if accept else 'bad', accept)


def angle(a,b):
    r = a @ b.T
    return np.rad2deg(np.arccos(np.clip((np.trace(r)-1)/2,-1,1)))


def make():
    t = BackgroundFeatureTracker()
    return t, TrackingRecovery(t, RecoveryConfig(reseed_after=3,warmup_frames=1))


def test_repeated_failures_reseed_preserves_orientation():
    t,r = make(); image=frame()
    t.set_orientation_anchor((14,-3,2)); original=t.rotation_matrix.copy()
    r.update(image, None, None)  # initial warmup
    for n in range(3):
        state = r.update(image, None, decision(False))
    assert state == RecoveryState.WARMUP
    assert r.failures == 0
    assert angle(original, t.rotation_matrix) < 0.03
    assert angle(original, t.smoothed_rotation) < 0.03
    assert t.previous_gray is not None


def test_initial_failure_holds():
    t,r=make(); r.update(frame(),None,decision(False))
    assert r.failures==0  # warmup
    r.update(frame(),None,decision(False))
    assert r.state==RecoveryState.HOLD and r.failures==1


def test_good_estimate_exits_hold():
    t,r=make(); im=frame();r.update(im,None,None)
    r.update(im,None,decision(False));r.update(im,None,decision(True))
    assert r.state==RecoveryState.TRACKING and r.failures==0


def test_rapid_failures_reseed_repeatedly():
    t,r=make();im=frame();r.update(im,None,None)
    for _ in range(2):r.update(im,None,decision(False))
    assert r.failures==2
    r.update(im,None,decision(False))
    assert r.state==RecoveryState.WARMUP and r.failures==0
    r.update(im,None,None)
    for _ in range(3):r.update(im,None,decision(False))
    assert r.state==RecoveryState.WARMUP


@pytest.mark.parametrize('angles',[(0,0,0),(12,-4,3),(-20,8,-7)])
def test_auto_manual_pose_continuous(angles):
    t,r=make();im=frame();t.set_orientation_anchor(angles)
    old=t.smoothed_rotation.copy()
    values=r.set_mode('MANUAL',angles,im)
    assert r.mode=='MANUAL'
    assert values==angles and angle(old,t.smoothed_rotation)<0.03
    assert r.update(im,None,decision(False))==RecoveryState.WARMUP
    values2=r.set_mode('AUTO',angles,im)
    assert values2==angles and r.mode=='AUTO'
    assert angle(old,t.rotation_matrix)<0.03
    assert t.last_valid_increment is None


def test_manual_modified_pose_is_auto_anchor():
    t,r=make();im=frame()
    r.set_mode('MANUAL',(5,2,1),im)
    r.set_mode('AUTO',(17,-6,4),im)
    expected=rotation_matrix(17,-6,4).T
    assert angle(expected,t.rotation_matrix)<0.03


def test_reseed_does_not_cross_frame_gap():
    t,r=make();im=frame();t.previous_gray=np.zeros((10,10),np.uint8)
    t.previous_points=np.zeros((2,1,2),np.float32)
    r._reseed(im)
    assert t.previous_gray.shape==im.shape[:2]
    assert t.previous_points is None or len(t.previous_points)<=t.feature_params['maxCorners']


def test_reseed_respects_foreground_exclusion():
    t,r=make();im=frame(); mask=np.zeros(im.shape[:2],np.uint8);mask[:,:80]=255
    r._reseed(im,mask)
    if t.previous_points is not None:
        assert (t.previous_points.reshape(-1,2)[:,0]>=80).all()


def test_reseed_clears_increment_not_absolute_pose():
    t,r=make();im=frame();t.set_orientation_anchor((9,2,-3))
    t.last_valid_increment=np.eye(3)
    before=t.rotation_matrix.copy();r._reseed(im)
    assert t.last_valid_increment is None and np.allclose(t.rotation_matrix,before)


def test_invalid_mode_rejected_without_mutation():
    t,r=make();im=frame();old=t.rotation_matrix.copy()
    with pytest.raises(ValueError): r.set_mode('WRONG',(0,0,0),im)
    assert r.mode=='AUTO' and np.allclose(t.rotation_matrix,old)


def test_nonfinite_pose_rejected():
    t,r=make()
    with pytest.raises(ValueError): r.set_mode('MANUAL',(np.nan,0,0),frame())
    assert r.mode=='AUTO'


def test_same_mode_is_no_op():
    t,r=make(); t.set_orientation_anchor((3,2,1)); old=t.rotation_matrix.copy()
    assert r.set_mode('AUTO',(3,2,1),frame())==(3.,2.,1.)
    assert np.allclose(t.rotation_matrix,old)


def test_config_validation():
    t=BackgroundFeatureTracker()
    with pytest.raises(ValueError):TrackingRecovery(t,RecoveryConfig(reseed_after=0))
