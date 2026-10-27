import numpy as np
import cv2
import pytest
from panovideo.temporal_mask import TemporalForegroundMask, TemporalMaskConfig
from panovideo.camera_tracking import BackgroundFeatureTracker


def frame(seed=0, shape=(96, 128)):
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 256, size=shape, dtype=np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def mask(shape=(96,128)):
    m = np.zeros(shape, np.uint8)
    m[25:68, 30:73] = 255
    return m


def test_detected_keeps_exact_mask():
    t=TemporalForegroundMask(); m=mask()
    assert np.array_equal(t.update(frame(),m,detected=True),m)
    assert t.last_status=='DETECTED'


def test_dropouts_expire_strictly(monkeypatch):
    t=TemporalForegroundMask(TemporalMaskConfig(max_missing_frames=2))
    m=mask(); f=frame()
    t.update(f,m,detected=True)
    monkeypatch.setattr(t,'_warp_consistent',lambda *a: m.copy())
    assert np.any(t.update(f,np.zeros_like(m),detected=False))
    assert np.any(t.update(f,np.zeros_like(m),detected=False))
    assert not np.any(t.update(f,np.zeros_like(m),detected=False))
    assert t.last_status=='EXPIRED'


def test_propagated_mask_moves(monkeypatch):
    t=TemporalForegroundMask(); m=mask(); f=frame(); t.update(f,m,detected=True)
    shifted=np.roll(m,5,axis=1)
    monkeypatch.setattr(t,'_warp_consistent',lambda *a: shifted)
    got=t.update(f,np.zeros_like(m),detected=False)
    assert np.array_equal(got,shifted) and t.last_status=='WARPED'


def test_fallback_only_one_frame(monkeypatch):
    t=TemporalForegroundMask(); m=mask(); f=frame(); t.update(f,m,detected=True)
    monkeypatch.setattr(t,'_warp_consistent',lambda *a: None)
    assert np.count_nonzero(t.update(f,np.zeros_like(m),detected=False))>np.count_nonzero(m)
    assert t.last_status=='FALLBACK'
    assert not np.any(t.update(f,np.zeros_like(m),detected=False))
    assert t.last_status=='UNRELIABLE'


def test_detection_restores_after_expiry(monkeypatch):
    t=TemporalForegroundMask(TemporalMaskConfig(max_missing_frames=1));m=mask();f=frame()
    t.update(f,m,detected=True)
    monkeypatch.setattr(t,'_warp_consistent',lambda *a: m)
    t.update(f,np.zeros_like(m),detected=False)
    t.update(f,np.zeros_like(m),detected=False)
    assert np.array_equal(t.update(f,m,detected=True),m) and t.missing==0


def test_explicit_empty_detection_clears():
    t=TemporalForegroundMask();m=mask();f=frame();t.update(f,m,detected=True)
    assert not np.any(t.update(f,np.zeros_like(m),detected=True))


def test_shape_change_does_not_propagate():
    t=TemporalForegroundMask();t.update(frame(),mask(),detected=True)
    small=frame(shape=(50,60)); result=t.update(small,np.zeros((50,60),np.uint8),detected=False)
    assert result.shape==(50,60) and not np.any(result)


def test_invalid_mask_dimension():
    with pytest.raises(ValueError): TemporalForegroundMask().update(frame(),np.zeros((4,5),np.uint8),detected=True)


def test_foreground_exclusion_reduces_contamination():
    # Deterministic contamination proxy: location mask excludes center-dense corners.
    image=frame(42)
    m=mask()
    tracker=BackgroundFeatureTracker()
    pts_nomask=tracker.detect_features(image)
    pts_mask=tracker.detect_features(image,foreground_mask=m)
    def contaminated(pts):
        coords=np.rint(pts.reshape(-1,2)).astype(int)
        return int((m[coords[:,1],coords[:,0]]!=0).sum())
    assert contaminated(pts_nomask)>0
    assert contaminated(pts_mask)==0


def test_warp_forward_backward_translation():
    t=TemporalForegroundMask();f=frame(7)
    m=mask();t.update(f,m,detected=True)
    matrix=np.float32([[1,0,3],[0,1,2]])
    moved=cv2.warpAffine(f,matrix,(128,96),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    t.update(moved,np.zeros_like(m),detected=False)
    assert t.last_status in ('WARPED','FALLBACK')
    assert np.any(t.last_mask)


def test_binary_output_with_nonbinary_input():
    t=TemporalForegroundMask();m=mask();m[m>0]=2
    got=t.update(frame(),m,detected=True)
    assert set(np.unique(got)).issubset({0,255})
