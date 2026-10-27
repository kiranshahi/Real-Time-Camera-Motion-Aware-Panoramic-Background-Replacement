import numpy as np
import pytest
from panovideo.mask_evaluation import (METHODS, make_alpha_scene,synthetic_yolo_mask, apply_method,
    evaluate_alpha, boundary_f1, benchmark_method)

@pytest.mark.parametrize('kind',['ellipse','rectangle','hair'])
def test_reference_alpha_valid(kind):
    a=make_alpha_scene((120,200),kind)
    assert a.shape==(120,200) and a.dtype==np.float32
    assert a.min()==0 and a.max()<=1 and a.max()>=.65

@pytest.mark.parametrize('method',METHODS)
def test_method_shape_and_range(method):
    raw=synthetic_yolo_mask(make_alpha_scene((120,200)))
    a=apply_method(raw,method)
    assert a.shape==raw.shape and np.isfinite(a).all() and 0<=a.min()<=a.max()<=1

@pytest.mark.parametrize('method',METHODS)
def test_repeatability(method):
    raw=synthetic_yolo_mask(make_alpha_scene((120,200)),noise=.1)
    assert np.array_equal(apply_method(raw,method),apply_method(raw,method))

def test_perfect_scores():
    a=make_alpha_scene((120,200))
    m=evaluate_alpha(a,a)
    assert m['iou']==1 and m['boundary_f1']==1 and m['sad']==0 and m['mad']==0

def test_empty_mask_scores():
    z=np.zeros((64,64),np.float32);o=np.ones((64,64),np.float32)
    assert evaluate_alpha(z,z)['iou']==1
    assert boundary_f1(z,z)==1
    assert evaluate_alpha(o,z)['iou']==0
    assert evaluate_alpha(o,z)['mad']==1

def test_sad_mad_relation():
    a=make_alpha_scene((120,200));p=synthetic_yolo_mask(a,shift=(2,1))
    m=evaluate_alpha(p,a)
    assert m['sad']==pytest.approx(m['mad']*a.size)

def test_boundary_shift_worse_than_perfect():
    a=make_alpha_scene((120,200));p=synthetic_yolo_mask(a,shift=(8,0))
    assert boundary_f1(a,a)>boundary_f1(p,a)

def test_invalid_shape():
    with pytest.raises(ValueError):evaluate_alpha(np.zeros((4,4)),np.zeros((5,5)))

def test_invalid_values():
    with pytest.raises(ValueError):evaluate_alpha(np.full((3,3),np.nan),np.zeros((3,3)))

def test_unknown_method_and_scene():
    with pytest.raises(ValueError):apply_method(np.zeros((8,8)),'unknown')
    with pytest.raises(ValueError):make_alpha_scene((8,8),'unknown')

def test_latency_returns_finite_nonnegative():
    raw=synthetic_yolo_mask(make_alpha_scene((64,96)))
    timing=benchmark_method(raw,'binary',iterations=3,warmup=1)
    assert np.isfinite(list(timing.values())).all()
    assert timing['latency_median_ms']>=0
