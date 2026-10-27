import numpy as np
import pytest
from panovideo.tracking_metrics import evaluate_trajectory,so3_angle
from panovideo.geometry import rotation_matrix

def synth(yaws,tx=None,loss=None,accepted=None):
    tx=tx or [0]*len(yaws);loss=loss or [False]*len(yaws);accepted=accepted or [False]+[True]*(len(yaws)-1)
    g=[];p=[]
    for i,y in enumerate(yaws):
        R=rotation_matrix(y,0,0)
        a=dict(frame_index=i,timestamp_s=i*.05,tx_m=tx[i],ty_m=0,tz_m=0,tracking_loss=int(loss[i]))
        a.update({f"R{r}{c}":R[r,c] for r in range(3) for c in range(3)})
        g.append(a);p.append(dict(source_frame_index=i,yaw_deg=y,pitch_deg=0,roll_deg=0,rotation_accepted=accepted[i],tracking_state='VALID' if accepted[i] else 'INVALID'))
    return g,p

@pytest.mark.parametrize('angle',[0,1,5,-8,90])
def test_geodesic(angle):
    assert so3_angle(rotation_matrix(angle,0,0),rotation_matrix(angle,0,0))<0.03

def test_perfect_trajectory():
    g,p=synth([0,1,2,3,4]);s,d=evaluate_trajectory(g,p,'yaw',warmup=1)
    assert s['are_mean_deg']<0.03 and s['rre_mean_deg']<0.03
    assert s['valid_motion_rejection_rate']==0
    assert len(d)==5

def test_false_positive():
    g,p=synth([0]*4,tx=[0,.01,.02,.03]);s,_=evaluate_trajectory(g,p,'translation')
    assert s['translation_evaluated']==3 and s['translation_false_positive_rate']==1

def test_rejections():
    g,p=synth([0,2,4],accepted=[False,False,True]);s,_=evaluate_trajectory(g,p,'yaw')
    assert s['valid_motion_rejection_rate']==.5

def test_recovery():
    g,p=synth([0,1,2,3,4],loss=[False,True,True,False,False],accepted=[False,False,False,False,True]);s,_=evaluate_trajectory(g,p,'loss')
    assert s['tracking_loss_frames']==2 and s['recovered'] is True and s['recovery_latency_frames']==2

def test_no_recovery():
    g,p=synth([0,1,2],loss=[False,True,False],accepted=[False]*3);s,_=evaluate_trajectory(g,p,'loss')
    assert s['recovered'] is False and s['recovery_latency_frames'] is None

def test_missing_and_duplicate():
    g,p=synth([0,1,2]);p.pop(1)
    with pytest.raises(ValueError):evaluate_trajectory(g,p,'invalid')
    g,p=synth([0,1]);p.append(p[-1])
    with pytest.raises(ValueError):evaluate_trajectory(g,p,'invalid')

def test_warmup():
    g,p=synth([0,1,2]);s,_=evaluate_trajectory(g,p,'warmup',warmup=2)
    assert s['evaluated_transitions']==1

def test_invalid_ground_truth():
    g,p=synth([0,1]);g=g[:1]
    with pytest.raises(ValueError):evaluate_trajectory(g,p,'bad')
