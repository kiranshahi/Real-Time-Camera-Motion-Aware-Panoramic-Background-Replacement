import csv
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from panovideo.evaluation_dataset import DatasetConfig,SCENARIOS,generate_dataset,load_manifest,motion_at,render_scene,_world_points
from panovideo.evaluation_runner import rotation_error_deg,run_dataset
from panovideo.geometry import rotation_matrix

@pytest.mark.parametrize('scenario',SCENARIOS)
def test_scenario_motion_groundtruth(scenario):
    angles,c=motion_at(scenario,9,10)
    assert angles.shape==(3,) and c.shape==(3,)
    if scenario=='static': assert np.array_equal(angles,np.zeros(3)) and np.all(c==0)
    if scenario=='translation': assert np.all(angles==0) and np.linalg.norm(c)>0
    if scenario=='mixed': assert np.linalg.norm(angles)>0 and np.linalg.norm(c)>0
    if scenario in ('yaw','pitch','roll','combined'): assert np.linalg.norm(angles)>0 and np.all(c==0)

@pytest.mark.parametrize('kwargs',[{'width':0},{'fps':0},{'frames':2},{'point_count':20},{'hfov':180}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):DatasetConfig(**kwargs)

def test_reproducible_byte_identical_groundtruth_and_video(tmp_path):
    cfg=DatasetConfig(96,64,12,8,22,point_count=240)
    a=generate_dataset(tmp_path/'a',cfg,('static','yaw','translation','occlusion','tracking_loss'))
    b=generate_dataset(tmp_path/'b',cfg,('static','yaw','translation','occlusion','tracking_loss'))
    aa=load_manifest(a);bb=load_manifest(b)
    assert [(x['video_sha256'],x['ground_truth_sha256']) for x in aa['sequences']]==[(x['video_sha256'],x['ground_truth_sha256']) for x in bb['sequences']]
    for entry in aa['sequences']:
        with (a.parent/entry['ground_truth']).open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==cfg.frames
        assert [int(r['frame_index']) for r in rows]==list(range(cfg.frames))
        assert np.allclose([float(r['timestamp_s']) for r in rows],np.arange(cfg.frames)/cfg.fps)
        cap=cv2.VideoCapture(str(a.parent/entry['video']))
        assert cap.isOpened()
        n=0
        while cap.read()[0]:n+=1
        cap.release()
        assert n==cfg.frames
    # tampering detected
    (a.parent/aa['sequences'][0]['ground_truth']).write_text('tampered')
    with pytest.raises(ValueError,match='Hash mismatch'):load_manifest(a)

def test_occlusion_and_tracking_loss_are_distinct():
    cfg=DatasetConfig(96,64,12,12,33,point_count=300)
    pts,colors,rad=_world_points(cfg)
    a,pos=motion_at('occlusion',5,12)
    _,mask=render_scene(pts,colors,rad,a,pos,cfg,scenario='occlusion',frame_index=5)
    assert np.count_nonzero(mask)>0
    img,_=render_scene(pts,colors,rad,a,pos,cfg,scenario='tracking_loss',frame_index=5)
    assert np.unique(img.reshape(-1,3),axis=0).shape[0]==1

def test_rotation_error_exact_and_difference():
    angles=[6.,-3.,2.];R=rotation_matrix(*angles)
    assert rotation_error_deg(angles,R)<0.03
    assert rotation_error_deg([0.,0.,0.],rotation_matrix(10,0,0))==pytest.approx(10,abs=.01)

def test_dataset_benchmark_integration(tmp_path,monkeypatch):
    cfg=DatasetConfig(96,64,12,8,11,point_count=270)
    manifest=generate_dataset(tmp_path/'dataset',cfg,('static','yaw'))
    pan=np.random.default_rng(17).integers(0,255,(128,256,3),dtype=np.uint8)
    pano=tmp_path/'pano.png';cv2.imwrite(str(pano),pan)
    monkeypatch.setattr(cv2,'imshow',lambda *a,**k:pytest.fail('UI must not be called'))
    results=run_dataset(manifest,pano,output_dir=tmp_path/'results',warmup=2,max_frames=6)
    assert len(results)==2
    assert {x['scenario'] for x in results}=={'static','yaw'}
    for x in results:
        assert x['frames']==6 and x['pose_metrics']['mean_rotation_error_deg'] is not None
        assert len(list(csv.DictReader(open(x['pose_errors_csv']))))==6
    assert (tmp_path/'results'/'t18_dataset_comparison.json').exists()

def test_bad_scenario_rejected(tmp_path):
    with pytest.raises(ValueError):generate_dataset(tmp_path,DatasetConfig(96,64,frames=4,point_count=100),('other',))
