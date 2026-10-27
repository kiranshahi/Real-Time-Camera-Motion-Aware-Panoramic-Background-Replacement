import csv
import numpy as np
import pytest
import cv2
from pathlib import Path
from panovideo.ablation import AblationConfig, VARIANTS, oracle_foreground_mask, make_mask, make_policy, run_sequence, run_study
from panovideo.temporal_mask import TemporalForegroundMask, TemporalMaskConfig

@pytest.mark.parametrize('variant',VARIANTS)
def test_config(variant):
    assert AblationConfig(variant).variant==variant

def test_unknown_variant_rejected():
    with pytest.raises(ValueError):AblationConfig('unknown')

def test_oracle_only_occlusion_and_interval():
    assert np.count_nonzero(oracle_foreground_mask('yaw',12,320,180,24))==0
    assert np.count_nonzero(oracle_foreground_mask('occlusion',3,320,180,24))==0
    assert np.count_nonzero(oracle_foreground_mask('occlusion',12,320,180,24))>0

def test_dilation_expands_mask():
    frame=np.zeros((180,320,3),np.uint8)
    oracle=oracle_foreground_mask('occlusion',12,320,180,24)
    m,_=make_mask('dilated_oracle',frame,oracle,12,None,())
    assert np.count_nonzero(m)>np.count_nonzero(oracle)

def test_disabled_has_empty_mask():
    frame=np.zeros((180,320,3),np.uint8)
    oracle=np.full((180,320),255,np.uint8)
    mask,status=make_mask('no_exclusion',frame,oracle,0,None,())
    assert not mask.any() and status=='DISABLED'

def test_temporal_expiry():
    mem=TemporalForegroundMask(TemporalMaskConfig(max_missing_frames=2))
    frame=np.zeros((80,120,3),np.uint8)
    oracle=np.zeros((80,120),np.uint8);oracle[20:45,20:60]=255
    make_mask('temporal_oracle',frame,oracle,6,mem,(7,8,9))
    for i in (7,8,9):
        mask,status=make_mask('temporal_oracle',frame,oracle,i,mem,(7,8,9))
    assert not mask.any()

def test_policy_strict_vs_permissive():
    assert make_policy('strict_acceptance').rotation_rmse_valid_px < make_policy('direct_oracle').rotation_rmse_valid_px
    assert make_policy('permissive_acceptance').rotation_rmse_valid_px > make_policy('direct_oracle').rotation_rmse_valid_px

def test_small_study_outputs(tmp_path):
    from panovideo.evaluation_dataset import generate_dataset, DatasetConfig
    dataset=tmp_path/'dataset'
    manifest=generate_dataset(dataset,DatasetConfig(width=160,height=90,frames=8,point_count=400),scenarios=('static',))
    result=run_study(dataset/'dataset_manifest.json',tmp_path/'output',variants=['no_exclusion','direct_oracle'],warmup=2)
    assert len(result)==2
    assert (tmp_path/'output'/'ablation_summary.csv').is_file()
    rows=list(csv.DictReader(open(tmp_path/'output'/'ablation_frames.csv')))
    assert len(rows)==16
