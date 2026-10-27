"""T20 controlled ablations on T18 synthetic sequences.

Foreground masks are geometry-derived *oracle proxy annotations*, NOT YOLO outputs.
Measurements are computed independently for each variant on the same decoded video.
"""
from __future__ import annotations
import argparse, csv, json, time, hashlib
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from .camera_tracking import BackgroundFeatureTracker, AcceptanceConfig, MotionState
from .recovery import TrackingRecovery, RecoveryState
from .temporal_mask import TemporalForegroundMask, TemporalMaskConfig
from .calibration import CameraCalibration
from .tracking_metrics import evaluate_trajectory
from .evaluation_dataset import load_manifest

VARIANTS = ('no_exclusion','direct_oracle','dilated_oracle','temporal_oracle','spatial_oracle','permissive_acceptance','strict_acceptance')

@dataclass(frozen=True)
class AblationConfig:
    variant: str
    warmup: int = 3
    threads: int = 1
    dropout_frames: tuple[int,...] = (9,10)
    def __post_init__(self):
        if self.variant not in VARIANTS or self.warmup < 0 or self.threads < 1:
            raise ValueError('Invalid ablation configuration')

def oracle_foreground_mask(scenario, index, width, height, frames):
    """Recreates only the T18 synthetic rectangular occlusion annotation."""
    mask = np.zeros((height,width),np.uint8)
    if scenario == 'occlusion' and frames//4 <= index < 3*frames//4:
        left=int(width*(.30+.1*index/frames)); right=min(width-1,left+int(width*.32))
        cv2.rectangle(mask,(left,int(height*.1)),(right,int(height*.92)),255,-1)
    return mask

def make_mask(variant, frame, oracle, frame_index, memory, dropout_frames):
    if variant == 'no_exclusion': return np.zeros_like(oracle), 'DISABLED'
    if variant in ('permissive_acceptance','strict_acceptance','spatial_oracle','direct_oracle'):
        return oracle.copy(), 'ORACLE'
    if variant == 'dilated_oracle':
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
        return cv2.dilate(oracle,kernel), 'ORACLE_DILATED'
    if variant == 'temporal_oracle':
        detected=frame_index not in dropout_frames and bool(np.any(oracle))
        result=memory.update(frame,oracle if detected else np.zeros_like(oracle),detected=detected)
        return result,memory.last_status
    raise ValueError(variant)

def make_policy(variant):
    if variant == 'permissive_acceptance':
        return AcceptanceConfig(min_inliers=12,min_inlier_ratio=0.25,
            rotation_rmse_valid_px=1e6,rotation_rmse_invalid_px=1e6,
            min_occupied_cells=1,min_coverage_fraction=0.,max_innovation_deg=180.)
    if variant == 'strict_acceptance':
        return AcceptanceConfig(rotation_rmse_valid_px=0.75,
            rotation_rmse_invalid_px=2.,min_inlier_ratio=.75,
            min_occupied_cells=7,min_coverage_fraction=.25,max_innovation_deg=3.)
    return AcceptanceConfig()

def run_sequence(video_path,gt_path,scenario,config,*, frames_expected=None):
    cv2.setNumThreads(config.threads)
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():raise RuntimeError('Could not open video '+str(video_path))
    with open(gt_path,newline='') as file: gt=list(csv.DictReader(file))
    if frames_expected is not None and len(gt)!=frames_expected:raise ValueError('Ground truth length mismatch')
    tracker=BackgroundFeatureTracker()
    memory=TemporalForegroundMask(TemporalMaskConfig(max_missing_frames=2))
    recovery=TrackingRecovery(tracker)
    cfg=make_policy(config.variant)
    rows=[];timings=[];i=0; excluded=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            if i>=len(gt):raise ValueError('Extra video frames')
            height,width=frame.shape[:2]
            if i==0:tracker.set_calibration(CameraCalibration.from_hfov(width,height,70.))
            oracle=oracle_foreground_mask(scenario,i,width,height,len(gt))
            start=time.perf_counter_ns()
            mask,status=make_mask(config.variant,frame,oracle,i,memory,config.dropout_frames)
            tracker.feature_grid=(4,4) if config.variant=='spatial_oracle' else (1,1)
            old,new=tracker.track_features(frame,foreground_mask=mask)
            decision=None
            if not (recovery.state==RecoveryState.WARMUP and recovery.warmup_left>0):
                decision=tracker.validate_rotation(old,new,frame_size=(width,height),config=cfg,mask_status=status)
            recovery.update(frame,mask,decision)
            tracker.smooth_rotation()
            yaw,pitch,roll=tracker.get_camera_orientation()
            latency=(time.perf_counter_ns()-start)/1e6
            rows.append({'source_frame_index':i,'yaw_deg':yaw,'pitch_deg':pitch,'roll_deg':roll,
                         'rotation_accepted':bool(decision and decision.accepted),
                         'tracking_state':decision.state.value if decision else 'WARMUP',
                         'recovery_state':recovery.state.value,'mask_status':status,
                         'matches':len(old) if old is not None else 0,
                         'inliers':decision.inliers if decision else 0,
                         'processing_ms':latency,
                         'foreground_excluded_pixels':int(np.count_nonzero(mask))})
            if i>=config.warmup:timings.append(latency)
            excluded.append(int(np.count_nonzero(mask)))
            i+=1
    finally:cap.release()
    if i!=len(gt):raise ValueError('Video and ground truth frame count mismatch')
    metrics,details=evaluate_trajectory(gt,rows,scenario,warmup=config.warmup)
    metrics.update(variant=config.variant,
                   processing_mean_ms=float(np.mean(timings)) if timings else None,
                   processing_median_ms=float(np.median(timings)) if timings else None,
                   processing_p95_ms=float(np.percentile(timings,95)) if timings else None,
                   excluded_mean_px=float(np.mean(excluded)) if excluded else 0)
    return metrics,rows

def run_study(manifest_path,output_dir,*,variants=VARIANTS,scenarios=None,warmup=3,threads=1):
    path=Path(manifest_path).resolve();root=path.parent
    manifest=load_manifest(path)
    entries=[x for x in manifest['sequences'] if scenarios is None or x['scenario'] in scenarios]
    variants=list(variants)
    if not entries or not variants or any(v not in VARIANTS for v in variants):raise ValueError('Invalid study selection')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    records=[];frames=[]
    for item in entries:
        for variant in variants:
            result,rows=run_sequence(root/item['video'],root/item['ground_truth'],item['scenario'],
                AblationConfig(variant,warmup,threads),frames_expected=item['frames'])
            records.append(result)
            frames.extend([dict(variant=variant,scenario=item['scenario'],**r) for r in rows])
    with (out/'ablation_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    with (out/'ablation_frames.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(frames[0]));w.writeheader();w.writerows(frames)
    (out/'ablation_metadata.json').write_text(json.dumps({
        'schema_version':'t20.v1','dataset_manifest':str(path),
        'dataset_manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'variants':variants,'scenarios':[x['scenario'] for x in entries],
        'warmup':warmup,'opencv_threads':threads,
        'mask_source':'T18 synthetic geometric rectangle oracle; not YOLO output',
        'timing_scope':'mask+feature tracking+motion validation+recovery+smoothing; excludes YOLO and rendering',
        'limitations':'Synthetic occlusion only; timed runs have OS and CPU variability; translation FP counts any accepted update even near zero magnitude.'},indent=2)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(11,5))
        names=[x['scenario'] for x in entries]
        for v in variants:
            data=[next(z for z in records if z['variant']==v and z['scenario']==s)['are_mean_deg'] for s in names]
            ax.plot(names,data,marker='o',label=v)
        ax.set_ylabel('Mean absolute rotation error (degrees)');ax.tick_params(axis='x',rotation=30);ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(out/'ablation_are.png',dpi=140);plt.close(fig)
    except ImportError:pass
    return records

def main(argv=None):
    p=argparse.ArgumentParser(description='T20 controlled synthetic scene ablation')
    p.add_argument('--dataset-manifest',required=True)
    p.add_argument('--output-dir',default='experiments/t20_results')
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    p.add_argument('--scenarios',nargs='+')
    args=p.parse_args(argv)
    result=run_study(args.dataset_manifest,args.output_dir,variants=args.variants,scenarios=args.scenarios,warmup=args.warmup,threads=args.threads)
    print(json.dumps({'runs':len(result),'output_dir':args.output_dir},indent=2))
if __name__=='__main__':main()
