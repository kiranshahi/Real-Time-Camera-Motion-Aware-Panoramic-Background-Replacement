"""T19 trajectory metrics for the T18 deterministic motion benchmark.

Definitions: all errors are geodesic SO(3) angles in degrees. Jitter is RMS
increment error after removing the mean increment-error vector, not raw motion.
Denominators for classification exclude the first and warmup frames.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from .geometry import rotation_matrix
from .evaluation_dataset import load_manifest
from .evaluation_runner import run_dataset

FIELDS=['scenario','frames','evaluated_transitions','are_mean_deg','are_median_deg','are_p95_deg','rre_mean_deg','rre_p95_deg','endpoint_drift_deg','drift_deg_per_s','jitter_rms_deg','translation_false_positive_rate','translation_fp_count','translation_evaluated','valid_motion_rejection_rate','valid_motion_rejected','valid_motion_evaluated','tracking_loss_frames','recovery_latency_frames','recovered','accepted_frames']

def so3_angle(a,b):
    d=np.asarray(a,dtype=np.float64)@np.asarray(b,dtype=np.float64).T
    return float(np.degrees(np.arccos(np.clip((np.trace(d)-1)/2,-1,1))))

def _matrix(g):return np.array([[float(g[f'R{i}{j}']) for j in range(3)] for i in range(3)],dtype=float)
def _yes(v):return str(v).lower() in ('true','1','yes')
def _finite(x):return float(x) if x is not None and np.isfinite(x) else None

def evaluate_trajectory(gt_rows,pred_rows,scenario,*,warmup=0,translation_threshold_m=1e-5,rotation_threshold_deg=0.05):
    """Join by zero-based source frame index. Missing frames cannot silently shift poses."""
    gt={int(r['frame_index']):r for r in gt_rows}
    pred={int(r['source_frame_index']):r for r in pred_rows}
    if not gt or not pred:raise ValueError('Empty trajectory')
    if len(gt)!=len(gt_rows) or len(pred)!=len(pred_rows):raise ValueError('Duplicate frame indices')
    ids=sorted(pred)
    if any(k not in gt for k in ids):raise ValueError('Missing ground truth frame')
    if any(k not in pred for k in range(min(ids),max(ids)+1)):raise ValueError('Missing estimate frame')
    gR={i:_matrix(gt[i]) for i in ids}
    pR={i:rotation_matrix(*[float(pred[i][key]) for key in ('yaw_deg','pitch_deg','roll_deg')]).astype(float) for i in ids}
    are={i:so3_angle(pR[i],gR[i]) for i in ids}
    eval_ids=[i for i in ids if i>=warmup]
    transitions=[(a,b) for a,b in zip(ids[:-1],ids[1:]) if b>=warmup and b==a+1]
    rel=[]; innovations=[]; trans_count=0;trans_fp=0;valid_count=0;valid_rej=0
    for a,b in transitions:
        observed=pR[a].T@pR[b]
        expected=gR[a].T@gR[b]
        rel.append(so3_angle(observed,expected))
        # small-angle log of relative error, expressed in the initial frame
        import cv2
        delta=observed@expected.T
        vec,_=cv2.Rodrigues(delta)
        innovations.append(np.degrees(vec.reshape(3)))
        travel=np.linalg.norm(np.array([float(gt[b][k])-float(gt[a][k]) for k in ('tx_m','ty_m','tz_m')]))
        gt_angle=so3_angle(gR[a],gR[b])
        accepted=_yes(pred[b].get('rotation_accepted',False))
        if travel>translation_threshold_m and gt_angle<rotation_threshold_deg:
            trans_count+=1;trans_fp+=int(accepted)
        if gt_angle>=rotation_threshold_deg and travel<=translation_threshold_m and not _yes(gt[b].get('tracking_loss',False)):
            valid_count+=1;valid_rej+=int(not accepted)
    times=np.array([float(gt[i]['timestamp_s']) for i in eval_ids],float)
    errors=np.array([are[i] for i in eval_ids],float)
    rel=np.array(rel,dtype=float)
    innovations=np.array(innovations,dtype=float).reshape(-1,3)
    # Jitter: remove systematic mean angular-increment bias, report remaining variation
    jitter=float(np.sqrt(np.mean(np.sum((innovations-innovations.mean(axis=0))**2,axis=1)))) if len(innovations)>1 else None
    loss=[i for i in ids if _yes(gt[i].get('tracking_loss',False))]
    recovery=None; recovered=None
    if loss:
        final_loss=max(loss)
        post=[i for i in ids if i>final_loss]
        first=next((i for i in post if _yes(pred[i].get('rotation_accepted',False))),None)
        recovered=first is not None
        recovery=first-final_loss if first is not None else None
    elapsed=float(times[-1]-times[0]) if len(times)>1 else 0.
    drift=so3_angle(pR[eval_ids[-1]],gR[eval_ids[-1]]) if eval_ids else None
    summary=dict(scenario=scenario,frames=len(ids),evaluated_transitions=len(transitions),
        are_mean_deg=_finite(np.mean(errors)) if len(errors) else None,
        are_median_deg=_finite(np.median(errors)) if len(errors) else None,
        are_p95_deg=_finite(np.percentile(errors,95)) if len(errors) else None,
        rre_mean_deg=_finite(np.mean(rel)) if len(rel) else None,
        rre_p95_deg=_finite(np.percentile(rel,95)) if len(rel) else None,
        endpoint_drift_deg=drift,drift_deg_per_s=drift/elapsed if drift is not None and elapsed>0 else None,
        jitter_rms_deg=jitter,
        translation_false_positive_rate=trans_fp/trans_count if trans_count else None,
        translation_fp_count=trans_fp,translation_evaluated=trans_count,
        valid_motion_rejection_rate=valid_rej/valid_count if valid_count else None,
        valid_motion_rejected=valid_rej,valid_motion_evaluated=valid_count,
        tracking_loss_frames=len(loss),recovery_latency_frames=recovery,recovered=recovered,
        accepted_frames=sum(_yes(pred[i].get('rotation_accepted',False)) for i in eval_ids))
    details=[dict(scenario=scenario,frame_index=i,timestamp_s=float(gt[i]['timestamp_s']),are_deg=are[i],
                  tracking_state=pred[i].get('tracking_state',''),recovery_state=pred[i].get('recovery_state',''),
                  rotation_accepted=_yes(pred[i].get('rotation_accepted',False)),
                  tracking_loss=_yes(gt[i].get('tracking_loss',False))) for i in ids]
    return summary,details

def _read_csv(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))

def evaluate_saved_runs(manifest_path,t18_results_path,output_dir,*,warmup=3):
    manifest_path=Path(manifest_path);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    source=load_manifest(manifest_path)
    results=json.loads(Path(t18_results_path).read_text())['sequences']
    lookup={e['scenario']:e for e in source['sequences']}
    summaries=[];details=[]
    for r in results:
        scenario=r['scenario']; entry=lookup[scenario]
        ground=_read_csv(manifest_path.parent/entry['ground_truth'])
        pred=_read_csv(Path(r['benchmark_summary']).parent/'frames.csv')
        summary,detail=evaluate_trajectory(ground,pred,scenario,warmup=warmup)
        summaries.append(summary);details.extend(detail)
    with (out/'t19_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,FIELDS);w.writeheader();w.writerows(summaries)
    with (out/'t19_frames.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['scenario','frame_index','timestamp_s','are_deg','tracking_state','recovery_state','rotation_accepted','tracking_loss']);w.writeheader();w.writerows(details)
    (out/'t19_summary.json').write_text(json.dumps({'metric_definitions':{
        'ARE':'SO(3) geodesic distance between estimated and ground-truth absolute orientation',
        'RRE':'SO(3) geodesic distance between estimated and GT consecutive relative rotations',
        'drift':'final absolute orientation error and final error divided by elapsed evaluated time; not bias-corrected drift slope',
        'jitter':'RMS demeaned rotation-vector error of consecutive orientation increments',
        'translation_FP':'accepted update during GT nonzero translation with negligible GT rotation',
        'valid_rejection':'non-accepted update during GT nonzero rotation, negligible translation, outside tracking loss',
        'recovery':'frames after final annotated loss frame until next accepted motion (not evidence of global pose recovery)'},
        'warmup':warmup,'dataset_manifest':str(manifest_path),'runs':summaries},indent=2))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,4.8));ax.bar([x['scenario'] for x in summaries],[x['are_mean_deg'] or 0 for x in summaries]);ax.set_ylabel('Mean absolute rotation error (deg)');ax.tick_params(axis='x',rotation=35);fig.tight_layout();fig.savefig(out/'t19_are_by_scenario.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,4.5))
        for name in ('yaw','combined','tracking_loss'):
            subset=[x for x in details if x['scenario']==name]
            if subset:ax.plot([x['timestamp_s'] for x in subset],[x['are_deg'] for x in subset],label=name)
        ax.set(xlabel='Time (s)',ylabel='Absolute rotation error (deg)');ax.legend();fig.tight_layout();fig.savefig(out/'t19_error_trajectories.png',dpi=160);plt.close(fig)
    except ImportError:pass
    return summaries

def main(argv=None):
    p=argparse.ArgumentParser(description='T19 quantitative pose evaluation')
    p.add_argument('--dataset-manifest',required=True);p.add_argument('--t18-results');p.add_argument('--panorama');p.add_argument('--output-dir',default='experiments/t19_results');p.add_argument('--warmup',type=int,default=3)
    a=p.parse_args(argv)
    if a.warmup<0: p.error('warmup must be nonnegative')
    results=a.t18_results
    if not results:
        if not a.panorama:p.error('Supply --t18-results or --panorama')
        run_dataset(a.dataset_manifest,a.panorama,output_dir=str(Path(a.output_dir)/'runs'),warmup=a.warmup)
        results=str(Path(a.output_dir)/'runs'/'t18_dataset_comparison.json')
    summary=evaluate_saved_runs(a.dataset_manifest,results,a.output_dir,warmup=a.warmup)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
