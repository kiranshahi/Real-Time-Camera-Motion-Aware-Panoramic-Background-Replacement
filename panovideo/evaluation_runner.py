"""T18 dataset → T17 benchmark integration with descriptive pose-error outputs."""
from __future__ import annotations
import argparse,csv,json
from dataclasses import replace
from pathlib import Path
import numpy as np
from .benchmark import BenchmarkConfig,run_benchmark
from .evaluation_dataset import load_manifest
from .geometry import rotation_matrix

def rotation_error_deg(pred_angles,groundtruth_matrix):
    Rp=rotation_matrix(*pred_angles).astype(np.float64)
    d=Rp@groundtruth_matrix.T
    return float(np.rad2deg(np.arccos(np.clip((np.trace(d)-1)/2,-1,1))))

def run_dataset(manifest_path,panorama,*,output_dir='experiments/t18_benchmarks',scenarios=None,segmentation=False,warmup=2,max_frames=None,segmenter_factory=None):
    manifest_path=Path(manifest_path)
    data=load_manifest(manifest_path)
    names=set(scenarios) if scenarios is not None else None
    if names and not names.issubset({e['scenario'] for e in data['sequences']}):raise ValueError('Unknown requested scenario')
    results=[]
    for entry in data['sequences']:
        if names is not None and entry['scenario'] not in names:continue
        gt_path=manifest_path.parent/entry['ground_truth']
        config=BenchmarkConfig(video=str(manifest_path.parent/entry['video']),panorama=str(panorama),
                               output_dir=str(output_dir),segmentation=segmentation,warmup=warmup,
                               max_frames=max_frames,label='t18_'+entry['scenario'])
        result=run_benchmark(config,**({'segmenter_factory':segmenter_factory} if segmenter_factory is not None else {}))
        run_dir=Path(result['manifest']).parent
        with gt_path.open(newline='') as f:gt={int(r['frame_index']):r for r in csv.DictReader(f)}
        with (run_dir/'frames.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        errs=[]
        with (run_dir/'pose_errors.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['frame_index','rotation_error_deg','tracking_state','ground_truth_tracking_loss']);writer.writeheader()
            for row in rows:
                idx=int(row['source_frame_index']);g=gt[idx]
                R=np.array([[float(g[f'R{i}{j}']) for j in range(3)] for i in range(3)])
                error=rotation_error_deg([float(row[k]) for k in ['yaw_deg','pitch_deg','roll_deg']],R)
                writer.writerow({'frame_index':idx,'rotation_error_deg':error,'tracking_state':row['tracking_state'],
                                 'ground_truth_tracking_loss':g['tracking_loss']})
                if idx>=warmup:errs.append(error)
        stats={'mean_rotation_error_deg':float(np.mean(errs)) if errs else None,
               'median_rotation_error_deg':float(np.median(errs)) if errs else None,
               'p95_rotation_error_deg':float(np.percentile(errs,95)) if errs else None}
        results.append({'scenario':entry['scenario'],'run_id':result['run_id'],'frames':len(rows),'pose_metrics':stats,
                        'benchmark_summary':str(run_dir/'benchmark_summary.json'),'pose_errors_csv':str(run_dir/'pose_errors.csv')})
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    comparison=out/'t18_dataset_comparison.json';comparison.write_text(json.dumps({'dataset_manifest':str(manifest_path),'sequences':results},indent=2)+'\n')
    return results

def main(argv=None):
    p=argparse.ArgumentParser(description='Benchmark all T18 sequences with T17 runner')
    p.add_argument('--dataset-manifest',required=True);p.add_argument('--panorama',required=True)
    p.add_argument('--output-dir',default='experiments/t18_benchmarks');p.add_argument('--scenarios',nargs='+')
    p.add_argument('--segmentation',action='store_true');p.add_argument('--warmup',type=int,default=2);p.add_argument('--max-frames',type=int)
    a=p.parse_args(argv); result=run_dataset(a.dataset_manifest,a.panorama,output_dir=a.output_dir,scenarios=a.scenarios,
                                              segmentation=a.segmentation,warmup=a.warmup,max_frames=a.max_frames)
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
