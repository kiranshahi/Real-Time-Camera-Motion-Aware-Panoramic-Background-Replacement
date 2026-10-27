"""Run from project root: python experiments/evaluate_t12.py"""
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panovideo.mask_evaluation import METHODS,make_alpha_scene,synthetic_yolo_mask,apply_method,evaluate_alpha,benchmark_method


def run(output=None, iterations=30):
    output=Path(output or 'experiments/t12_mask_refinement.csv')
    output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    scenarios=[('ellipse_clean','ellipse',(0,0),0),('ellipse_shifted','ellipse',(3,2),0),('rectangle_noisy','rectangle',(1,0),.08),('hair_noisy','hair',(2,1),.05)]
    for name,kind,shift,noise in scenarios:
        gt=make_alpha_scene((360,640),kind)
        raw=synthetic_yolo_mask(gt,shift=shift,noise=noise)
        for method in METHODS:
            pred=apply_method(raw,method)
            rows.append({'scene':name,'method':method,**evaluate_alpha(pred,gt),**benchmark_method(raw,method,iterations=iterations)})
    with output.open('w',newline='',encoding='utf-8') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {output}')
    for r in rows:print(f"{r['scene']:19s} {r['method']:18s} IoU={r['iou']:.4f} F1={r['boundary_f1']:.4f} MAD={r['mad']:.4f} med={r['latency_median_ms']:.3f}ms")
    return rows
if __name__=='__main__':run()
