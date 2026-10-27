"""T08 confidence diagnostic across unchanged T03 controlled scenes."""
from pathlib import Path
import csv,sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'tests'))
from test_physical_motion import CASES,generate_scene
from test_camera_rotation import tracker
rows=[]
for name,category,angles,translation,noise,outliers,depths in CASES:
    t=tracker()
    old,new,_=generate_scene(t,angles,translation,depths,noise,outliers)
    result=t.validate_rotation(old,new,frame_size=(640,360),mask_status='DETECTED')
    rows.append(dict(case=name,category=category,state=result.state.value,
                     accepted=int(result.accepted), confidence=result.confidence,
                     inliers=result.inliers,inlier_ratio=result.inlier_ratio,
                     rotation_rmse_px=result.rotation_rmse_px,coverage_fraction=result.coverage_fraction,
                     occupied_cells=result.occupied_cells,innovation_deg=result.innovation_deg,
                     mask_status='DETECTED',reason=result.reason))
out=Path(__file__).with_name('t08_confidence_synthetic.csv')
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
for category in ('pure_rotation','pure_translation','mixed'):
    subset=[r for r in rows if r['category']==category]
    print(category, 'n=',len(subset),'accepted=',sum(r['accepted'] for r in subset),
          'mean_confidence=',round(sum(r['confidence'] for r in subset)/len(subset),3))
print('Saved',out)
