"""Compare T03 original, 2 px gate and T04 policy on synthetic motions."""
import csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from test_physical_motion import CASES, generate_scene, classify
from test_camera_rotation import tracker

rows=[]
for case, category, angles, translation, noise, outliers, depths in CASES:
    t=tracker()
    old,new,truth=generate_scene(t,angles,translation,depths,noise,outliers)
    _, previous=classify(t,old,new,truth,category)
    decision=t.validate_rotation(old,new,frame_size=(640,360))
    rows.append({'case':case,'category':category,'original_accept':previous['old_accept'],
         'rmse_gate_accept':previous['gated_accept'],'t04_state':decision.state.value,
         't04_reason':decision.reason,'t04_accept':int(decision.accepted),
         'rotation_rmse_px':decision.rotation_rmse_px,'coverage_fraction':decision.coverage_fraction,
         'occupied_cells':decision.occupied_cells})
path=Path(__file__).with_name('t04_acceptance_comparison.csv')
with path.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
for category in ('pure_rotation','pure_translation','mixed'):
    subset=[r for r in rows if r['category']==category]
    print(category,len(subset),'original',sum(x['original_accept'] for x in subset),'rmse_gate',sum(x['rmse_gate_accept'] for x in subset),'T04',sum(x['t04_accept'] for x in subset))
for r in rows: print(r['case'],r['t04_state'],r['t04_reason'])
print(path)
