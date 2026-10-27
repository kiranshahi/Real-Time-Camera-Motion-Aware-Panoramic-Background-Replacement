"""T18 deterministic multi-depth pinhole scene/video generator and ground-truth poses.

World/camera convention: x right, y down, z forward; camera-to-world R=Ry Rx Rz.
X_camera = R.T @ (X_world - camera_position). Depths are in arbitrary metric metres.
Synthetic frames intentionally do not simulate realistic human appearance or motion blur.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import cv2
import numpy as np
from .geometry import rotation_matrix

SCENARIOS = ('static','yaw','pitch','roll','combined','translation','mixed','occlusion','tracking_loss')

@dataclass(frozen=True)
class DatasetConfig:
    width: int = 320
    height: int = 180
    fps: int = 20
    frames: int = 25
    seed: int = 2026
    hfov: float = 70.
    point_count: int = 2400
    def __post_init__(self):
        if self.width < 64 or self.height < 48 or self.frames < 4 or self.fps < 1 or self.point_count < 100 or not 10 < self.hfov < 160:
            raise ValueError('Invalid dataset settings')

def _hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for buf in iter(lambda:f.read(1048576), b''): h.update(buf)
    return h.hexdigest()

def motion_at(scenario: str, index: int, total: int):
    if scenario not in SCENARIOS: raise ValueError('Unknown scenario')
    s = index / max(1,total-1)
    yaw=pitch=roll=0.
    c=np.zeros(3,np.float64)
    if scenario == 'yaw': yaw=10*s
    elif scenario == 'pitch': pitch=8*s
    elif scenario == 'roll': roll=9*s
    elif scenario == 'combined': yaw,pitch,roll=8*s,-5*s,4*s
    elif scenario == 'translation': c[0]=0.20*s; c[1]=0.08*s
    elif scenario == 'mixed': yaw,pitch,roll=6*s,-3*s,2*s; c[0]=0.16*s
    elif scenario == 'occlusion': yaw=4*s
    elif scenario == 'tracking_loss': yaw=5*s
    return np.array([yaw,pitch,roll],np.float64), c

def _world_points(cfg):
    rng=np.random.default_rng(cfg.seed)
    z=rng.choice(np.array([1.5,3.,6.,12.]),cfg.point_count)
    x=rng.uniform(-1.65,1.65,cfg.point_count)*z
    y=rng.uniform(-0.95,0.95,cfg.point_count)*z
    points=np.column_stack([x,y,z])
    colors=rng.integers(35,250,(cfg.point_count,3),dtype=np.uint8)
    radius=rng.integers(1,3,cfg.point_count)
    return points,colors,radius

def render_scene(points,colors,radii,angles,camera_position,cfg,*,scenario='static',frame_index=0):
    R=rotation_matrix(*angles).astype(np.float64)
    Pc=(points-camera_position)@R
    focal=cfg.width/(2*np.tan(np.deg2rad(cfg.hfov)/2))
    valid=Pc[:,2]>0.05
    inds=np.where(valid)[0]
    coords=Pc[inds,:2]/Pc[inds,2,None]*focal+np.array([cfg.width/2,cfg.height/2])
    inside=(coords[:,0]>=-3)&(coords[:,0]<cfg.width+3)&(coords[:,1]>=-3)&(coords[:,1]<cfg.height+3)
    inds=inds[inside]; coords=coords[inside]
    frame=np.full((cfg.height,cfg.width,3),22,np.uint8)
    # Draw far-to-near for simple deterministic point-splat occlusion.
    for j in np.argsort(-Pc[inds,2]):
        k=int(inds[j]); u,v=np.rint(coords[j]).astype(int)
        cv2.circle(frame,(int(u),int(v)),int(radii[k]),tuple(int(x) for x in colors[k]),-1,cv2.LINE_8)
    mask=np.zeros((cfg.height,cfg.width),np.uint8)
    if scenario=='occlusion' and cfg.frames//4 <= frame_index < 3*cfg.frames//4:
        left=int(cfg.width*(.30+.1*frame_index/cfg.frames)); right=min(cfg.width-1,left+int(cfg.width*.32))
        cv2.rectangle(mask,(left,int(cfg.height*.1)),(right,int(cfg.height*.92)),255,-1)
        frame[mask>0]=(55,110,170)
    if scenario=='tracking_loss' and cfg.frames//3 <= frame_index < 2*cfg.frames//3:
        frame[:]=(22,22,22)  # no reliable features
    return frame,mask

def generate_dataset(output_dir, config=DatasetConfig(), scenarios=SCENARIOS):
    root=Path(output_dir); root.mkdir(parents=True,exist_ok=True)
    scenarios=tuple(scenarios)
    if not scenarios or any(x not in SCENARIOS for x in scenarios) or len(set(scenarios))!=len(scenarios):
        raise ValueError('Scenarios must be unique and recognised')
    pts,colors,radii=_world_points(config)
    entries=[]
    for name in scenarios:
        folder=root/name; folder.mkdir(exist_ok=True)
        video=folder/'video.avi'; gt=folder/'ground_truth.csv'
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),float(config.fps),(config.width,config.height))
        if not writer.isOpened(): raise RuntimeError('MJPG VideoWriter unavailable')
        try:
            with gt.open('w',newline='',encoding='utf-8') as f:
                fields=['frame_index','timestamp_s','yaw_deg','pitch_deg','roll_deg','tx_m','ty_m','tz_m','occluded_fraction','tracking_loss',*[f'R{r}{c}' for r in range(3) for c in range(3)]]
                cw=csv.DictWriter(f,fieldnames=fields); cw.writeheader()
                for i in range(config.frames):
                    angles,xyz=motion_at(name,i,config.frames)
                    frame,occ=render_scene(pts,colors,radii,angles,xyz,config,scenario=name,frame_index=i)
                    writer.write(frame)
                    R=rotation_matrix(*angles).astype(float)
                    row={'frame_index':i,'timestamp_s':i/config.fps,'yaw_deg':angles[0],'pitch_deg':angles[1],'roll_deg':angles[2],
                         'tx_m':xyz[0],'ty_m':xyz[1],'tz_m':xyz[2], 'occluded_fraction':np.count_nonzero(occ)/occ.size,
                         'tracking_loss':int(name=='tracking_loss' and config.frames//3<=i<2*config.frames//3)}
                    row.update({f'R{r}{c}':float(R[r,c]) for r in range(3) for c in range(3)})
                    cw.writerow(row)
        finally: writer.release()
        entries.append({'scenario':name,'video':str(video.relative_to(root)),'ground_truth':str(gt.relative_to(root)),
                        'video_sha256':_hash(video),'ground_truth_sha256':_hash(gt),'frames':config.frames})
    manifest={'schema_version':'t18.v1','generator':'deterministic_multidepth_pinhole_splats',
              'config':asdict(config),'depth_planes_m':[1.5,3.0,6.0,12.0],
              'camera_convention':'x-right y-down z-forward; R_camera_to_world=Ry Rx Rz; Xcam=R.T@(Xworld-C)',
              'frame_index_zero_based':True,'sequences':entries,
              'limitations':['Static world 3D point splats rather than photorealistic scenes',
                             'Foreground occlusion is a synthetic rectangle and is not annotated as a YOLO person',
                             'Tracking loss uses texture removal; no illumination/noise/rolling shutter model',
                             'MJPG video compression may alter pixel-level features']}
    (root/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return root/'dataset_manifest.json'

def load_manifest(path, *,verify_hashes=True):
    path=Path(path); data=json.loads(path.read_text())
    if data.get('schema_version')!='t18.v1': raise ValueError('Unknown dataset schema')
    for entry in data['sequences']:
        for key,hash_key in [('video','video_sha256'),('ground_truth','ground_truth_sha256')]:
            file=(path.parent/entry[key]).resolve()
            if not file.is_file(): raise FileNotFoundError(file)
            if verify_hashes and _hash(file)!=entry[hash_key]: raise ValueError(f'Hash mismatch: {file}')
    return data

def main(argv=None):
    p=argparse.ArgumentParser(description='Generate deterministic T18 camera motion evaluation dataset')
    p.add_argument('--output',default='experiments/t18_dataset');p.add_argument('--width',type=int,default=320)
    p.add_argument('--height',type=int,default=180);p.add_argument('--frames',type=int,default=25)
    p.add_argument('--fps',type=int,default=20);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--points',type=int,default=2400)
    p.add_argument('--scenarios',nargs='+',choices=SCENARIOS,default=list(SCENARIOS))
    a=p.parse_args(argv)
    cfg=DatasetConfig(a.width,a.height,a.fps,a.frames,a.seed,point_count=a.points)
    path=generate_dataset(a.output,cfg,a.scenarios);print(path)
if __name__=='__main__':main()
