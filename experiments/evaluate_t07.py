"""Deterministic feature-contamination experiment; not a real-video performance benchmark."""
import csv
from pathlib import Path
import cv2
import numpy as np
from panovideo.camera_tracking import BackgroundFeatureTracker
from panovideo.temporal_mask import TemporalForegroundMask, TemporalMaskConfig


def run():
    rng=np.random.default_rng(20261026)
    h,w=180,320
    gray=rng.integers(0,256,(h,w),dtype=np.uint8)
    image=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    true_mask=np.zeros((h,w),np.uint8)
    true_mask[35:150,90:220]=255
    tracker=BackgroundFeatureTracker()
    t=TemporalForegroundMask(TemporalMaskConfig(max_missing_frames=2))
    initial=t.update(image,true_mask,detected=True)
    rows=[]
    for i in range(1,5):
        # Inject explicit dropout: no YOLO detection; scene remains stationary.
        propagated=t.update(image,np.zeros_like(true_mask),detected=False)
        for name, exclusion in [('no_temporal',np.zeros_like(true_mask)),('temporal',propagated)]:
            pts=tracker.detect_features(image,foreground_mask=exclusion)
            coords=np.rint(pts.reshape(-1,2)).astype(int)
            contamination=int(np.count_nonzero(true_mask[coords[:,1],coords[:,0]]))
            rows.append(dict(frame=i,method=name,features=len(coords),foreground_features=contamination,
                             contamination_fraction=contamination/max(len(coords),1),mask_status=t.last_status))
    dest=Path(__file__).with_name('t07_tracking_contamination.csv')
    with dest.open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(dest)
    for row in rows:print(row)

if __name__=='__main__':run()
