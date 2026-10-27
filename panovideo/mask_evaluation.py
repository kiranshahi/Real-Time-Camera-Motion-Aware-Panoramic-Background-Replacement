"""Deterministic mask-refinement evaluation. No YOLO weights required.

IoU and boundary F1 evaluate binarised foreground (>=0.5), while SAD/MAD
compare the continuous estimated alpha to ground-truth alpha.
"""
from dataclasses import dataclass
from time import perf_counter_ns
import cv2
import numpy as np
from .segmenter import refine_mask, refine_mask_v2

METHODS = ('binary', 'dilated_blurred', 'softened_yolo')


def make_alpha_scene(shape=(360, 640), kind='ellipse'):
    """Antialiased ground-truth alpha; reproducible with no random inputs."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    if kind == 'ellipse':
        signed = 1.0 - np.sqrt(((xx - w * .5) / (w * .19)) ** 2 + ((yy - h * .51) / (h * .37)) ** 2)
        alpha = np.clip(signed * min(h, w) * .075 + .5, 0, 1)
    elif kind == 'rectangle':
        # Signed distance to an axis aligned rectangle with antialiased edge.
        dx = np.abs(xx-w*.5)-w*.2
        dy = np.abs(yy-h*.5)-h*.29
        outside = np.hypot(np.maximum(dx, 0), np.maximum(dy, 0))
        inside = np.minimum(np.maximum(dx,dy),0)
        alpha = np.clip(.5-outside-inside,0,1)
    elif kind == 'hair':
        base = make_alpha_scene(shape, 'ellipse')
        strands = np.zeros(shape, np.uint8)
        cx=int(w*.5)
        for i in range(-14,15,2):
            x=cx+i*max(1,w//120)
            cv2.line(strands, (x,int(h*.20)), (x+i,int(h*.04)), 255, 1, cv2.LINE_AA)
        alpha = np.maximum(base, strands.astype(np.float32)/255*.65)
    else:
        raise ValueError('Unknown synthetic scene')
    return alpha.astype(np.float32)


def synthetic_yolo_mask(gt_alpha, threshold=0.5, shift=(0,0), noise=0.0, seed=17):
    """Synthetic proxy for *binary* YOLO segmentation, not actual YOLO output."""
    gt = np.asarray(gt_alpha,dtype=np.float32)
    binary=(gt >= threshold).astype(np.float32)
    h,w=binary.shape
    M=np.float32([[1,0,shift[0]],[0,1,shift[1]]])
    raw=cv2.warpAffine(binary,M,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    if noise:
        rng=np.random.default_rng(seed)
        raw=np.clip(raw+rng.normal(0,noise,raw.shape),0,1)
    return raw.astype(np.float32)


def apply_method(raw, method):
    if method == 'binary':
        return (np.asarray(raw)>=0.5).astype(np.float32)
    if method == 'dilated_blurred':
        return refine_mask(raw)
    if method == 'softened_yolo':
        return refine_mask_v2(raw)
    raise ValueError(f'Unknown refinement method: {method}')


def _boundary(binary):
    u=binary.astype(np.uint8)
    return cv2.morphologyEx(u,cv2.MORPH_GRADIENT, np.ones((3,3),np.uint8)).astype(bool)


def boundary_f1(pred, gt, tolerance_px=2):
    """Symmetric tolerant boundary F1 at >=0.5, with empty-mask handling."""
    p=_boundary(pred>=0.5)
    g=_boundary(gt>=0.5)
    if not np.any(p) and not np.any(g): return 1.0
    if not np.any(p) or not np.any(g): return 0.0
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*tolerance_px+1,)*2)
    gp=cv2.dilate(g.astype(np.uint8), k).astype(bool)
    pp=cv2.dilate(p.astype(np.uint8), k).astype(bool)
    precision=np.count_nonzero(p&gp)/np.count_nonzero(p)
    recall=np.count_nonzero(g&pp)/np.count_nonzero(g)
    return float(2*precision*recall/(precision+recall)) if precision+recall else 0.


def evaluate_alpha(pred, gt, tolerance_px=2):
    pred=np.asarray(pred,dtype=np.float32);gt=np.asarray(gt,dtype=np.float32)
    if pred.shape != gt.shape or pred.ndim !=2: raise ValueError('Alpha maps must have same 2D shape')
    if not (np.all(np.isfinite(pred)) and np.all(np.isfinite(gt))): raise ValueError('Alpha maps must be finite')
    if np.any(pred<0) or np.any(pred>1) or np.any(gt<0) or np.any(gt>1):raise ValueError('Alpha outside [0,1]')
    pb=pred>=.5; gb=gt>=.5
    union=np.count_nonzero(pb|gb)
    iou=np.count_nonzero(pb&gb)/union if union else 1.
    diff=np.abs(pred-gt)
    return dict(iou=float(iou), boundary_f1=boundary_f1(pred,gt,tolerance_px),sad=float(np.sum(diff,dtype=np.float64)), mad=float(np.mean(diff,dtype=np.float64)))


def benchmark_method(raw, method, iterations=30, warmup=5):
    if iterations <=0 or warmup<0: raise ValueError('Invalid iteration count')
    for _ in range(warmup): apply_method(raw,method)
    durations=[]
    for _ in range(iterations):
        t=perf_counter_ns();apply_method(raw,method);durations.append((perf_counter_ns()-t)/1e6)
    return {'latency_median_ms':float(np.median(durations)), 'latency_p95_ms':float(np.percentile(durations,95))}
