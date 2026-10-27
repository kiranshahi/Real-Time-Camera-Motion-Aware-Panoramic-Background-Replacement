"""Low-overhead per-frame wall-clock stage profiler for the video pipeline.

All durations use perf_counter_ns; absent stages are recorded as 0 ms.
For CUDA timing, synchronise around GPU work or use CUDA events when available.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Callable
import numpy as np

STAGES = ('capture', 'rectification', 'yolo_inference', 'mask_refinement',
          'foreground_exclusion', 'optical_flow', 'motion_estimation',
          'pose_update', 'panorama_render', 'compositing', 'visualisation')


@dataclass
class PipelineProfiler:
    """Accumulated frame measurements; total_processing excludes capture/UI by design."""
    clock: Callable[[], int] = perf_counter_ns

    def __post_init__(self):
        self._begin = None
        self._stage_start = None
        self._active = None
        self._times = {}
        self.rows = []

    def begin_frame(self):
        self._begin = self.clock()
        self._stage_start = None
        self._active = None
        self._times = {}

    def start(self, stage):
        if stage not in STAGES:
            raise ValueError(f'Unknown stage: {stage}')
        if self._begin is None or self._active is not None:
            raise RuntimeError('Frame missing or stage already running')
        self._stage_start = self.clock()
        self._active = stage

    def stop(self, stage):
        if self._active != stage:
            raise RuntimeError(f'Expected running stage {self._active}, got {stage}')
        elapsed = max(0, self.clock() - self._stage_start) / 1e6
        self._times[stage] = self._times.get(stage, 0.0) + elapsed
        self._active = None
        self._stage_start = None
        return elapsed

    def finish_frame(self, frame, *, mode='AUTO', status='UNKNOWN'):
        if self._begin is None or self._active is not None:
            raise RuntimeError('Cannot finish frame before begin or while a stage is active')
        elapsed = max(0, self.clock() - self._begin) / 1e6
        row = {'frame': int(frame), 'mode': str(mode), 'tracking_status': str(status),
               **{s + '_ms': float(self._times.get(s, 0.0)) for s in STAGES}}
        row['total_processing_ms'] = sum(row[s + '_ms'] for s in STAGES if s not in ('capture','visualisation'))
        row['total_loop_ms'] = elapsed
        row['processing_fps'] = 1000 / row['total_processing_ms'] if row['total_processing_ms'] > 0 else 0.0
        row['loop_fps'] = 1000 / elapsed if elapsed > 0 else 0.0
        self.rows.append(row)
        self._begin = None
        return row

    def summary(self, *, warmup=0):
        if warmup < 0:
            raise ValueError('warmup must be >= 0')
        rows = self.rows[warmup:]
        metrics = [s + '_ms' for s in STAGES] + ['total_processing_ms', 'total_loop_ms', 'processing_fps', 'loop_fps']
        result = []
        for metric in metrics:
            values = np.array([r[metric] for r in rows], dtype=float)
            result.append({'metric': metric, 'samples': len(values),
                           'mean': float(values.mean()) if len(values) else 0.0,
                           'median': float(np.median(values)) if len(values) else 0.0,
                           'p95': float(np.percentile(values, 95)) if len(values) else 0.0})
        return result

    def save(self, frame_csv, summary_csv, *, warmup=0):
        frame_csv, summary_csv = Path(frame_csv), Path(summary_csv)
        frame_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        keys = ['frame','mode','tracking_status'] + [s + '_ms' for s in STAGES] + ['total_processing_ms','total_loop_ms','processing_fps','loop_fps']
        with frame_csv.open('w',newline='',encoding='utf-8') as f:
            w = csv.DictWriter(f,fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
        with summary_csv.open('w',newline='',encoding='utf-8') as f:
            w = csv.DictWriter(f,fieldnames=['metric','samples','mean','median','p95'])
            w.writeheader()
            w.writerows(self.summary(warmup=warmup))
