"""T16: per-run structured experimental data. Nonfinite numeric diagnostics become null.

One run owns a manifest.json and frames.csv; original T14/T15 outputs remain available.
"""
from __future__ import annotations
import csv
import json
import math
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

from .profiling import STAGES

FRAME_FIELDS = (
    'run_id','frame_index','source_frame_index','input_timestamp_s','processing_timestamp_s',
    'capture_timestamp_s','mode','yaw_deg','pitch_deg','roll_deg',
    'tracking_state','tracking_reason','rotation_accepted','matches','inliers','inlier_ratio',
    'rotation_rmse_px','homography_rmse_px','rotation_innovation_deg','occupied_cells',
    'coverage_fraction','tracking_confidence','mask_reliability','mask_status',
    'excluded_pixels','recovery_state','recovery_failures','recording_active',
    'recording_input_frames','recording_written_frames','recording_repeated_frames',
    'recording_skipped_frames','recording_output_fps',
    *(stage + '_ms' for stage in STAGES),'total_processing_ms','total_loop_ms',
    'processing_fps','loop_fps'
)


def _serialisable(value):
    if is_dataclass(value):
        return _serialisable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialisable(v) for v in value]
    if hasattr(value, 'item'):
        return _serialisable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class ExperimentLogger:
    def __init__(self, root, *, configuration, source, run_id=None, created_at=None):
        self.run_id = run_id or f'{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:12]}'
        if not self.run_id or not all(c.isalnum() or c in '_-' for c in self.run_id):
            raise ValueError('Invalid run ID')
        self.directory = Path(root) / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.frames_path = self.directory / 'frames.csv'
        self.manifest_path = self.directory / 'manifest.json'
        self._stream = self.frames_path.open('w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._stream, fieldnames=FRAME_FIELDS)
        self._writer.writeheader()
        self.closed = False
        self.frame_count = 0
        self._last_index = -1
        self._last_times = {}
        self.manifest = {
            'schema_version': 't16.v1', 'run_id': self.run_id,
            'created_at_utc': created_at or datetime.now(timezone.utc).isoformat(),
            'configuration': _serialisable(configuration), 'source': _serialisable(source),
            'artifacts': {'frames_csv': 'frames.csv'}, 'recordings': [],
            'summary': {'frames_logged': 0, 'recording': {'sessions': 0, 'input_frames': 0,
                'written_frames': 0, 'repeated_frames': 0, 'skipped_frames': 0}}
        }
        self._write_manifest()

    def _write_manifest(self):
        self.manifest_path.write_text(json.dumps(_serialisable(self.manifest), indent=2, allow_nan=False) + '\n', encoding='utf-8')

    def log_frame(self, frame_index, *, frame_profile, **fields):
        if self.closed:
            raise RuntimeError('Logger closed')
        frame_index = int(frame_index)
        if frame_index <= self._last_index:
            raise ValueError('Frame indices must strictly increase')
        unknown = set(fields).difference(FRAME_FIELDS)
        if unknown:
            raise ValueError(f'Unrecognised frame columns: {sorted(unknown)}')
        row = {k: None for k in FRAME_FIELDS}
        row.update(_serialisable(frame_profile))
        row.update(_serialisable(fields))
        row['frame_index'] = frame_index
        row['run_id'] = self.run_id
        for time_key in ('input_timestamp_s', 'processing_timestamp_s', 'capture_timestamp_s'):
            value = row.get(time_key)
            if value is not None:
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f'Invalid {time_key}')
                if time_key in self._last_times and value < self._last_times[time_key]:
                    raise ValueError(f'Nonmonotonic {time_key}')
        self._writer.writerow({k: row[k] for k in FRAME_FIELDS})
        self.frame_count += 1
        self._last_index = frame_index
        for time_key in ('input_timestamp_s', 'processing_timestamp_s', 'capture_timestamp_s'):
            if row.get(time_key) is not None:
                self._last_times[time_key] = float(row[time_key])

    def add_recording(self, path, stats, *, timing_csv=None):
        if self.closed:
            raise RuntimeError('Logger closed')
        record = {'video_path': str(path), 'timing_csv': str(timing_csv) if timing_csv else None,
                  'statistics': _serialisable(stats)}
        self.manifest['recordings'].append(record)
        totals = self.manifest['summary']['recording']
        totals['sessions'] += 1
        for name in ('input_frames','written_frames','repeated_frames','skipped_frames'):
            totals[name] += int(getattr(stats, name))
        self._write_manifest()

    def close(self, *, profiling_frames=None, profiling_summary=None, motion_csv=None):
        if not self.closed:
            self._stream.close()
            self.closed = True
            self.manifest['summary']['frames_logged'] = self.frame_count
            for key, p in [('profile_frames_csv', profiling_frames), ('profile_summary_csv', profiling_summary), ('legacy_motion_csv', motion_csv)]:
                if p is not None:
                    self.manifest['artifacts'][key] = str(p)
            self._write_manifest()
        return self.manifest_path
