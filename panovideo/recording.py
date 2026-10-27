"""Constant-frame-rate MP4 recording with explicitly documented timing semantics.

Offline: one output per decoded input frame with the source's constant FPS.
Live: wall-clock resampling (repeat last output / skip intermediate outputs).
OpenCV VideoWriter does not support individual frame presentation timestamps.
"""
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
import math
import cv2


@dataclass(frozen=True)
class RecordingStats:
    input_frames: int
    written_frames: int
    repeated_frames: int
    skipped_frames: int
    source_fps: float
    output_fps: float


def usable_fps(fps):
    return fps is not None and math.isfinite(float(fps)) and 0.1 <= float(fps) <= 240.0


class TimedRecorder:
    """A CFR recorder. Timestamp origin is the first supplied frame, not recording start.

    Offline videos preserve *nominal* CFR timing by writing every decoded frame.
    Camera mode resamples processed frames onto a CFR wall-clock grid.
    """
    def __init__(self, path, width, height, *, source_fps, live=False,
                 fallback_fps=20.0, writer_factory=None, max_repeats=300):
        if not usable_fps(source_fps) and not live:
            raise ValueError('Offline source has unknown/invalid FPS; specify a valid source FPS')
        self.live = bool(live)
        self.source_fps = float(source_fps) if usable_fps(source_fps) else float('nan')
        self.fps = float(source_fps) if usable_fps(source_fps) else float(fallback_fps)
        if not usable_fps(self.fps):
            raise ValueError('Invalid recording FPS')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = int(width), int(height)
        factory = writer_factory or (lambda name, fourcc, fps, dim: cv2.VideoWriter(name, fourcc, fps, dim))
        self.writer = factory(str(self.path), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))
        if not self.writer.isOpened():
            raise RuntimeError('Could not open MP4 writer')
        self.ledger_path = self.path.with_suffix('.timing.csv')
        self.ledger = self.ledger_path.open('w', newline='', encoding='utf-8')
        self.csv = csv.writer(self.ledger)
        self.csv.writerow(['output_index','output_pts_seconds','input_index','input_timestamp_seconds','repeat'])
        self.origin = None
        self.last_frame = None
        self.last_input_index = None
        self.last_input_time = None
        self.input_frames = self.written_frames = self.repeated_frames = self.skipped_frames = 0
        self.max_repeats = max_repeats
        self.closed = False

    def _emit(self, frame, input_index, input_time, repeated=False):
        self.writer.write(frame)
        self.csv.writerow([self.written_frames, f'{self.written_frames / self.fps:.9f}',
                           input_index, f'{input_time:.9f}', int(repeated)])
        self.written_frames += 1
        self.repeated_frames += int(repeated)

    def add(self, frame, *, input_index, timestamp_seconds):
        if self.closed:
            raise RuntimeError('Recorder is closed')
        if frame.shape[:2] != (self.height, self.width) or len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError('Frame dimensions do not match recorder')
        stamp = float(timestamp_seconds)
        if not math.isfinite(stamp):
            raise ValueError('Timestamp must be finite')
        if self.last_input_index is not None and input_index <= self.last_input_index:
            raise ValueError('Input indices must increase')
        if self.last_input_time is not None and stamp < self.last_input_time:
            raise ValueError('Timestamps must be nondecreasing')
        self.input_frames += 1
        if self.origin is None:
            self.origin = stamp
        if not self.live:
            if self.last_input_index is not None and input_index > self.last_input_index + 1:
                self.skipped_frames += input_index - self.last_input_index - 1
            self._emit(frame, input_index, stamp)
        else:
            # Nearest previous CFR slot, with a half-period tolerance for float noise.
            target = max(0, int(math.floor((stamp - self.origin) * self.fps + 1e-7)))
            if target < self.written_frames:
                self.skipped_frames += 1
            else:
                repeats = max(0, target - self.written_frames)
                if repeats > self.max_repeats:
                    raise RuntimeError('Excessive timestamp gap; restart recorder to avoid huge output')
                if self.last_frame is not None:
                    for _ in range(repeats):
                        self._emit(self.last_frame, self.last_input_index, self.last_input_time, repeated=True)
                self._emit(frame, input_index, stamp)
                self.last_frame = frame.copy()
        self.last_input_index = input_index
        self.last_input_time = stamp

    def close(self):
        if not self.closed:
            self.writer.release()
            self.ledger.close()
            self.closed = True
        return RecordingStats(self.input_frames, self.written_frames, self.repeated_frames,
                              self.skipped_frames, self.source_fps, self.fps)
