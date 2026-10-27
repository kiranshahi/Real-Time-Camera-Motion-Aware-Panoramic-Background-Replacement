"""T17 headless deterministic-configuration offline video benchmark.

Deterministic *configuration and input* do not imply deterministic latency or YOLO kernels.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .calibration import CameraCalibration
from .camera_tracking import AcceptanceConfig, BackgroundFeatureTracker
from .experiment_logging import ExperimentLogger
from .geometry import PanoramaRenderer
from .pose import ManualPoseProvider
from .profiling import PipelineProfiler
from .recovery import RecoveryState, TrackingRecovery
from .segmenter import PersonSegmenter, composite, create_tracking_mask, refine_mask, refine_mask_v2
from .temporal_mask import TemporalForegroundMask, TemporalMaskConfig


@dataclass(frozen=True)
class BenchmarkConfig:
    video: str
    panorama: str
    output_dir: str = 'experiments/benchmarks'
    segmentation: bool = False
    model: str = 'yolo26n-seg.pt'
    device: str = 'cpu'
    imgsz: int = 416
    refine_version: int = 2
    fov: float = 70.0
    calibration: str | None = None
    warmup: int = 10
    max_frames: int | None = None
    seed: int = 2026
    label: str = 'baseline'
    threads: int = 1

    def __post_init__(self):
        if self.warmup < 0 or self.max_frames is not None and self.max_frames < 1:
            raise ValueError('Invalid warmup/max_frames')
        if self.refine_version not in (1, 2) or self.imgsz < 32 or self.threads < 1:
            raise ValueError('Invalid refinement, image size or threads')
        if not 0 < self.fov < 180 or not self.label or not all(c.isalnum() or c in '_-' for c in self.label):
            raise ValueError('Invalid FOV or label')


def _sha256(path):
    sha = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def _summary(profiler, warmup):
    return {item['metric']: item for item in profiler.summary(warmup=warmup)}


def run_benchmark(config: BenchmarkConfig, *, segmenter_factory=PersonSegmenter):
    """Run one fixed input video through the whole CPU pipeline without UI or MP4 encoding."""
    np.random.seed(config.seed)
    cv2.setNumThreads(config.threads)
    input_video = Path(config.video).resolve()
    pano_path = Path(config.panorama).resolve()
    if not input_video.is_file() or not pano_path.is_file():
        raise FileNotFoundError('Video/panorama not found')
    panorama = cv2.imread(str(pano_path), cv2.IMREAD_COLOR)
    if panorama is None:
        raise ValueError('Cannot decode panorama')
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError('Cannot open benchmark video')
    logger = None
    try:
        ok, first = cap.read()
        if not ok:
            raise ValueError('No decodable frames')
        height, width = first.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if np.isfinite(fps) and fps > 0 else None
        calibration = (CameraCalibration.from_json(config.calibration, width, height)
                       if config.calibration else CameraCalibration.from_hfov(width, height, config.fov))
        renderer = PanoramaRenderer(panorama, width, height, config.fov, calibration=calibration)
        tracker = BackgroundFeatureTracker()
        tracker.set_calibration(calibration)
        recovery = TrackingRecovery(tracker)
        mask_memory = TemporalForegroundMask(TemporalMaskConfig(max_missing_frames=2))
        acceptance = AcceptanceConfig()
        pose_provider = ManualPoseProvider()
        segmenter = (segmenter_factory(config.model, config.device, config.imgsz)
                     if config.segmentation else None)
        configuration = asdict(config)
        source = {'kind': 'offline_benchmark', 'path': str(input_video), 'fps': fps,
                  'width': width, 'height': height, 'sha256': _sha256(input_video),
                  'panorama_sha256': _sha256(pano_path)}
        logger = ExperimentLogger(Path(config.output_dir), configuration=configuration, source=source)
        profiler = PipelineProfiler()
        frame = first
        index = 0
        while True:
            if config.max_frames is not None and index >= config.max_frames:
                break
            if index:
                ok, frame = cap.read()
                if not ok:
                    break
            index += 1
            profiler.begin_frame()
            profiler.start('rectification')
            rectified = calibration.rectify(frame)
            profiler.stop('rectification')
            mask = np.zeros((height, width), np.uint8)
            alpha = np.zeros((height, width), np.float32)
            if segmenter is not None:
                profiler.start('yolo_inference')
                raw = segmenter.predict(rectified)
                profiler.stop('yolo_inference')
                profiler.start('mask_refinement')
                alpha = refine_mask(raw) if config.refine_version == 1 else refine_mask_v2(raw)
                profiler.stop('mask_refinement')
                profiler.start('foreground_exclusion')
                mask = mask_memory.update(rectified, create_tracking_mask(raw), detected=bool(np.any(raw >= 0.5)))
                profiler.stop('foreground_exclusion')
            profiler.start('optical_flow')
            old_pts, new_pts = tracker.track_features(rectified, foreground_mask=mask)
            profiler.stop('optical_flow')
            profiler.start('motion_estimation')
            decision = None
            if not (recovery.state == RecoveryState.WARMUP and recovery.warmup_left > 0):
                decision = tracker.validate_rotation(old_pts, new_pts, frame_size=(width, height),
                                                     config=acceptance,
                                                     mask_status=mask_memory.last_status if segmenter else 'DISABLED')
            recovery.update(rectified, mask, decision)
            profiler.stop('motion_estimation')
            profiler.start('pose_update')
            tracker.smooth_rotation()
            yaw, pitch, roll = tracker.get_camera_orientation()
            pose = pose_provider.update(rectified, 0.0, controls=(yaw, pitch, roll))
            profiler.stop('pose_update')
            profiler.start('panorama_render')
            background = renderer.render(pose.yaw, pose.pitch, pose.roll)
            profiler.stop('panorama_render')
            profiler.start('compositing')
            _output = composite(rectified, background, alpha)
            profiler.stop('compositing')
            row = profiler.finish_frame(index, mode='AUTO', status=decision.state.value if decision else 'WARMUP')
            logger.log_frame(index, frame_profile=row, source_frame_index=index-1,
                             input_timestamp_s=(index-1)/fps if fps else None,
                             processing_timestamp_s=time.perf_counter(),
                             yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                             tracking_state=decision.state.value if decision else 'WARMUP',
                             tracking_reason=decision.reason if decision else 'recovery_warmup',
                             rotation_accepted=decision.accepted if decision else False,
                             matches=len(old_pts) if old_pts is not None else 0,
                             inliers=decision.inliers if decision else 0,
                             rotation_rmse_px=decision.rotation_rmse_px if decision else None,
                             rotation_innovation_deg=decision.innovation_deg if decision else None,
                             occupied_cells=decision.occupied_cells if decision else 0,
                             coverage_fraction=decision.coverage_fraction if decision else None,
                             tracking_confidence=decision.confidence if decision else 0.,
                             mask_reliability=decision.mask_reliability if decision else None,
                             mask_status=mask_memory.last_status if segmenter else 'DISABLED',
                             excluded_pixels=int(np.count_nonzero(mask)),
                             recovery_state=recovery.state.value, recovery_failures=recovery.failures)
        profiler.save(logger.directory / 'profile_frames.csv', logger.directory / 'profile_summary.csv', warmup=config.warmup)
        logger.close(profiling_frames='profile_frames.csv', profiling_summary='profile_summary.csv')
        summary = _summary(profiler, config.warmup)
        output = {'run_id': logger.run_id, 'label': config.label, 'frames': index,
                  'warmup': config.warmup, 'measured_frames': max(0,index-config.warmup),
                  'configuration': configuration, 'source': source,
                  'environment': {'python': sys.version.split()[0], 'opencv': cv2.__version__,
                                  'numpy': np.__version__, 'platform': platform.platform(),
                                  'cv2_threads': cv2.getNumThreads()},
                  'metrics': summary, 'manifest': str(logger.manifest_path)}
        (logger.directory / 'benchmark_summary.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
        return output
    finally:
        cap.release()
        if logger is not None and not logger.closed:
            logger.close()


def compare_summaries(items, output_csv):
    """Compare separate runs without pretending they share identical runtime conditions."""
    rows = []
    for item in items:
        for metric, stats in item['metrics'].items():
            rows.append({'run_id':item['run_id'], 'label':item['label'], 'frames':item['frames'],
                         'measured_frames':item['measured_frames'], 'metric':metric,
                         'mean':stats['mean'], 'median':stats['median'], 'p95':stats['p95']})
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['run_id','label','frames','measured_frames','metric','mean','median','p95'])
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description='T17 headless offline video benchmark')
    parser.add_argument('--video', required=True)
    parser.add_argument('--panorama', required=True)
    parser.add_argument('--output-dir', default='experiments/benchmarks')
    parser.add_argument('--segmentation', action='store_true')
    parser.add_argument('--model', default='yolo26n-seg.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--imgsz', type=int, default=416)
    parser.add_argument('--refine-version', type=int, choices=[1,2], default=2)
    parser.add_argument('--fov', type=float, default=70)
    parser.add_argument('--calibration')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--label', default='baseline')
    args = parser.parse_args(argv)
    result = run_benchmark(BenchmarkConfig(**vars(args)))
    print(json.dumps({'run_id': result['run_id'], 'frames':result['frames'],
                      'measured_frames':result['measured_frames'],
                      'total_processing_ms':result['metrics']['total_processing_ms'],
                      'processing_fps':result['metrics']['processing_fps'],
                      'manifest':result['manifest']}, indent=2))


if __name__ == '__main__':
    main()
