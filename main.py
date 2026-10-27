"""Windows webcam demo: segmentation + spherical rendering + local recording."""
import argparse
from datetime import datetime
from pathlib import Path
import time
import csv
from dataclasses import asdict
from panovideo.experiment_logging import ExperimentLogger

import cv2
import numpy as np

from panovideo.geometry import PanoramaRenderer
from panovideo.calibration import CameraCalibration
from panovideo.pose import ManualPoseProvider
from panovideo.segmenter import PersonSegmenter, composite, create_tracking_mask, refine_mask, refine_mask_v2
from panovideo.camera_tracking import BackgroundFeatureTracker, AcceptanceConfig
from panovideo.temporal_mask import TemporalForegroundMask, TemporalMaskConfig
from panovideo.recovery import TrackingRecovery, RecoveryState
from panovideo.profiling import PipelineProfiler
from panovideo.recording import TimedRecorder, usable_fps


def parse_args():
    parser = argparse.ArgumentParser(description='Rotation-aware 360 panorama virtual background prototype')
    parser.add_argument('--panorama', default='assets/demo_panorama.jpg', help='Equirectangular 2:1 panorama path')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360)
    parser.add_argument('--calibration', default=None, help='JSON file containing calibrated width, height, fx, fy, cx, cy, optional distortion')
    parser.add_argument('--fov', type=float, default=70, help='Horizontal camera FOV in degrees')
    parser.add_argument('--model', default='yolo26n-seg.pt', help='Ultralytics segmentation weights')
    parser.add_argument('--device', default='auto', help='auto, cpu, 0 for first CUDA GPU')
    parser.add_argument('--imgsz', type=int, default=416)
    parser.add_argument('--no-segmentation', action='store_true', help='Debug panorama only')
    parser.add_argument('--video', default=None, help='Use existing MP4 instead of camera')
    parser.add_argument('--refine-version', type=int, default=1)
    parser.add_argument('--record-fps', type=float, default=None, help='Override capture FPS when the camera or video metadata is unreliable')
    parser.add_argument('--profile-warmup', type=int, default=10, help='Exclude first N frames from summary statistics')
    return parser.parse_args()


def track(name, default, upper):
    cv2.createTrackbar(name, 'Controls', default, upper, lambda _: None)


def start_recording(width, height, *, source_fps, live):
    filename = Path('outputs') / f'panoramic_{datetime.now():%Y%m%d_%H%M%S_%f}.mp4'
    recorder = TimedRecorder(filename, width, height, source_fps=source_fps, live=live)
    return recorder, filename


def main():
    args = parse_args()
    panorama = cv2.imread(args.panorama, cv2.IMREAD_COLOR)
    if panorama is None:
        raise FileNotFoundError(f'Panorama not found: {args.panorama}')
    if not 1.8 <= panorama.shape[1] / panorama.shape[0] <= 2.2:
        print('Warning: expected a 2:1 equirectangular panorama.')
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise RuntimeError('Camera/video could not open. Try --camera 1 or check Windows camera permissions.')
    source_fps = args.record_fps if args.record_fps is not None else cap.get(cv2.CAP_PROP_FPS)
    if not args.video and not usable_fps(source_fps):
        source_fps = 20.0
        print('Warning: camera FPS unavailable; using 20 FPS CFR output. Provide --record-fps for a calibrated rate.')
    elif args.video and not usable_fps(source_fps):
        raise ValueError('Input video FPS unavailable. Specify --record-fps.')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError('Video stream opened but returned no frames.')
    height, width = first.shape[:2]
    calibration = (CameraCalibration.from_json(args.calibration, width, height)
                   if args.calibration else CameraCalibration.from_hfov(width, height, args.fov))
    frame = calibration.rectify(first)
    renderer = PanoramaRenderer(panorama, width, height, args.fov, calibration=calibration)
    pose_provider = ManualPoseProvider()
    feature_tracker = BackgroundFeatureTracker()
    feature_tracker.set_calibration(calibration)

    # enable automatic camera tracking
    auto_tracking = True
    
    acceptance_config = AcceptanceConfig()
    recovery = TrackingRecovery(feature_tracker)

    segmenter = None if args.no_segmentation else PersonSegmenter(args.model, args.device, args.imgsz)
    temporal_mask = TemporalForegroundMask(TemporalMaskConfig(max_missing_frames=2))
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    track('yaw + 180', 180, 360)
    track('pitch + 90', 90, 180)
    track('roll + 180', 180, 360)
    cv2.namedWindow('MotionMatte Prototype', cv2.WINDOW_NORMAL)

    # initialize csv to log experiment data
    Path("experiments").mkdir(exist_ok=True)
    log_path = Path("experiments") / (f"motion_{datetime.now():%Y%m%d_%H%M%S_%f}.csv")
    motion_log = open(log_path,"w", newline="", encoding="utf-8")
    csv_writer = csv.writer(motion_log)
    csv_writer.writerow(["frame", "timestamp", "matches", "inliers", "inlier_ratio", "rotation_accepted", "yaw", "pitch", "roll", "tracking_state", "tracking_reason", "rotation_rmse_px", "occupied_cells", "coverage_fraction", "innovation_deg", "foreground_mask_status", "excluded_pixels", "tracking_confidence_score", "mask_reliability", "recovery_state", "recovery_failures"])


    experiment = ExperimentLogger(
        Path('experiments') / 'runs',
        configuration={**vars(args), 'calibration': asdict(calibration),
                       'acceptance': asdict(acceptance_config),
                       'temporal_mask': asdict(temporal_mask.config),
                       'recovery': asdict(recovery.config)},
        source={'kind': 'video' if args.video else 'webcam',
                'path': args.video if args.video else None,
                'device_index': None if args.video else args.camera,
                'fps': source_fps, 'width': width, 'height': height}
    )
    print(f'Experiment run: {experiment.run_id} -> {experiment.directory}')
    recorder = None
    last = time.perf_counter()
    recorded_sessions = []
    frame = first
    count = 0
    profiler = PipelineProfiler()
    profile_stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    profile_frames = Path("experiments") / f"profile_frames_{profile_stem}.csv"
    profile_summary = Path("experiments") / f"profile_summary_{profile_stem}.csv"
    try:
        while True:
            profiler.begin_frame()
            t0 = time.perf_counter()
            profiler.start("capture")
            if count:
                ok, frame = cap.read()
                if not ok:
                    break
            profiler.stop("capture")
            profiler.start("rectification")
            frame = calibration.rectify(frame)
            profiler.stop("rectification")
            count += 1
            controls = (cv2.getTrackbarPos('yaw + 180', 'Controls') - 180, cv2.getTrackbarPos('pitch + 90', 'Controls') - 90, cv2.getTrackbarPos('roll + 180', 'Controls') - 180)
            # pose = pose_provider.update(frame, t0, controls=controls)
            
            t_seg = time.perf_counter()

            alpha = np.zeros((height, width), np.float32)
            tracking_mask = np.zeros((height, width), dtype=np.uint8)

            if segmenter is not None:
                profiler.start("yolo_inference")
                raw_mask = segmenter.predict(frame)
                profiler.stop("yolo_inference")
                profiler.start("mask_refinement")
                
                if args.refine_version == 1:
                    alpha = refine_mask(raw_mask)
                else:
                    alpha = refine_mask_v2(raw_mask)

                profiler.stop("mask_refinement")
                profiler.start("foreground_exclusion")
                tracking_mask = create_tracking_mask(raw_mask)
                tracking_mask = temporal_mask.update(frame, tracking_mask, detected=bool(np.any(raw_mask >= 0.5)))
                profiler.stop("foreground_exclusion")

            # Step 1: Track background features
            profiler.start("optical_flow")
            old_points, new_points = feature_tracker.track_features(frame, foreground_mask=tracking_mask)
            profiler.stop("optical_flow")
            profiler.start("motion_estimation")
            # Step 2: Diagnose rotation on the RANSAC inliers and conservatively
            # accept, hold, or reject the increment. Only AUTO integrates pose.
            decision = None
            if auto_tracking and not (recovery.state == RecoveryState.WARMUP and recovery.warmup_left > 0):
                decision = feature_tracker.validate_rotation(
                    old_points, new_points, frame_size=(width, height), config=acceptance_config,
                    mask_status=temporal_mask.last_status if segmenter is not None else "DISABLED"
                )
            if auto_tracking:
                recovery.update(frame, tracking_mask, decision)
            inliers = decision.inliers if decision is not None else 0
            tracking_confidence = decision.confidence if decision is not None else 0.0
            rotation_accepted = decision.accepted if decision is not None else False

            profiler.stop("motion_estimation")
            profiler.start("pose_update")
            # Step 4: Select automatic or manual pose
            if auto_tracking:
                feature_tracker.smooth_rotation()
                yaw, pitch, roll = (feature_tracker.get_camera_orientation())
                
                csv_writer.writerow([count, time.perf_counter(), len(old_points) if old_points is not None else 0, inliers, tracking_confidence, int(rotation_accepted), yaw, pitch, roll, decision.state.value if decision else "WARMUP", decision.reason if decision else "recovery_warmup", decision.rotation_rmse_px if decision else float("nan"), decision.occupied_cells if decision else 0, decision.coverage_fraction if decision else 0.0, decision.innovation_deg if decision else float("nan"), temporal_mask.last_status if segmenter is not None else "DISABLED", int(np.count_nonzero(tracking_mask)), decision.confidence if decision else 0.0, decision.mask_reliability if decision else 0.0, recovery.state.value, recovery.failures])

            else:
                yaw, pitch, roll = controls

            # Step 5: Render with selected pose
            pose = pose_provider.update(frame, t0, controls=(yaw, pitch, roll))
            
            profiler.stop("pose_update")
            profiler.start("visualisation")
            # Step 6: Visualize feature tracking
            tracking_preview = feature_tracker.draw_tracking(frame, old_points, new_points)

            cv2.putText(tracking_preview, f"Inliers: {inliers} | " f"Yaw: {yaw:.1f} " f"Pitch: {pitch:.1f} " f"Roll: {roll:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            if decision is not None:
                cv2.putText(tracking_preview, f"Tracking: {decision.state.value} confidence={decision.confidence:.2f} ({decision.reason})", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Background Optical Flow", tracking_preview)


            profiler.stop("visualisation")
            t_render = time.perf_counter()
            profiler.start("panorama_render")
            background = renderer.render(pose.yaw, pose.pitch, pose.roll)
            profiler.stop("panorama_render")
            profiler.start("compositing")
            output = composite(frame, background, alpha)
            profiler.stop("compositing")
            t_end = time.perf_counter()
            fps = 1 / max(t_end - last, 1e-6)
            last = t_end
            info = f'FPS {fps:.1f} | YOLO {profiler._times.get("yolo_inference",0):.1f}ms | motion {profiler._times.get("motion_estimation",0):.1f}ms'
            cv2.putText(output, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(output, f'Yaw {pose.yaw:.0f} Pitch {pose.pitch:.0f} Roll {pose.roll:.0f}', (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
            # Record clean composited output, without debugging overlays.
            if recorder is not None:
                if args.video:
                    # Recording starts with first frame after pressing r; output time starts at 0.
                    record_timestamp = (count - 1) / source_fps
                else:
                    record_timestamp = time.perf_counter()
                recorder.add(composite(frame, background, alpha),
                             input_index=count - 1, timestamp_seconds=record_timestamp)
            tiles = np.hstack([frame, (alpha[..., None] * np.ones((1, 1, 3)) * 255).astype(np.uint8), output])
            cv2.imshow('MotionMatte Prototype', tiles)


            profile_row = profiler.finish_frame(count, mode="AUTO" if auto_tracking else "MANUAL",
                                                status=decision.state.value if decision else "WARMUP" if auto_tracking else "MANUAL")
            stats = recorder if recorder is not None else None
            experiment.log_frame(
                count, frame_profile=profile_row,
                source_frame_index=count - 1,
                input_timestamp_s=(count - 1) / source_fps if args.video else None,
                capture_timestamp_s=t0 if not args.video else None,
                processing_timestamp_s=time.perf_counter(),
                mode='AUTO' if auto_tracking else 'MANUAL',
                yaw_deg=pose.yaw, pitch_deg=pose.pitch, roll_deg=pose.roll,
                tracking_state=decision.state.value if decision else ('WARMUP' if auto_tracking else 'MANUAL'),
                tracking_reason=decision.reason if decision else ('recovery_warmup' if auto_tracking else 'manual'),
                rotation_accepted=bool(decision.accepted) if decision else False,
                matches=len(old_points) if old_points is not None else 0,
                inliers=inliers,
                inlier_ratio=(inliers/len(old_points)) if old_points is not None and len(old_points) else 0.0,
                rotation_rmse_px=decision.rotation_rmse_px if decision else None,
                homography_rmse_px=getattr(decision, 'homography_rmse_px', None) if decision else None,
                rotation_innovation_deg=decision.innovation_deg if decision else None,
                occupied_cells=decision.occupied_cells if decision else 0,
                coverage_fraction=decision.coverage_fraction if decision else 0.0,
                tracking_confidence=decision.confidence if decision else 0.0,
                mask_reliability=decision.mask_reliability if decision else 0.0,
                mask_status=temporal_mask.last_status if segmenter is not None else 'DISABLED',
                excluded_pixels=int(np.count_nonzero(tracking_mask)),
                recovery_state=recovery.state.value, recovery_failures=recovery.failures,
                recording_active=recorder is not None,
                recording_input_frames=stats.input_frames if stats else 0,
                recording_written_frames=stats.written_frames if stats else 0,
                recording_repeated_frames=stats.repeated_frames if stats else 0,
                recording_skipped_frames=stats.skipped_frames if stats else 0,
                recording_output_fps=stats.fps if stats else None,
            )
            key = cv2.waitKey(1) & 0xFF
            if key == ord("a"):
                new_mode = "MANUAL" if auto_tracking else "AUTO"
                recovery.set_mode(new_mode, (yaw, pitch, roll), frame, tracking_mask)
                auto_tracking = (new_mode == "AUTO")
                if not auto_tracking:
                    # Match UI controls to the last displayed automatic pose.
                    for key_name, value, offset, high in (
                        ('yaw + 180', yaw, 180, 360),
                        ('pitch + 90', pitch, 90, 180),
                        ('roll + 180', roll, 180, 360),
                    ):
                        cv2.setTrackbarPos(key_name, 'Controls', int(np.clip(round(value + offset), 0, high)))
                print(f"\nCamera tracking mode: {new_mode}")
            if key == ord("c"):
                feature_tracker.reset_rotation()
                recovery._reseed(frame, tracking_mask)
                if not auto_tracking:
                    for key_name, offset in (("yaw + 180",180),("pitch + 90",90),("roll + 180",180)):
                        cv2.setTrackbarPos(key_name, "Controls", offset)
                print("\nCamera orientation reset")
            if key in (ord('q'), 27):
                break
            if key == ord('r'):
                if recorder is not None:
                    finished = recorder
                    recorder = None
                    stats = finished.close()
                    experiment.add_recording(finished.path, stats, timing_csv=finished.ledger_path)
                    print(f'Recording stopped: {stats}')
                else:
                    recorder, path = start_recording(width, height, source_fps=source_fps, live=not bool(args.video))
                    print(f'Recording to {path} ({source_fps:g} FPS CFR)')
            if key == ord('s'):
                Path('outputs').mkdir(exist_ok=True)
                path = Path('outputs') / f'frame_{datetime.now():%Y%m%d_%H%M%S}.png'
                cv2.imwrite(str(path), composite(frame, background, alpha))
                print(f'Saved {path}')
    finally:
        if recorder is not None:
            stats = recorder.close()
            experiment.add_recording(recorder.path, stats, timing_csv=recorder.ledger_path)
            print(f'Recording finalised: {stats}')
        cap.release()
        cv2.destroyAllWindows()
        motion_log.close()
        profiler.save(profile_frames, profile_summary, warmup=args.profile_warmup)
        experiment.close(profiling_frames=profile_frames, profiling_summary=profile_summary, motion_csv=log_path)
        print(f"Saved profiling: {profile_frames}, {profile_summary}")
        print(f"Saved experiment manifest: {experiment.manifest_path}")


if __name__ == '__main__':
    main()
