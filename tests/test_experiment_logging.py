import csv
import json
from dataclasses import dataclass
import math
import pytest
from panovideo.experiment_logging import ExperimentLogger, FRAME_FIELDS


@pytest.fixture
def logger(tmp_path):
    obj = ExperimentLogger(tmp_path, configuration={'model':'synthetic','threshold':float('nan')},
                           source={'kind':'video','fps':30}, run_id='run_001',
                           created_at='2026-01-01T00:00:00+00:00')
    yield obj
    obj.close()


def test_manifest_schema_and_configuration(logger):
    manifest=json.loads(logger.manifest_path.read_text())
    assert manifest['schema_version']=='t16.v1'
    assert manifest['run_id']=='run_001'
    assert manifest['configuration']['threshold'] is None
    assert manifest['source']['fps']==30


def test_unique_directory(tmp_path):
    a=ExperimentLogger(tmp_path, configuration={}, source={}, run_id='fixed')
    with pytest.raises(FileExistsError):
        ExperimentLogger(tmp_path, configuration={}, source={}, run_id='fixed')
    a.close()


@pytest.mark.parametrize('run_id',['../other','nested/other','hi there',''])
def test_bad_run_id(tmp_path,run_id):
    if run_id=='':
        # An empty ID auto-generates one.
        x=ExperimentLogger(tmp_path,configuration={},source={},run_id=run_id)
        assert x.run_id
        x.close()
    else:
        with pytest.raises(ValueError):
            ExperimentLogger(tmp_path,configuration={},source={},run_id=run_id)


def test_frame_schema_pose_and_diagnostics(logger):
    logger.log_frame(1, frame_profile={'optical_flow_ms':3.5,'total_processing_ms':20.0},
                     source_frame_index=0, yaw_deg=12, pitch_deg=-4, roll_deg=1,
                     tracking_state='VALID',rotation_rmse_px=0.6, mask_status='DETECTED',
                     recording_skipped_frames=2)
    logger.close()
    with logger.frames_path.open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==1
    assert tuple(rows[0])==FRAME_FIELDS
    assert rows[0]['yaw_deg']=='12'
    assert rows[0]['rotation_rmse_px']=='0.6'
    assert rows[0]['optical_flow_ms']=='3.5'
    assert rows[0]['recording_skipped_frames']=='2'


def test_missing_decision_and_nonfinite_are_safe(logger):
    logger.log_frame(1, frame_profile={}, tracking_state='WARMUP',rotation_rmse_px=float('nan'))
    logger.close()
    rows=list(csv.DictReader(logger.frames_path.open(newline='')))
    assert rows[0]['rotation_rmse_px']==''
    assert rows[0]['inliers']==''


def test_monotonic_frame_index(logger):
    logger.log_frame(2,frame_profile={})
    with pytest.raises(ValueError): logger.log_frame(2,frame_profile={})
    with pytest.raises(ValueError): logger.log_frame(1,frame_profile={})
    logger.log_frame(3,frame_profile={})


def test_unknown_column_rejected(logger):
    with pytest.raises(ValueError): logger.log_frame(1,frame_profile={},wrong_name='bad')


@dataclass
class FakeStats:
    input_frames:int=10
    written_frames:int=12
    repeated_frames:int=3
    skipped_frames:int=1
    source_fps:float=30
    output_fps:float=30


def test_multiple_recording_sessions_accumulate(logger):
    logger.add_recording('one.mp4',FakeStats(),timing_csv='one.timing.csv')
    logger.add_recording('two.mp4',FakeStats(input_frames=5,written_frames=5,repeated_frames=0,skipped_frames=0))
    logger.close()
    result=json.loads(logger.manifest_path.read_text())
    stats=result['summary']['recording']
    assert stats=={'sessions':2,'input_frames':15,'written_frames':17,'repeated_frames':3,'skipped_frames':1}
    assert result['recordings'][0]['timing_csv']=='one.timing.csv'


def test_close_idempotent_and_artifact_metadata(logger):
    p=logger.close(profiling_frames='profile.csv',profiling_summary='summary.csv',motion_csv='motion.csv')
    assert logger.close()==p
    result=json.loads(p.read_text())
    assert result['artifacts']['profile_summary_csv']=='summary.csv'
    assert result['summary']['frames_logged']==0
    with pytest.raises(RuntimeError): logger.log_frame(1,frame_profile={})


def test_frame_counter_manifest(logger):
    for frame in range(1,5):
        logger.log_frame(frame,frame_profile={'total_loop_ms':float(frame)})
    logger.close()
    assert json.loads(logger.manifest_path.read_text())['summary']['frames_logged']==4


def test_duplicate_run_id_random(tmp_path):
    a=ExperimentLogger(tmp_path,configuration={},source={})
    b=ExperimentLogger(tmp_path,configuration={},source={})
    assert a.run_id != b.run_id
    a.close(); b.close()


def test_capture_and_recording_fields_exist():
    for key in ('input_timestamp_s','capture_timestamp_s','rotation_innovation_deg','mask_reliability',
                'recording_repeated_frames','recording_skipped_frames','yolo_inference_ms','total_loop_ms'):
        assert key in FRAME_FIELDS


def test_timestamps_monotonic_and_nonnegative(logger):
    logger.log_frame(1, frame_profile={}, input_timestamp_s=0.0,processing_timestamp_s=10.0)
    logger.log_frame(2, frame_profile={}, input_timestamp_s=1/30,processing_timestamp_s=10.1)
    with pytest.raises(ValueError, match='Nonmonotonic'):
        logger.log_frame(3,frame_profile={},input_timestamp_s=0.01)
    with pytest.raises(ValueError, match='Invalid'):
        logger.log_frame(3,frame_profile={},input_timestamp_s=-1.0)
    logger.log_frame(3,frame_profile={},input_timestamp_s=2/30)


def test_manifest_has_config_and_recording_data_without_nan(logger):
    logger.log_frame(1,frame_profile={},rotation_rmse_px=float('inf'))
    logger.add_recording('test.mp4',FakeStats())
    logger.close()
    assert 'NaN' not in logger.manifest_path.read_text()
    assert 'Infinity' not in logger.manifest_path.read_text()
