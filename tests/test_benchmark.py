import csv
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from panovideo.benchmark import BenchmarkConfig, run_benchmark, compare_summaries


def fixture_media(tmp_path, n=8):
    video = tmp_path / 'source.avi'
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 15.0, (96, 64))
    assert out.isOpened()
    rng = np.random.default_rng(19)
    background = rng.integers(0, 220, (64, 96, 3), dtype=np.uint8)
    for idx in range(n):
        frame = np.roll(background, idx % 3, axis=1).copy()
        cv2.rectangle(frame, (35 + idx % 3, 15), (65 + idx % 3, 55), (180, 180, 180), -1)
        out.write(frame)
    out.release()
    panorama = tmp_path / 'panorama.png'
    cv2.imwrite(str(panorama), np.tile(background, (2, 2, 1)))
    return str(video), str(panorama)


@pytest.mark.parametrize('kwargs', [{'warmup':-1}, {'max_frames':0}, {'refine_version':3}, {'threads':0}, {'fov':180}, {'label':'../../bad'}])
def test_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        BenchmarkConfig(video='x',panorama='y', **kwargs)


def test_no_segmentation_headless_and_complete_outputs(tmp_path, monkeypatch):
    video, panorama = fixture_media(tmp_path)
    monkeypatch.setattr(cv2,'namedWindow', lambda *a,**k: pytest.fail('UI used'))
    monkeypatch.setattr(cv2,'imshow', lambda *a,**k: pytest.fail('UI used'))
    result = run_benchmark(BenchmarkConfig(video, panorama, output_dir=str(tmp_path/'runs'),max_frames=7,warmup=2))
    assert result['frames'] == 7
    assert result['measured_frames'] == 5
    assert result['metrics']['total_processing_ms']['samples']==5
    assert result['metrics']['yolo_inference_ms']['mean']==0
    run_dir=Path(result['manifest']).parent
    manifest=json.loads((run_dir/'manifest.json').read_text())
    assert manifest['summary']['frames_logged']==7
    assert manifest['source']['sha256']==result['source']['sha256']
    assert (run_dir/'profile_frames.csv').is_file()
    with (run_dir/'frames.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==7
    assert [int(x['source_frame_index']) for x in rows]==list(range(7))
    assert all(x['mask_status']=='DISABLED' for x in rows)
    assert all(float(x['total_processing_ms'])>=0 for x in rows)


class MockSegmenter:
    def __init__(self, weights, device, imgsz):
        pass
    def predict(self, frame):
        mask=np.zeros(frame.shape[:2],np.float32)
        mask[15:55,35:66]=1
        return mask


def test_mock_segmenter(tmp_path):
    video, pano=fixture_media(tmp_path)
    result=run_benchmark(BenchmarkConfig(video,pano,output_dir=str(tmp_path/'runs'),segmentation=True,max_frames=5,warmup=1),segmenter_factory=MockSegmenter)
    assert result['measured_frames']==4
    assert result['metrics']['yolo_inference_ms']['samples']==4
    with (Path(result['manifest']).parent/'frames.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==5
    assert all(int(row['excluded_pixels'])>0 for row in rows)


def test_warmup_over_all_frames(tmp_path):
    video,pano=fixture_media(tmp_path,n=3)
    result=run_benchmark(BenchmarkConfig(video,pano,output_dir=str(tmp_path/'runs'),warmup=10))
    assert result['measured_frames']==0
    assert result['metrics']['total_processing_ms']['samples']==0


def test_compare_writes_mean_median_p95(tmp_path):
    demo={'run_id':'test','label':'cpu','frames':5,'measured_frames':4,
          'metrics':{'total_processing_ms':{'mean':12,'median':11,'p95':14}}}
    path=compare_summaries([demo,demo],tmp_path/'comparison.csv')
    with path.open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==2 and rows[0]['p95']=='14' and rows[0]['label']=='cpu'


def test_source_content_hash_is_stable(tmp_path):
    video,pano=fixture_media(tmp_path)
    c=BenchmarkConfig(video,pano,output_dir=str(tmp_path/'runs'),max_frames=2,warmup=0)
    first=run_benchmark(c)
    second=run_benchmark(c)
    assert first['source']['sha256']==second['source']['sha256']
    assert first['run_id']!=second['run_id']
    assert first['source']['panorama_sha256']==second['source']['panorama_sha256']


def test_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_benchmark(BenchmarkConfig('missing.avi','missing.png',output_dir=str(tmp_path)))
