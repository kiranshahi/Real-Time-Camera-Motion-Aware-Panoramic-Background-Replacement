import csv
import numpy as np
import pytest

from panovideo.recording import TimedRecorder, usable_fps


class FakeWriter:
    def __init__(self):
        self.frames = []
        self.released = False
    def isOpened(self):
        return True
    def write(self, frame):
        self.frames.append(frame.copy())
    def release(self):
        self.released = True


def recorder(tmp_path, *, fps=20, live=False):
    writer = FakeWriter()
    rec = TimedRecorder(tmp_path / 'test.mp4', 8, 6, source_fps=fps, live=live,
                        writer_factory=lambda *args: writer)
    return rec, writer


def frame(n):
    return np.full((6, 8, 3), n, np.uint8)


def test_offline_preserves_source_fps_and_all_frames(tmp_path):
    r, w = recorder(tmp_path, fps=30, live=False)
    for i in range(5):
        r.add(frame(i), input_index=i, timestamp_seconds=i/30)
    stats = r.close()
    assert stats.written_frames == 5 and stats.output_fps == 30
    assert stats.repeated_frames == stats.skipped_frames == 0
    assert [int(f[0,0,0]) for f in w.frames] == list(range(5))


def test_offline_processing_speed_does_not_change_timing(tmp_path):
    r, w = recorder(tmp_path, fps=25, live=False)
    for i in range(4):
        r.add(frame(i), input_index=i, timestamp_seconds=i/25)
    r.close()
    with r.ledger_path.open() as f:
        rows = list(csv.DictReader(f))
    assert [float(row['output_pts_seconds']) for row in rows] == pytest.approx([0,.04,.08,.12])


def test_live_repeats_last_processed_frame_for_slow_pipeline(tmp_path):
    r, w = recorder(tmp_path, fps=10, live=True)
    r.add(frame(1), input_index=0, timestamp_seconds=100)
    r.add(frame(2), input_index=1, timestamp_seconds=100.31)
    stats = r.close()
    assert [int(f[0,0,0]) for f in w.frames] == [1,1,1,2]
    assert stats.repeated_frames == 2


def test_live_skips_fast_processed_frames(tmp_path):
    r, w = recorder(tmp_path, fps=10, live=True)
    for i, t in enumerate([1., 1.02, 1.05, 1.10]):
        r.add(frame(i), input_index=i, timestamp_seconds=t)
    stats = r.close()
    assert [int(f[0,0,0]) for f in w.frames] == [0,3]
    assert stats.skipped_frames == 2


def test_live_does_not_reuse_dropped_frame(tmp_path):
    r, w = recorder(tmp_path, fps=10, live=True)
    for i, t in enumerate([0., .03, .21]):
        r.add(frame(i), input_index=i, timestamp_seconds=t)
    r.close()
    assert [int(f[0,0,0]) for f in w.frames] == [0,0,2]


def test_live_timestamp_jitter_within_one_slot(tmp_path):
    r, w = recorder(tmp_path, fps=20, live=True)
    r.add(frame(1), input_index=0, timestamp_seconds=10)
    r.add(frame(2), input_index=1, timestamp_seconds=10.0499)
    r.add(frame(3), input_index=2, timestamp_seconds=10.0501)
    assert r.close().written_frames == 2


def test_timestamp_regression_rejected(tmp_path):
    r, _ = recorder(tmp_path)
    r.add(frame(1), input_index=0, timestamp_seconds=2)
    with pytest.raises(ValueError, match='nondecreasing'):
        r.add(frame(2), input_index=1, timestamp_seconds=1)
    r.close()


def test_duplicate_index_rejected(tmp_path):
    r, _ = recorder(tmp_path)
    r.add(frame(1), input_index=0, timestamp_seconds=1)
    with pytest.raises(ValueError, match='increase'):
        r.add(frame(2), input_index=0, timestamp_seconds=2)
    r.close()


def test_invalid_shape_rejected(tmp_path):
    r, _ = recorder(tmp_path)
    with pytest.raises(ValueError, match='dimensions'):
        r.add(frame(2)[:3], input_index=0, timestamp_seconds=1)
    r.close()


def test_offline_unknown_fps_rejected(tmp_path):
    with pytest.raises(ValueError, match='unknown/invalid FPS'):
        recorder(tmp_path, fps=float('nan'), live=False)


def test_live_unknown_fps_uses_fallback(tmp_path):
    r, _ = recorder(tmp_path, fps=float('nan'), live=True)
    assert r.fps == 20
    r.close()


def test_close_idempotent_and_sidecar(tmp_path):
    r, w = recorder(tmp_path)
    r.add(frame(1), input_index=2, timestamp_seconds=10)
    a = r.close()
    b = r.close()
    assert a == b and w.released
    with r.ledger_path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]['input_index'] == '2' and rows[0]['repeat'] == '0'


def test_max_repeat_gap_guards_memory(tmp_path):
    r, _ = recorder(tmp_path, fps=10, live=True)
    r.max_repeats = 2
    r.add(frame(1), input_index=0, timestamp_seconds=1)
    with pytest.raises(RuntimeError, match='Excessive'):
        r.add(frame(2), input_index=1, timestamp_seconds=3)
    r.close()


def test_invalid_fps_values():
    for x in (None, 0, -1, float('nan'), float('inf'), 1000):
        assert not usable_fps(x)
    assert usable_fps(29.97)
