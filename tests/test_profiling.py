import csv
from pathlib import Path
import pytest
from panovideo.profiling import PipelineProfiler, STAGES


class FakeClock:
    def __init__(self):
        self.now = 0
    def __call__(self):
        return self.now
    def advance_ms(self, t):
        self.now += int(t * 1_000_000)


def make_profiler():
    clock = FakeClock()
    return PipelineProfiler(clock=clock), clock


def test_all_expected_stages_present():
    assert {'yolo_inference','mask_refinement','optical_flow', 'motion_estimation',
            'panorama_render','compositing','rectification','foreground_exclusion'} <= set(STAGES)


def test_stage_timing_and_total_exclusion():
    p,c = make_profiler()
    p.begin_frame()
    for name,ms in [('capture',3), ('yolo_inference',12), ('optical_flow',5),
                    ('compositing',2), ('visualisation',4)]:
        p.start(name); c.advance_ms(ms); p.stop(name)
    row=p.finish_frame(1)
    assert row['total_processing_ms'] == pytest.approx(19)
    assert row['total_loop_ms'] == pytest.approx(26)
    assert row['yolo_inference_ms'] == pytest.approx(12)
    assert row['mask_refinement_ms'] == 0
    assert row['processing_fps'] == pytest.approx(1000/19)


def test_repeated_stage_accumulates():
    p,c=make_profiler(); p.begin_frame()
    for _ in range(2):
        p.start('optical_flow'); c.advance_ms(2); p.stop('optical_flow')
    assert p.finish_frame(1)['optical_flow_ms'] == pytest.approx(4)


@pytest.mark.parametrize('stage', ['unknown','', 'yolo'])
def test_reject_unknown_stage(stage):
    p,_=make_profiler(); p.begin_frame()
    with pytest.raises(ValueError): p.start(stage)


def test_reject_mismatched_stop():
    p,_=make_profiler(); p.begin_frame(); p.start('capture')
    with pytest.raises(RuntimeError):p.stop('compositing')


def test_reject_nested_stages():
    p,_=make_profiler(); p.begin_frame(); p.start('capture')
    with pytest.raises(RuntimeError):p.start('optical_flow')


def test_reject_no_begin():
    p,_=make_profiler()
    with pytest.raises(RuntimeError):p.start('capture')


def test_reject_unclosed_frame():
    p,_=make_profiler();p.begin_frame();p.start('capture')
    with pytest.raises(RuntimeError):p.finish_frame(1)


def test_summary_percentiles_and_warmup():
    p,c=make_profiler()
    for i,ms in enumerate([100,10,20,30]):
        p.begin_frame();p.start('optical_flow');c.advance_ms(ms);p.stop('optical_flow');p.finish_frame(i)
    stats={row['metric']:row for row in p.summary(warmup=1)}
    assert stats['optical_flow_ms']['samples'] == 3
    assert stats['optical_flow_ms']['mean'] == pytest.approx(20)
    assert stats['optical_flow_ms']['median'] == pytest.approx(20)
    assert stats['optical_flow_ms']['p95'] == pytest.approx(29)


def test_summary_empty_and_large_warmup():
    p,_=make_profiler()
    assert all(s['samples']==0 for s in p.summary(warmup=100))
    with pytest.raises(ValueError):p.summary(warmup=-1)


def test_save_csv_schema(tmp_path):
    p,c=make_profiler();p.begin_frame();p.start('compositing');c.advance_ms(4);p.stop('compositing');p.finish_frame(7, mode='MANUAL', status='MANUAL')
    frame,summary=tmp_path/'sub'/'frame.csv',tmp_path/'summary.csv'
    p.save(frame,summary)
    with frame.open(newline='') as f: rows=list(csv.DictReader(f))
    assert len(rows)==1
    assert rows[0]['frame']=='7' and rows[0]['compositing_ms']=='4.0'
    with summary.open(newline='') as f: data=list(csv.DictReader(f))
    assert any(r['metric']=='compositing_ms' and r['median']=='4.0' for r in data)


def test_disabled_stages_record_zero():
    p,_=make_profiler();p.begin_frame(); row=p.finish_frame(1, mode='MANUAL')
    assert row['yolo_inference_ms']==0 and row['motion_estimation_ms']==0
    assert row['processing_fps']==0
