import numpy as np
import pytest
from panovideo.subject_tracking import SubjectTracker, SubjectConfig, PersonDetection

H,W=100,160

def person(x,y=25,w=22,h=40):
    mask=np.zeros((H,W),np.float32)
    mask[y:y+h,x:x+w]=1
    return PersonDetection(np.array([x,y,x+w,y+h],np.float64),mask)


def test_centre_initial_selection():
    t=SubjectTracker()
    assert t.update([person(10),person(70)],(H,W)).box[0]==70
    assert t.last_index==1


def test_stable_under_reordering():
    t=SubjectTracker()
    t.update([person(70),person(10)],(H,W))
    assert t.update([person(10),person(72)],(H,W)).box[0]==72
    assert t.last_index==1


def test_crossing_does_not_jump_to_distant_subject():
    t=SubjectTracker()
    t.update([person(70)],(H,W))
    assert t.update([person(72),person(125)],(H,W)).box[0]==72
    assert t.update([person(120),person(74)],(H,W)).box[0]==74


def test_short_disappearance_preserves_target():
    t=SubjectTracker()
    t.update([person(70)],(H,W))
    for _ in range(2):
        assert t.update([], (H,W)) is None
    assert t.update([person(72)],(H,W)).box[0]==72
    assert t.last_status=='TRACKED'


def test_disappearance_does_not_switch_to_bystander():
    t=SubjectTracker()
    t.update([person(70)],(H,W))
    assert t.update([person(5)],(H,W)) is None
    assert t.target is not None


def test_expiry_allows_new_acquisition_next_frame():
    t=SubjectTracker(SubjectConfig(max_missing=2))
    t.update([person(70)],(H,W))
    for _ in range(3):
        assert t.update([], (H,W)) is None
    assert t.last_status=='EXPIRED'
    assert t.update([person(10)],(H,W)).box[0]==10


def test_no_mask_retained_for_compositing_during_dropout():
    t=SubjectTracker()
    t.update([person(70)],(H,W))
    assert t.update([], (H,W)) is None


def test_non_overlapping_new_subject_is_rejected():
    t=SubjectTracker()
    t.update([person(20)],(H,W))
    assert t.update([person(115)],(H,W)) is None


def test_iou_identical_and_disjoint():
    assert SubjectTracker.box_iou(person(10).box, person(10).box)==1
    assert SubjectTracker.box_iou(person(10).box, person(100).box)==0
    assert SubjectTracker.mask_iou(person(10).mask,person(10).mask)==1
    assert SubjectTracker.mask_iou(person(10).mask,person(100).mask)==0


def test_invalid_mask_size():
    t=SubjectTracker()
    with pytest.raises(ValueError):
        t.update([PersonDetection(person(5).box,np.zeros((3,3)))],(H,W))


def test_invalid_box():
    t=SubjectTracker()
    with pytest.raises(ValueError):
        t.update([PersonDetection(np.array([np.nan,0,1,2]), np.zeros((H,W)))],(H,W))


def test_reset_clears_identity():
    t=SubjectTracker()
    t.update([person(70)],(H,W))
    t.reset()
    assert t.target is None and t.last_status=='EMPTY'


def test_stationary_candidate_beats_another_close_candidate():
    t=SubjectTracker()
    t.update([person(70)],(H,W))
    out=t.update([person(70),person(85)],(H,W))
    assert out is not None and out.box[0]==70


def test_ambiguous_candidates_return_none():
    t=SubjectTracker()
    t.update([person(70)],(H,W))
    assert t.update([person(69),person(71)],(H,W)) is None
    assert t.last_status=='AMBIGUOUS'


def test_reentry_after_expiry_requires_reacquisition():
    t=SubjectTracker(SubjectConfig(max_missing=1))
    t.update([person(70)],(H,W))
    t.update([], (H,W)); t.update([], (H,W))
    assert t.update([person(10),person(70)],(H,W)).box[0]==70
    assert t.last_status=='ACQUIRED'


def test_zero_missing_expires_immediately():
    t=SubjectTracker(SubjectConfig(max_missing=0))
    t.update([person(70)],(H,W))
    assert t.update([], (H,W)) is None
    assert t.target is None

class _FakeTensor:
    def __init__(self, data):
        self._data = np.asarray(data)
    def cpu(self):
        return self
    def numpy(self):
        return self._data
    def __getitem__(self, index):
        return _FakeTensor(self._data[index])
    def item(self):
        return self._data.item()

class _FakeModel:
    def __init__(self, frames):
        self.frames = iter(frames)
    def predict(self, **kwargs):
        from types import SimpleNamespace
        detections = next(self.frames)
        if not detections:
            return [SimpleNamespace(masks=None,boxes=None)]
        boxes=np.array([d.box for d in detections])
        masks=np.array([d.mask for d in detections])
        return [SimpleNamespace(masks=SimpleNamespace(data=_FakeTensor(masks)),
            boxes=SimpleNamespace(cls=_FakeTensor(np.zeros(len(detections))),
            xyxy=_FakeTensor(boxes),conf=_FakeTensor(np.full(len(detections), .9))))]


def test_person_segmenter_predict_associates_instead_of_centre_switch():
    from panovideo.segmenter import PersonSegmenter
    segmenter=PersonSegmenter.__new__(PersonSegmenter)
    segmenter.model=_FakeModel([[person(70),person(10)], [person(15),person(73)]])
    segmenter.device='cpu'
    segmenter.imgsz=416
    segmenter.subject_tracker=SubjectTracker()
    frame=np.zeros((H,W,3),np.uint8)
    first=segmenter.predict(frame)
    second=segmenter.predict(frame)
    assert first[45,80]==1
    assert second[45,80]==1
    assert second[45,20]==0


def test_person_segmenter_missing_returns_empty_mask():
    from panovideo.segmenter import PersonSegmenter
    segmenter=PersonSegmenter.__new__(PersonSegmenter)
    segmenter.model=_FakeModel([[person(70)], []])
    segmenter.device='cpu'; segmenter.imgsz=416
    segmenter.subject_tracker=SubjectTracker()
    frame=np.zeros((H,W,3),np.uint8)
    assert np.any(segmenter.predict(frame))
    assert not np.any(segmenter.predict(frame))
    assert segmenter.subject_tracker.last_status=='MISSING'
