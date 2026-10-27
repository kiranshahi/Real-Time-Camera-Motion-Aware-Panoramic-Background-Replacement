"""YOLO instance segmentation with person-class filtering."""
import cv2
import numpy as np
from .subject_tracking import SubjectTracker, PersonDetection


class PersonSegmenter:
    def __init__(self, weights: str = 'yolo26n-seg.pt', device: str = 'auto', imgsz: int = 416):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError('Install requirements.txt before using YOLO segmentation') from exc
        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz
        self.subject_tracker = SubjectTracker()

    def predict(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        args = dict(source=frame, conf=0.3, imgsz=self.imgsz, retina_masks=True, verbose=False)

        if self.device != 'auto':
            args['device'] = self.device

        result = self.model.predict(**args)[0]

        mask = np.zeros((h, w), dtype=np.float32)

        if result.masks is None or result.boxes is None:
            self.subject_tracker.update([], (h, w))
            return mask
        
        classes = result.boxes.cls.cpu().numpy().astype(int)
        indices = np.where(classes == 0)[0]  # first class is person

        if len(indices) == 0:
            self.subject_tracker.update([], (h, w))
            return mask
        
        boxes = result.boxes.xyxy.cpu().numpy()
        detections = []
        for idx in indices:
            raw = result.masks.data[idx].cpu().numpy().astype(np.float32)
            if raw.shape != (h, w):
                raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
            detections.append(PersonDetection(
                box=np.asarray(boxes[idx], dtype=np.float64),
                mask=np.clip(raw, 0.0, 1.0),
                confidence=float(result.boxes.conf[idx].item())
            ))
        selected = self.subject_tracker.update(detections, (h, w))
        return selected.mask.copy() if selected is not None else mask


def refine_mask(mask: np.ndarray, blur_sigma: float = 1.25, dilate_pixels: int = 3) -> np.ndarray:
    binary = (mask >= 0.5).astype(np.uint8)
    if dilate_pixels > 0:
        size = 2 * dilate_pixels + 1
        binary = cv2.dilate(binary, np.ones((size, size), np.uint8))
    return cv2.GaussianBlur(binary.astype(np.float32), (0, 0), blur_sigma).clip(0, 1)

def refine_mask_v2(mask: np.ndarray, blur_sigma: float = 0.8) -> np.ndarray:
    alpha = np.clip(mask.astype(np.float32), 0.0, 1.0)

    if blur_sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), blur_sigma)

    return np.clip(alpha, 0.0, 1.0)

def composite(foreground: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.asarray(alpha, dtype=np.float32)[..., None]
    return np.clip(foreground.astype(np.float32) * a + background.astype(np.float32) * (1 - a), 0, 255).astype(np.uint8)


def create_tracking_mask(mask: np.ndarray, threshold: float = 0.3, dilation: int = 10) -> np.ndarray:
    """
    Return a uint8 foreground exclusion mask.

    255 = excluded from feature tracking
      0 = valid background region
    """

    binary = (mask >= threshold).astype(np.uint8)

    if dilation > 0:
        size = 2 * dilation + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        binary = cv2.dilate(binary, kernel)
    return binary * 255