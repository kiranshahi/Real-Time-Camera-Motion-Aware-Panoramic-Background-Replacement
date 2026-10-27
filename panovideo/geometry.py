"""Rotation-only equirectangular rendering (camera x right, y down, z forward)."""
import cv2
import numpy as np
from .calibration import CameraCalibration


def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Degrees; positive yaw looks right, positive pitch looks up, roll clockwise."""
    y, p, r = np.deg2rad([yaw, pitch, roll])
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    return ry @ rx @ rz


class PanoramaRenderer:
    def __init__(self, panorama: np.ndarray, width: int, height: int, hfov_degrees: float = 70, *, calibration=None):
        if panorama is None or panorama.ndim != 3 or panorama.shape[2] != 3:
            raise ValueError('Panorama must be a BGR color image.')
        self.panorama = panorama
        # Pad only the longitude axis: its end columns are neighbours on a sphere.
        self._panorama_horizontal_wrap = np.concatenate(
            (panorama[:, -1:], panorama, panorama[:, :1]), axis=1
        )
        self.width, self.height = int(width), int(height)
        self.calibration = calibration if calibration is not None else CameraCalibration.from_hfov(width, height, hfov_degrees)
        if (self.calibration.width, self.calibration.height) != (self.width, self.height):
            raise ValueError("Renderer dimensions and calibration must match")
        if calibration is None:
            # Retain pre-T10 legacy ray sampling for existing standalone renderer users.
            # The calibrated main pipeline always supplies the shared calibration.
            legacy = CameraCalibration(width, height, self.calibration.fx, self.calibration.fy,
                                       (width - 1) / 2, (height - 1) / 2)
            self.rays = legacy.pixel_rays().reshape(-1, 3).astype(np.float32)
        else:
            self.rays = self.calibration.pixel_rays().reshape(-1, 3).astype(np.float32)

    def render(self, yaw: float = 0, pitch: float = 0, roll: float = 0) -> np.ndarray:
        R = rotation_matrix(yaw, pitch, roll)
        world = self.rays @ R.T
        lon = np.arctan2(world[:, 0], world[:, 2])
        lat = np.arctan2(-world[:, 1], np.hypot(world[:, 0], world[:, 2]))
        ph, pw = self.panorama.shape[:2]
        # Longitude is periodic; pad one column either side so bilinear
        # interpolation remains continuous at the +/-180-degree seam.
        map_x = (np.mod(lon / (2 * np.pi) + 0.5, 1.0) * pw + 1.0)
        # Latitude has physical endpoints (north/south poles), not a seam.
        # Never wrap vertically into the opposite pole.
        map_y = np.clip((0.5 - lat / np.pi) * (ph - 1), 0.0, ph - 1.0)
        map_x = map_x.astype(np.float32).reshape(self.height, self.width)
        map_y = map_y.astype(np.float32).reshape(self.height, self.width)
        return cv2.remap(self._panorama_horizontal_wrap, map_x, map_y,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
