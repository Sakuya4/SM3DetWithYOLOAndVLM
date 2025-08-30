import re
import threading
import time
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from .detector import Detection
# ===== tracker.py =====

@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    cls_name: str
    score: float
    timestamp: float


class ByteTrackTracker:
    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 30, min_score: float = 0.1, min_box: int = 10):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.min_score = min_score
        self.min_box = min_box
        self._next_id = 1
        self._tracks: Dict[int, Dict] = {}

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def _iou(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def update(self, detections: List[Detection], timestamp: float) -> List[Track]:
        for tid in list(self._tracks.keys()):
            self._tracks[tid]['lost'] += 1

        for det in detections:
            x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
            w, h = x2 - x1, y2 - y1
            if det.confidence < self.min_score or w < self.min_box or h < self.min_box:
                continue

            best_iou, best_id = 0.0, None
            for tid, t in self._tracks.items():
                iou = self._iou((x1, y1, x2, y2), t['bbox'])
                if iou > best_iou:
                    best_iou, best_id = iou, tid

            if best_iou >= self.iou_threshold and best_id is not None:
                t = self._tracks[best_id]
                t['bbox'] = (x1, y1, x2, y2)
                t['score'] = float(det.confidence)
                t['cls_name'] = det.class_name
                t['timestamp'] = timestamp
                t['lost'] = 0
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    'bbox': (x1, y1, x2, y2),
                    'cls_name': det.class_name,
                    'score': float(det.confidence),
                    'timestamp': timestamp,
                    'lost': 0,
                }

        for tid in list(self._tracks.keys()):
            if self._tracks[tid]['lost'] > self.max_lost:
                del self._tracks[tid]

        out: List[Track] = []
        for tid, t in self._tracks.items():
            out.append(Track(
                track_id=tid,
                bbox=t['bbox'],
                cls_name=t['cls_name'],
                score=t['score'],
                timestamp=t['timestamp']
            ))
        return out


# ===== plate_detector.py =====

@dataclass
class PlateBox:
    quad: List[Tuple[int, int]]
    score: float


class PlateDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._loaded = False

    def load(self, model_path: Optional[str] = None) -> None:
        if model_path:
            self.model_path = model_path
        self._loaded = True

    def detect_plates(self, frame_bgr: np.ndarray, vehicle_bbox: Tuple[int, int, int, int]) -> List[PlateBox]:
        if not self._loaded:
            self.load(self.model_path)
        return []


# ===== rectifier.py =====

def warp_plate(frame_bgr: np.ndarray, quad: List[Tuple[int, int]], out_size: Tuple[int, int] = (200, 64)) -> np.ndarray:
    if len(quad) != 4:
        x1 = min(p[0] for p in quad)
        y1 = min(p[1] for p in quad)
        x2 = max(p[0] for p in quad)
        y2 = max(p[1] for p in quad)
        crop = frame_bgr[y1:y2, x1:x2]
        return cv2.resize(crop, out_size) if crop.size else crop
    src = np.array(quad, dtype=np.float32)
    dst = np.array([[0, 0], [out_size[0]-1, 0], [out_size[0]-1, out_size[1]-1], [0, out_size[1]-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame_bgr, M, out_size)


# ===== enhancement.py =====

def score_quality(img: np.ndarray) -> Dict:
    if img is None or img.size == 0:
        return {"sharpness": 0.0, "contrast": 0.0, "size": (0, 0)}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    h, w = gray.shape[:2]
    return {"sharpness": sharpness, "contrast": contrast, "size": (w, h)}


def maybe_enhance(img: np.ndarray, strategy: Dict) -> Tuple[np.ndarray, Dict]:
    q = score_quality(img)
    sharp_thr = float(strategy.get("sharpness_min", 50.0))
    enhanced = False
    out = img
    meta = {"enhanced": False, "method": "none", "quality": q}
    if q["sharpness"] < sharp_thr:
        out = img
        enhanced = False
        meta["method"] = "pass"
    meta["enhanced"] = enhanced
    return out, meta


# ===== ocr.py =====

class OCR:
    def __init__(self, backend: str = "paddleocr", config: Optional[Dict] = None):
        self.backend = backend
        self.config = config or {}
        self._ready = False

    def load(self) -> None:
        self._ready = True

    def recognize(self, plate_img: np.ndarray) -> Dict:
        if not self._ready:
            self.load()
        return {"text": "", "conf": 0.0, "boxes": []}


# ===== rules.py =====

_TW_PATTERN = re.compile(r"^[A-Z]{1,3}-?[0-9]{3,4}[A-Z0-9]?$", re.I)


def validate(text: str) -> Dict:
    return {"valid": bool(_TW_PATTERN.match(text or "")), "pattern": "TW_SIMPLE"}


def fix_confusions(text: str) -> str:
    if not text:
        return text
    mapping = {"O": "0", "I": "1", "B": "8"}
    out = []
    for ch in text:
        out.append(mapping.get(ch, ch))
    return "".join(out)


def post_process(text: str, ocr_conf: float) -> Dict:
    fixed = fix_confusions(text.upper())
    v = validate(fixed)
    conf_final = ocr_conf * (1.0 if v["valid"] else 0.8)
    return {"text_fixed": fixed, "conf_final": conf_final, "valid": v["valid"]}


# ===== plate_pipeline.py =====

@dataclass
class TrackPlateState:
    best_text: Optional[str]
    best_conf: float
    best_crop: Optional[np.ndarray]
    last_update_ts: float
    status: str


class PlatePipeline:
    def __init__(self, plate_detector: PlateDetector, ocr: OCR, enhance_strategy: Optional[Dict] = None):
        self.plate_detector = plate_detector
        self.ocr = ocr
        self.enhance_strategy = enhance_strategy or {"sharpness_min": 50.0}
        self._states: Dict[int, TrackPlateState] = {}
        self._q: "Queue[Tuple[np.ndarray, Track]]" = Queue(maxsize=64)
        self._workers: List[threading.Thread] = []
        self._running = False

    def start(self, num_workers: int = 1) -> None:
        if self._running:
            return
        self._running = True
        for _ in range(max(1, num_workers)):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._workers.append(t)

    def stop(self) -> None:
        self._running = False
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except Empty:
                break

    def submit(self, frame_bgr: np.ndarray, track: Track) -> None:
        if not self._running:
            return
        st = self._states.get(track.track_id)
        now = time.time()
        if st and now - st.last_update_ts < 0.5:
            return
        try:
            self._q.put_nowait((frame_bgr.copy(), track))
        except Exception:
            pass

    def get_state(self, track_id: int) -> Optional[TrackPlateState]:
        return self._states.get(track_id)

    def clear(self, track_id: int) -> None:
        self._states.pop(track_id, None)

    def _worker_loop(self) -> None:
        while self._running:
            try:
                frame, track = self._q.get(timeout=0.2)
            except Empty:
                continue
            try:
                x1, y1, x2, y2 = track.bbox
                x1, y1 = max(0, x1), max(0, y1)
                crop_vehicle = frame[y1:y2, x1:x2]
                plates = self.plate_detector.detect_plates(frame, track.bbox)
                best_text, best_conf, best_crop = None, 0.0, None
                for pb in plates:
                    plate_img = warp_plate(frame, pb.quad)
                    plate_img2, meta = maybe_enhance(plate_img, self.enhance_strategy)
                    o = self.ocr.recognize(plate_img2)
                    pp = post_process(o.get("text", ""), float(o.get("conf", 0.0)))
                    if pp["conf_final"] > best_conf:
                        best_conf = pp["conf_final"]
                        best_text = pp["text_fixed"]
                        best_crop = plate_img2
                self._states[track.track_id] = TrackPlateState(
                    best_text=best_text,
                    best_conf=best_conf,
                    best_crop=best_crop,
                    last_update_ts=time.time(),
                    status="ok" if best_text else "pending"
                )
            except Exception:
                self._states[track.track_id] = TrackPlateState(
                    best_text=None,
                    best_conf=0.0,
                    best_crop=None,
                    last_update_ts=time.time(),
                    status="error"
                )
