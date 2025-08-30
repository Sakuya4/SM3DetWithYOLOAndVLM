import logging
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)
@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str


class BaseDetector:
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        raise NotImplementedError

    def get_class_names(self) -> List[str]:
        raise NotImplementedError


class UltralyticsDetector(BaseDetector):
    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model_name = model_name
        self.model: Optional[YOLO] = None
        self.class_names: List[str] = []
        self._load()

    def _load(self) -> None:
        try:
            logger.info(f"載入偵測模型: {self.model_name}")
            self.model = YOLO(self.model_name)
            if hasattr(self.model, "names") and isinstance(self.model.names, dict):
                self.class_names = [self.model.names[i] for i in sorted(self.model.names.keys())]
            logger.info(f"偵測模型已就緒: {self.model_name}")
        except Exception as e:
            logger.error(f"載入偵測模型失敗: {e}")
            self.model = None

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if self.model is None:
            return []

        detections: List[Detection] = []
        results = self.model.predict(source=frame_bgr, imgsz=640, verbose=False)
        for r in results:
            names_map = getattr(r, "names", {}) if hasattr(r, "names") else {}
            if hasattr(r, "boxes"):
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    cls_name = names_map.get(cls, str(cls))
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append(Detection(x1, y1, x2, y2, conf, cls, cls_name))
        return detections

    def get_class_names(self) -> List[str]:
        return self.class_names


