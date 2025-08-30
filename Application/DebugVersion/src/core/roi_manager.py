"""
ROI (Region of Interest) 管理模組
負責禁停區的設定、儲存和碰撞檢測
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class ROIManager:
    def __init__(self, config_path: str = "config/roi_config.json"):
        self.roi_points: List[Tuple[int, int]] = []
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(exist_ok=True)
        self._load_roi_config()
    
    def set_roi_points(self, points: List[Tuple[int, int]]) -> bool:
        """設定 ROI 點"""
        try:
            if len(points) >= 3:  # 至少需要3個點形成多邊形
                self.roi_points = points
                self._save_roi_config()
                logger.info(f"已設定 ROI: {points}")
                return True
            else:
                logger.warning("ROI 點數不足，至少需要3個點")
                return False
        except Exception as e:
            logger.error(f"設定 ROI 失敗: {e}")
            return False
    
    def get_roi_points(self) -> List[Tuple[int, int]]:
        """獲取當前 ROI 點"""
        return self.roi_points.copy()
    
    def clear_roi(self) -> bool:
        """清除 ROI 設定"""
        try:
            self.roi_points.clear()
            self._save_roi_config()
            logger.info("已清除 ROI 設定")
            return True
        except Exception as e:
            logger.error(f"清除 ROI 失敗: {e}")
            return False
    
    def get_default_roi(self, frame_shape: Tuple[int, int, int]) -> List[Tuple[int, int]]:
        """獲取預設的 ROI 區域（畫面中央的矩形區域）"""
        try:
            h, w = frame_shape[:2]
            # 預設 ROI：畫面中央 60% 的區域
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            
            roi_points = [
                (margin_x, margin_y),                    # 左上
                (w - margin_x, margin_y),                # 右上
                (w - margin_x, h - margin_y),            # 右下
                (margin_x, h - margin_y)                 # 左下
            ]
            
            logger.info(f"已設定預設 ROI: {roi_points}")
            return roi_points
            
        except Exception as e:
            logger.error(f"設定預設 ROI 失敗: {e}")
            return []
    
    def is_point_in_roi(self, point: Tuple[int, int]) -> bool:
        """檢查點是否在 ROI 內"""
        try:
            if not self.roi_points or len(self.roi_points) < 3:
                return False
            
            # 使用 OpenCV 的 pointPolygonTest 檢查點是否在多邊形內
            roi_array = np.array(self.roi_points, dtype=np.int32)
            result = cv2.pointPolygonTest(roi_array, point, False)
            
            # result > 0 表示點在多邊形內
            return result > 0
            
        except Exception as e:
            logger.error(f"檢查點是否在 ROI 內失敗: {e}")
            return False
    
    def is_bbox_in_roi(self, bbox: Tuple[int, int, int, int]) -> bool:
        """檢查邊界框是否與 ROI 重疊"""
        try:
            if not self.roi_points or len(self.roi_points) < 3:
                return False
            
            x1, y1, x2, y2 = bbox
            
            # 檢查邊界框的四個角點
            corners = [
                (x1, y1),  # 左上
                (x2, y1),  # 右上
                (x2, y2),  # 右下
                (x1, y2)   # 左下
            ]
            
            # 如果任何一個角點在 ROI 內，則認為重疊
            for corner in corners:
                if self.is_point_in_roi(corner):
                    return True
            
            # 檢查 ROI 是否完全包含邊界框
            roi_array = np.array(self.roi_points, dtype=np.int32)
            bbox_array = np.array(corners, dtype=np.int32)
            
            # 計算交集面積
            intersection = cv2.intersectConvexConvex(roi_array, bbox_array)[0]
            if intersection > 0:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"檢查邊界框是否在 ROI 內失敗: {e}")
            return False
    
    def draw_roi_on_frame(self, frame: np.ndarray, 
                          vehicle_in_roi: bool = False) -> np.ndarray:
        """在畫面上繪製 ROI"""
        try:
            if not self.roi_points:
                return frame
            
            annotated_frame = frame.copy()
            roi_array = np.array(self.roi_points, dtype=np.int32)
            
            # 根據車輛是否在 ROI 內決定顏色
            if vehicle_in_roi:
                roi_color = (0, 0, 255)  # 紅色：禁停區
                roi_label = "禁停區"
            else:
                roi_color = (0, 255, 255)  # 黃色：非禁停區
                roi_label = "非禁停區"
            
            # 畫 ROI 多邊形
            cv2.polylines(annotated_frame, [roi_array], True, roi_color, 3)
            
            # 填充 ROI 區域（半透明）
            overlay = annotated_frame.copy()
            cv2.fillPoly(overlay, [roi_array], roi_color)
            cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
            
            # 添加 ROI 標籤
            if self.roi_points:
                label_x = self.roi_points[0][0] + 10
                label_y = self.roi_points[0][1] + 30
                cv2.putText(annotated_frame, roi_label, (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, roi_color, 2)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"繪製 ROI 失敗: {e}")
            return frame
    
    def _load_roi_config(self):
        """載入 ROI 設定"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.roi_points = config.get('roi_points', [])
                    logger.info(f"已載入 ROI 設定: {self.roi_points}")
            else:
                logger.info("ROI 設定檔不存在，使用預設設定")
        except Exception as e:
            logger.error(f"載入 ROI 設定失敗: {e}")
    
    def _save_roi_config(self):
        """儲存 ROI 設定"""
        try:
            config = {
                'roi_points': self.roi_points,
                'timestamp': str(np.datetime64('now'))
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"ROI 設定已儲存到: {self.config_path}")
            
        except Exception as e:
            logger.error(f"儲存 ROI 設定失敗: {e}")
