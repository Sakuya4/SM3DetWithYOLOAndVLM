"""
影像處理模組
負責車輛標註、ROI 繪製和違停標示
"""

import cv2
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Optional
from .roi_manager import ROIManager
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.roi_manager = ROIManager()
    
    def draw_vehicle_box(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                        vehicle_type: str, track_id: int, confidence: float, 
                        dwell_time: float, is_violation: bool = False) -> np.ndarray:
        """在畫面上繪製車輛框和標籤"""
        try:
            annotated_frame = frame.copy()
            x1, y1, x2, y2 = map(int, bbox)
            
            # 根據違停狀態決定顏色
            if is_violation:
                color = (0, 0, 255)  # 紅色：違停
            else:
                color = (0, 255, 0)  # 綠色：正常
            
            # 畫車輛框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # 添加標籤
            label = f"{vehicle_type}#{track_id} {confidence:.2f}"
            if dwell_time >= 1.0:
                label += f" | {dwell_time:.0f}s"
            if is_violation:
                label += " [違停!]"
            
            # 計算標籤位置
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            label_y = max(y1 - 10, 20)
            
            # 畫標籤背景
            cv2.rectangle(annotated_frame, (x1, label_y - label_size[1] - 10), 
                         (x1 + label_size[0] + 10, label_y + 5), color, -1)
            cv2.rectangle(annotated_frame, (x1, label_y - label_size[1] - 10), 
                         (x1 + label_size[0] + 10, label_y + 5), (255, 255, 255), 2)
            
            # 畫標籤文字
            cv2.putText(annotated_frame, label, (x1 + 5, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"繪製車輛框失敗: {e}")
            return frame
    
    def create_violation_frame(self, frame: np.ndarray, vehicle_id: str, 
                             vehicle_type: str, confidence: float, duration: float, 
                             extra_info: Optional[Dict] = None) -> np.ndarray:
        """創建違停截圖，包含車輛標註"""
        try:
            # 複製原始畫面
            annotated_frame = frame.copy()
            
            # 如果有車輛 bbox 資訊，畫出車輛框
            if extra_info and 'bbox' in extra_info:
                bbox = extra_info['bbox']
                track_id = extra_info.get('track_id', vehicle_id)
                
                annotated_frame = self.draw_vehicle_box(
                    annotated_frame, bbox, vehicle_type, track_id, 
                    confidence, duration, duration >= 30.0
                )
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"創建違停截圖失敗: {e}")
            return frame
    
    def draw_violation_annotations(self, frame: np.ndarray, vehicle_id: str, 
                                 vehicle_type: str, confidence: float, duration: float, 
                                 in_roi: bool, extra_info: Optional[Dict] = None,
                                 red_line_violation: bool = False, red_line_ratio: float = 0.0) -> np.ndarray:
        """在截圖上繪製違停標註、ROI 和紅線資訊"""
        try:
            annotated_frame = frame.copy()
            
            # 1. 繪製 ROI 區域
            if self.roi_manager.get_roi_points():
                annotated_frame = self.roi_manager.draw_roi_on_frame(
                    annotated_frame, vehicle_in_roi=in_roi
                )
            
            # 2. 添加違停狀態標註
            status_text = f"違停狀態: {'違停' if in_roi else '未違停'}"
            status_color = (0, 0, 255) if in_roi else (0, 255, 0)
            
            cv2.putText(annotated_frame, status_text, (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
            
            # 3. 添加紅線違停資訊
            if red_line_violation:
                red_line_text = f"紅線違停: 是 (比例: {red_line_ratio:.3f})"
                cv2.putText(annotated_frame, red_line_text, (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                red_line_text = f"紅線違停: 否"
                cv2.putText(annotated_frame, red_line_text, (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 4. 添加時間戳記
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(annotated_frame, timestamp, (20, annotated_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"繪製違停標註失敗: {e}")
            return frame
    
    def get_vehicle_center(self, bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
        """獲取車輛中心點座標"""
        try:
            if bbox is None:
                return None
            
            x1, y1, x2, y2 = bbox
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            return (center_x, center_y)
            
        except Exception as e:
            logger.error(f"計算車輛中心點失敗: {e}")
            return None
    
    def set_roi_points(self, points: list) -> bool:
        """設定 ROI 點"""
        return self.roi_manager.set_roi_points(points)
    
    def get_roi_points(self) -> list:
        """獲取 ROI 點"""
        return self.roi_manager.get_roi_points()
    
    def clear_roi(self) -> bool:
        """清除 ROI"""
        return self.roi_manager.clear_roi()
