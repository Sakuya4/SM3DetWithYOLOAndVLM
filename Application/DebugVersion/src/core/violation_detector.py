"""
違停判定模組
整合停留時間、ROI 檢查和速度計算來判定違停
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from .roi_manager import ROIManager
import logging

logger = logging.getLogger(__name__)

class ViolationDetector:
    def __init__(self, 
                 dwell_threshold: float = 30.0,
                 speed_threshold: float = 0.1,
                 min_track_points: int = 5):
        self.dwell_threshold = dwell_threshold
        self.speed_threshold = speed_threshold
        self.min_track_points = min_track_points
        
        # 追蹤歷史：{track_id: [(timestamp, center_x, center_y), ...]}
        self.track_history: Dict[int, List[Tuple[float, int, int]]] = {}
        self.first_seen: Dict[int, float] = {}
        self.violation_records: Dict[int, Dict] = {}
        
        # ROI 管理器
        self.roi_manager = ROIManager()
    
    def update_track(self, track_id: int, bbox: Tuple[int, int, int, int], 
                    current_time: float) -> None:
        """更新車輛追蹤資訊"""
        try:
            # 計算車輛中心點
            x1, y1, x2, y2 = bbox
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # 記錄首次出現時間
            if track_id not in self.first_seen:
                self.first_seen[track_id] = current_time
            
            # 更新追蹤歷史
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            self.track_history[track_id].append((current_time, center_x, center_y))
            
            # 保持歷史記錄在合理範圍內（最近100個點）
            if len(self.track_history[track_id]) > 100:
                self.track_history[track_id] = self.track_history[track_id][-100:]
                
        except Exception as e:
            logger.error(f"更新追蹤資訊失敗: {e}")
    
    def calculate_speed(self, track_id: int) -> float:
        """計算車輛移動速度 (pixel/second)"""
        try:
            if track_id not in self.track_history:
                return 0.0
            
            track_points = self.track_history[track_id]
            if len(track_points) < self.min_track_points:
                return 0.0
            
            # 取最近幾個點計算速度
            recent_points = track_points[-self.min_track_points:]
            
            # 計算總移動距離
            total_distance = 0.0
            total_time = 0.0
            
            for i in range(1, len(recent_points)):
                prev_time, prev_x, prev_y = recent_points[i-1]
                curr_time, curr_x, curr_y = recent_points[i]
                
                # 計算歐幾里得距離
                distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                time_diff = curr_time - prev_time
                
                total_distance += distance
                total_time += time_diff
            
            if total_time > 0:
                return total_distance / total_time
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"計算速度失敗: {e}")
            return 0.0
    
    def check_violation(self, track_id: int, bbox: Tuple[int, int, int, int], 
                       current_time: float) -> Dict:
        """檢查是否違停"""
        try:
            # 計算停留時間
            dwell_time = current_time - self.first_seen.get(track_id, current_time)
            
            # 計算移動速度
            speed = self.calculate_speed(track_id)
            
            # 檢查是否在 ROI 內
            in_roi = self.roi_manager.is_bbox_in_roi(bbox)
            
            # 違停判定邏輯
            is_violation = False
            violation_reason = []
            
            # 1. 停留時間檢查
            if dwell_time >= self.dwell_threshold:
                violation_reason.append(f"停留時間過長 ({dwell_time:.1f}s >= {self.dwell_threshold}s)")
                is_violation = True
            
            # 2. ROI 位置檢查
            if in_roi:
                violation_reason.append("位於禁停區內")
                is_violation = True
            
            # 3. 速度檢查（如果停留時間較長且速度很慢）
            if dwell_time >= 10.0 and speed <= self.speed_threshold:
                violation_reason.append(f"車輛靜止 (速度: {speed:.2f} pixel/s)")
                is_violation = True
            
            # 記錄違停資訊
            violation_info = {
                'track_id': track_id,
                'dwell_time': dwell_time,
                'speed': speed,
                'in_roi': in_roi,
                'is_violation': is_violation,
                'violation_reason': violation_reason,
                'bbox': bbox,
                'timestamp': current_time,
                'severity': self._calculate_violation_severity(dwell_time, in_roi, speed)
            }
            
            self.violation_records[track_id] = violation_info
            
            logger.info(f"違停檢查 - track_id: {track_id}, "
                       f"停留時間: {dwell_time:.1f}s, "
                       f"速度: {speed:.2f} pixel/s, "
                       f"在ROI內: {in_roi}, "
                       f"違停: {is_violation}")
            
            return violation_info
            
        except Exception as e:
            logger.error(f"違停檢查失敗: {e}")
            return {
                'track_id': track_id,
                'is_violation': False,
                'error': str(e)
            }
    
    def _calculate_violation_severity(self, dwell_time: float, in_roi: bool, 
                                    speed: float) -> str:
        """計算違停嚴重程度"""
        try:
            severity_score = 0
            
            # 停留時間評分
            if dwell_time >= 60.0:  # 1分鐘以上
                severity_score += 3
            elif dwell_time >= 30.0:  # 30秒以上
                severity_score += 2
            elif dwell_time >= 15.0:  # 15秒以上
                severity_score += 1
            
            # ROI 位置評分
            if in_roi:
                severity_score += 2
            
            # 速度評分
            if speed <= 0.05:  # 幾乎靜止
                severity_score += 1
            
            # 根據分數判定嚴重程度
            if severity_score >= 5:
                return "嚴重"
            elif severity_score >= 3:
                return "中等"
            elif severity_score >= 1:
                return "輕微"
            else:
                return "無"
                
        except Exception as e:
            logger.error(f"計算違停嚴重程度失敗: {e}")
            return "未知"
    
    def get_violation_summary(self) -> Dict:
        """獲取違停統計摘要"""
        try:
            total_violations = sum(1 for record in self.violation_records.values() 
                                 if record.get('is_violation', False))
            
            severity_counts = {}
            for record in self.violation_records.values():
                if record.get('is_violation', False):
                    severity = record.get('severity', '未知')
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            return {
                'total_violations': total_violations,
                'total_tracks': len(self.violation_records),
                'severity_distribution': severity_counts,
                'active_tracks': len(self.track_history)
            }
            
        except Exception as e:
            logger.error(f"獲取違停統計失敗: {e}")
            return {}
    
    def clear_track(self, track_id: int) -> None:
        """清除指定追蹤的資料"""
        try:
            if track_id in self.track_history:
                del self.track_history[track_id]
            if track_id in self.first_seen:
                del self.first_seen[track_id]
            if track_id in self.violation_records:
                del self.violation_records[track_id]
            
            logger.info(f"已清除追蹤 {track_id} 的資料")
            
        except Exception as e:
            logger.error(f"清除追蹤資料失敗: {e}")
    
    def reset(self) -> None:
        """重置所有資料"""
        try:
            self.track_history.clear()
            self.first_seen.clear()
            self.violation_records.clear()
            logger.info("違停偵測器已重置")
            
        except Exception as e:
            logger.error(f"重置違停偵測器失敗: {e}")
    
    def set_roi_points(self, points: List[Tuple[int, int]]) -> bool:
        """設定 ROI 點"""
        return self.roi_manager.set_roi_points(points)
    
    def get_roi_points(self) -> List[Tuple[int, int]]:
        """獲取 ROI 點"""
        return self.roi_manager.get_roi_points()
    
    def clear_roi(self) -> bool:
        """清除 ROI"""
        return self.roi_manager.clear_roi()
