import cv2
import time
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class WebcamStreamProcessor:
    def __init__(self, camera_index: int = 0, yolo_weights: str = "yolov8n.pt"):
        self.camera_index = camera_index
        self.yolo_weights = yolo_weights
        self.cap = None
        self.is_running = False
        self.detection_thread = None
        self.vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
        self.detection_threshold = 0.3
        self.vehicle_detection_time = {}
        self.auto_upload_duration = 3.0
        self.output_dir = Path("webcam_captures")
        self.output_dir.mkdir(exist_ok=True)
        self.yolo_model = None
        self.last_annotated_frame = None
        self._load_yolo_model()
    
    def _load_yolo_model(self):
        try:
            logger.info(f"正在載入 YOLO 模型: {self.yolo_weights}")
            self.yolo_model = YOLO(self.yolo_weights)
            logger.info(f"成功載入 YOLO 模型: {self.yolo_weights}")
            return True
        except Exception as e:
            logger.error(f"載入 YOLO 模型失敗: {e}")
            self.yolo_model = None
            return False
    
    def start_stream(self) -> bool:
        try:
            if self.yolo_model is None:
                if not self._load_yolo_model():
                    logger.error("無法載入 YOLO 模型，無法啟動串流")
                    return False
            
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"無法開啟網路攝影機 {self.camera_index}")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.is_running = True
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            
            logger.info(f"成功啟動網路攝影機串流 (攝影機 {self.camera_index})")
            logger.info(f"YOLO 就緒，開始車輛偵測")
            return True
            
        except Exception as e:
            logger.error(f"啟動網路攝影機串流失敗: {e}")
            return False
    
    def stop_stream(self):
        self.is_running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("網路攝影機串流已停止")
    
    def _detection_loop(self):
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("無法讀取攝影機畫面")
                    time.sleep(0.1)
                    continue
                
                self._detect_vehicles(frame)
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"偵測循環錯誤: {e}")
                time.sleep(0.5)
    
    def _detect_vehicles(self, frame):
        if self.yolo_model is None:
            logger.warning("YOLO 模型未載入，跳過偵測")
            return
        
        try:
            results = self.yolo_model.predict(
                source=frame, 
                imgsz=640, 
                conf=self.detection_threshold, 
                verbose=False
            )
            
            current_time = time.time()
            detected_vehicles = []
            detection_count = 0
            
            annotated_frame = frame.copy()
            
            for r in results:
                if hasattr(r, 'boxes'):
                    for box in r.boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        cls_name = r.names.get(cls, str(cls)).lower()
                        
                        if detection_count < 3:
                            logger.debug(f"偵測到物件: {cls_name}, 置信度: {conf:.2f}")
                        
                        x1, y1, x2, y2 = map(int, xyxy)
                        
                        if cls_name in self.vehicle_classes:
                            color = (0, 255, 0)
                            thickness = 2
                        else:
                            color = (128, 128, 128)
                            thickness = 1
                        
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                        
                        label = f"{cls_name} {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        if cls_name in self.vehicle_classes:
                            vehicle_id = f"{cls_name}_{x1}_{y1}"
                            
                            if vehicle_id not in self.vehicle_detection_time:
                                self.vehicle_detection_time[vehicle_id] = current_time
                                logger.info(f"偵測到新車輛: {cls_name} (ID: {vehicle_id}, 置信度: {conf:.2f})")
                            
                            detection_duration = current_time - self.vehicle_detection_time[vehicle_id]
                            
                            if detection_duration >= self.auto_upload_duration:
                                self._auto_upload_frame(annotated_frame, vehicle_id, cls_name, conf, detection_duration)
                                self.vehicle_detection_time[vehicle_id] = current_time
                            
                            detected_vehicles.append(vehicle_id)
                        
                        detection_count += 1
            
            current_vehicles = set(detected_vehicles)
            expired_vehicles = [vid for vid in self.vehicle_detection_time.keys() 
                              if vid not in current_vehicles and 
                              current_time - self.vehicle_detection_time[vid] > 5.0]
            
            for vid in expired_vehicles:
                del self.vehicle_detection_time[vid]
                logger.info(f"車輛 {vid} 已離開畫面")
            
            if detection_count > 0 and time.time() % 10 < 0.1:
                logger.info(f"偵測狀態: 總物件 {detection_count}, 活躍車輛 {len(detected_vehicles)}")           
            self.last_annotated_frame = annotated_frame
                
        except Exception as e:
            logger.error(f"車輛偵測錯誤: {e}")
            import traceback
            logger.error(f"詳細錯誤: {traceback.format_exc()}")
    
    def _auto_upload_frame(self, frame, vehicle_id: str, vehicle_type: str, confidence: float, duration: float):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vehicle_{vehicle_type}_{vehicle_id}_{timestamp}.jpg"
            filepath = self.output_dir / filename
            
            cv2.imwrite(str(filepath), frame)
            
            upload_info = {
                'timestamp': timestamp,
                'vehicle_id': vehicle_id,
                'vehicle_type': vehicle_type,
                'confidence': confidence,
                'duration': duration,
                'filepath': str(filepath)
            }
            
            logger.info(f"自動上傳車輛影像: {filename}")
            logger.info(f"車輛資訊: {vehicle_type}, 置信度: {confidence:.2f}, 持續時間: {duration:.1f}秒")
            
            if hasattr(self, 'on_auto_upload'):
                self.on_auto_upload(upload_info)
                
        except Exception as e:
            logger.error(f"自動上傳影像失敗: {e}")
    
    def get_frame(self) -> Optional[np.ndarray]:
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if hasattr(self, 'last_annotated_frame') and self.last_annotated_frame is not None:
                    return self.last_annotated_frame
                return frame
        return None
    
    def get_raw_frame(self) -> Optional[np.ndarray]:
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None
    
    def set_auto_upload_callback(self, callback_func):
        self.on_auto_upload = callback_func
    
    def get_detection_stats(self) -> Dict:
        current_time = time.time()
        active_vehicles = []
        
        for vid, detect_time in self.vehicle_detection_time.items():
            duration = current_time - detect_time
            if duration <= 10.0:
                active_vehicles.append({
                    'id': vid,
                    'duration': duration,
                    'type': vid.split('_')[0] if '_' in vid else 'unknown'
                })
        
        return {
            'active_vehicles': active_vehicles,
            'total_detected': len(self.vehicle_detection_time),
            'auto_upload_duration': self.auto_upload_duration,
            'detection_threshold': self.detection_threshold
        }
    
    def set_auto_upload_duration(self, duration: float):
        self.auto_upload_duration = max(1.0, duration)
        logger.info(f"自動上傳持續時間已設定為: {self.auto_upload_duration}秒")
    
    def set_detection_threshold(self, threshold: float):
        self.detection_threshold = max(0.1, min(1.0, threshold))
        logger.info(f"偵測閾值已設定為: {self.detection_threshold}")

    def get_status_info(self) -> str:
        stats = self.get_detection_stats()
        if stats['active_vehicles']:
            return f"偵測中: {len(stats['active_vehicles'])} 輛車"
        else:
            return "等待車輛..."

    def is_stream_active(self) -> bool:
        return self.is_running and self.cap and self.cap.isOpened()

    def get_camera_info(self) -> Dict:
        if not self.cap or not self.cap.isOpened():
            return {"status": "未連接"}
        
        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            return {
                "status": "已連接",
                "resolution": f"{width}x{height}",
                "fps": fps,
                "camera_index": self.camera_index
            }
        except Exception as e:
            return {"status": f"錯誤: {e}"}

    def capture_single_frame(self) -> Optional[str]:
        if not self.is_stream_active():
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"manual_capture_{timestamp}.jpg"
                filepath = self.output_dir / filename
                
                cv2.imwrite(str(filepath), frame)
                logger.info(f"手動拍攝照片: {filename}")
                return str(filepath)
        except Exception as e:
            logger.error(f"手動拍攝失敗: {e}")
        
        return None

    def get_vehicle_detection_summary(self) -> str:
        stats = self.get_detection_stats()
        
        summary = f"""
=== 網路攝影機偵測摘要 ===
活躍車輛數: {len(stats['active_vehicles'])}
總偵測數: {stats['total_detected']}
自動上傳時間: {stats['auto_upload_duration']:.1f}秒
偵測閾值: {stats['detection_threshold']:.2f}

活躍車輛:
"""
        for vehicle in stats['active_vehicles']:
            summary += f"- {vehicle['type']}: {vehicle['duration']:.1f}秒\n"
        
        if not stats['active_vehicles']:
            summary += "- 無活躍車輛\n"
        
        return summary

    def reset_detection_counters(self):
        self.vehicle_detection_time.clear()
        logger.info("已重置車輛偵測計數器")

    def change_camera(self, new_camera_index: int) -> bool:
        if self.is_running:
            self.stop_stream()
        
        self.camera_index = new_camera_index
        
        if self.is_running:
            return self.start_stream()
        
        return True
    
    def test_yolo_model(self) -> bool:
        try:
            if self.yolo_model is None:
                logger.warning("YOLO 模型未載入，嘗試重新載入...")
                if not self._load_yolo_model():
                    return False
            
            test_image = np.zeros((640, 640, 3), dtype=np.uint8)
            
            results = self.yolo_model.predict(
                source=test_image,
                imgsz=640,
                conf=0.1,
                verbose=False
            )
            
            logger.info("YOLO 模型測試成功")
            return True
            
        except Exception as e:
            logger.error(f"YOLO 模型測試失敗: {e}")
            return False
    
    def get_yolo_status(self) -> Dict:
        return {
            "model_loaded": self.yolo_model is not None,
            "model_path": self.yolo_weights,
            "detection_threshold": self.detection_threshold,
            "vehicle_classes": self.vehicle_classes
        }
