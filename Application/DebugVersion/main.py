import sys
import os
import time
from datetime import datetime
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QTextEdit, QMessageBox, QProgressBar, QCheckBox, QSizePolicy
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QThread, Signal
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from model import BLIP2VLM
from webcam_stream import WebcamStreamProcessor
from memory_monitor import (
    start_global_monitoring, 
    stop_global_monitoring, 
    print_memory_report,
    get_memory_info,
    check_memory_leak,
    clear_memory
)


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)



def detect_red_lines(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 120])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 120])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    return mask


class DetectionWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(list, str)

    def __init__(self, source_path, yolo_weights, vlm_enabled, debug_mask, detailed_analysis):
        super().__init__()
        self.source_path = source_path
        self.yolo_weights = yolo_weights
        self.vlm_enabled = vlm_enabled
        self.debug_mask = debug_mask
        self.detailed_analysis = detailed_analysis

    def bbox_hits_redline(self, bbox, red_mask, band_px=30, min_ratio=0.005):
        x1, y1, x2, y2 = map(int, bbox)
        y2 = min(y2, red_mask.shape[0]-1)
        y_top = max(y2 - band_px, 0)
        strip = red_mask[y_top:y2+1, x1:x2+1]
        if strip.size == 0:
            return False, 0.0
        red_pixels = cv2.countNonZero(strip)
        ratio = red_pixels / strip.size
        return ratio >= min_ratio, ratio


    def run(self):
        try:
            yolo = YOLO(self.yolo_weights)
            vlm = BLIP2VLM() if self.vlm_enabled else None

            img = cv2.imread(str(self.source_path))
            if img is None:
                self.log.emit("[ERROR] 無法讀取圖片")
                self.finished.emit([], "")
                return
            
            blurred = cv2.GaussianBlur(img, (0, 0), 3)
            img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
            self.log.emit("[INFO] 已套用去模糊處理")

            self.log.emit("[INFO] 偵測紅線...")
            mask = detect_red_lines(img)

            if self.debug_mask:
                mask_path = OUTPUT_DIR / f"red_mask_{int(time.time())}.png"
                cv2.imwrite(str(mask_path), mask)
                self.log.emit(f"[DEBUG] 紅線遮罩已儲存: {mask_path}")

            self.log.emit("[INFO] 執行 YOLO 偵測...")
            results = yolo.predict(source=np.array(img), imgsz=640, conf=0.25, verbose=False)

            detections = []
            labels = {}
            for r in results:
                labels = getattr(r, 'names', labels)
                if hasattr(r, 'boxes'):
                    for box in r.boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        detections.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls))

            flagged = []
            out_img = img.copy()
            for idx, det in enumerate(detections):
                x1, y1, x2, y2, conf, cls = det
                cls_name = labels.get(cls, str(cls)).lower()
                crop = img[int(y1):int(y2), int(x1):int(x2)]

                # 判斷違規
                overlapped, ratio = self.bbox_hits_redline(
                    (x1, y1, x2, y2), mask, band_px=30, min_ratio=0.005
                )

                vlm_score = None
                is_violation = overlapped and conf > 0.3

                self.log.emit(f"[DEBUG] 車輛={cls_name}, conf={conf:.2f}, redline_ratio={ratio:.3f}, overlapped={overlapped}")

                if self.vlm_enabled and crop.size:
                    try:
                        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        
                        if self.detailed_analysis:
                            detailed_result = vlm.detailed_analysis(pil_crop)
                            caption = vlm.image_captioning(pil_crop)
                            vqa_question = "How many vehicles are visible in this image?"
                            vqa_answer = vlm.vqa_question(pil_crop, vqa_question)
                            
                            self.log.emit(f"[DEBUG] BLIP-2 詳細分析: {detailed_result['full_response']}")
                            self.log.emit(f"[DEBUG] Image Captioning: {caption}")
                            self.log.emit(f"[DEBUG] VQA ({vqa_question}): {vqa_answer}")
                            
                            if detailed_result['is_illegal']:
                                vlm_score = 0.8
                            else:
                                vlm_score = 0.2
                        else:
                            texts = ["The vehicle is illegally parked on a red line.", "The vehicle is parked legally."]
                            scores = vlm.score_image_with_texts(pil_crop, texts)
                            vlm_score = scores[texts[0]]
                        
                        self.log.emit(f"[DEBUG] BLIP-2違規分數={vlm_score:.2f}")
                        if is_violation:
                            is_violation = vlm_score > 0.3
                    except Exception as e:
                        self.log.emit(f"[WARNING] VLM 分析失敗: {e}")
                        vlm_score = None

                color = (0, 0, 255) if is_violation else (0, 255, 0)
                cv2.rectangle(out_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                if is_violation:
                    case_path = OUTPUT_DIR / f"case_{idx}_{int(time.time())}.jpg"
                    cv2.imwrite(str(case_path), crop)
                    flagged.append({'cls': cls_name, 'conf': conf, 'vlm_score': vlm_score})

                self.progress.emit(int((idx + 1) / max(1, len(detections)) * 100))

            out_path = OUTPUT_DIR / f"out_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(out_path), out_img)
            self.finished.emit(flagged, str(out_path))

        except Exception as e:
            self.log.emit(f"[ERROR] {e}")
            self.finished.emit([], "")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("紅線違停辨識 (YOLO + BLIP-2)")
        self.resize(1100, 720)

        self.source_path = None
        self.yolo_weights = "yolov8n.pt"
        self.vlm_enabled = True
        self.debug_mask = True
        self.detailed_analysis = False
        
        # 網路攝影機相關
        self.webcam_processor = None
        self.webcam_running = False
        self.webcam_frame = None
        self.show_detection_boxes = True  # 是否顯示偵測框

        self.image_label = QLabel("(No image)")
        self.image_label.setAlignment(Qt.AlignCenter)

        btn_load_image = QPushButton("上傳照片")
        btn_load_image.clicked.connect(self.on_load_image)

        btn_run = QPushButton("開始辨識")
        btn_run.clicked.connect(self.on_run_detection)

        btn_test_blip = QPushButton("測試 BLIP-2 功能")
        btn_test_blip.clicked.connect(self.on_test_blip)

        btn_memory_monitor = QPushButton("記憶體監控")
        btn_memory_monitor.clicked.connect(self.on_memory_monitor)

        btn_clear_memory = QPushButton("清理記憶體")
        btn_clear_memory.clicked.connect(self.on_clear_memory)

        btn_start_webcam = QPushButton("啟動網路攝影機")
        btn_start_webcam.clicked.connect(self.on_start_webcam)
        
        btn_stop_webcam = QPushButton("停止網路攝影機")
        btn_stop_webcam.clicked.connect(self.on_stop_webcam)
        
        btn_webcam_settings = QPushButton("網路攝影機設定")
        btn_webcam_settings.clicked.connect(self.on_webcam_settings)
        
        btn_test_yolo = QPushButton("測試 YOLO 模型")
        btn_test_yolo.clicked.connect(self.on_test_yolo)
        
        btn_toggle_display = QPushButton("切換顯示模式")
        btn_toggle_display.clicked.connect(self.on_toggle_display)

        self.detailed_checkbox = QCheckBox("啟用詳細分析 (BLIP-2 兩階段分析)")
        self.detailed_checkbox.setChecked(self.detailed_analysis)
        self.detailed_checkbox.toggled.connect(self.on_detailed_toggled)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        self.progress = QProgressBar()

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.image_label)

        right_layout = QVBoxLayout()
        right_layout.addWidget(btn_load_image)
        right_layout.addWidget(btn_run)
        right_layout.addWidget(btn_test_blip)
        right_layout.addWidget(btn_memory_monitor)
        right_layout.addWidget(btn_clear_memory)
        
        # 網路攝影機按鈕
        right_layout.addWidget(btn_start_webcam)
        right_layout.addWidget(btn_stop_webcam)
        right_layout.addWidget(btn_webcam_settings)
        right_layout.addWidget(btn_test_yolo)
        right_layout.addWidget(btn_toggle_display)
        
        right_layout.addWidget(self.detailed_checkbox)
        right_layout.addWidget(self.progress)
        right_layout.addWidget(self.log_text)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 2)

        self.setLayout(main_layout)


    def log(self, msg):
        ts = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{ts}] {msg}")

    def on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇圖片", filter="Images (*.png *.jpg *.jpeg)")
        if path:
            self.source_path = path
            self.show_image(path)
            self.log(f"載入圖片: {path}")

    def show_image(self, path):
        img = QPixmap(path)
        self.image_label.setPixmap(img.scaled(800, 600, Qt.KeepAspectRatio))

    def on_detailed_toggled(self, checked):
        self.detailed_analysis = checked
        if checked:
            self.log("已啟用詳細分析模式 (BLIP-2 兩階段分析)")
        else:
            self.log("已停用詳細分析模式")

    def on_run_detection(self):
        if not self.source_path:
            QMessageBox.warning(self, "提醒", "請先上傳圖片")
            return
        self.log("開始辨識...")
        self.progress.setValue(0)
        self.worker = DetectionWorker(
            self.source_path, 
            self.yolo_weights, 
            self.vlm_enabled, 
            self.debug_mask,
            self.detailed_analysis
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.log)
        self.worker.finished.connect(self.on_detection_finished)
        self.worker.start()

    def on_detection_finished(self, flagged_list, out_path):
        if out_path:
            self.show_image(out_path)
        if flagged_list:
            self.log(f"完成：發現 {len(flagged_list)} 起違規")
            if self.detailed_analysis:
                for i, case in enumerate(flagged_list):
                    self.log(f"違規 {i+1}: 車輛類型={case['cls']}, 置信度={case['conf']:.2f}, VLM分數={case['vlm_score']:.2f}")
        else:
            self.log("完成：未發現違規")
            
    def on_test_blip(self):
        if not self.source_path:
            QMessageBox.warning(self, "提醒", "請先上傳圖片")
            return
        
        try:
            self.log("開始測試 BLIP-2 功能...")
            
            img = Image.open(self.source_path).convert("RGB")
            
            vlm = BLIP2VLM()
            
            self.log("測試 Image Captioning...")
            caption = vlm.image_captioning(img)
            self.log(f"Image Captioning: {caption}")
            
            self.log("測試 VQA...")
            vqa_question = "How many vehicles are visible in this image?"
            vqa_answer = vlm.vqa_question(img, vqa_question)
            self.log(f"VQA ({vqa_question}): {vqa_answer}")
            
            self.log("測試詳細分析...")
            detailed_result = vlm.detailed_analysis(img)
            self.log(f"詳細分析結果: {detailed_result['full_response']}")
            
            self.log("BLIP-2 功能測試完成！")
            
        except Exception as e:
            self.log(f"測試失敗: {e}")
            QMessageBox.critical(self, "錯誤", f"測試失敗: {e}")

    def on_memory_monitor(self):
        try:
            leak_info = check_memory_leak()
            
            stats = get_memory_info()
            
            from memory_monitor import get_global_monitor
            report = get_global_monitor().generate_report()
            
            QMessageBox.information(self, "記憶體監控報告", report)
            
            self.log("=== 記憶體監控報告 ===")
            self.log(f"當前記憶體: {stats.get('current_memory', 0):.2f} GB")
            self.log(f"系統使用率: {stats.get('system_memory_percent', 0):.1f}%")
            self.log(f"GPU記憶體: {stats.get('gpu_memory_allocated', 0):.2f} GB")
            
            if leak_info['leak_detected']:
                self.log(f"檢測到記憶體洩漏! 增長率: {leak_info['growth_rate']:.2f}x")
            else:
                self.log("記憶體使用正常")
                
        except Exception as e:
            self.log(f"記憶體監控失敗: {e}")

    def on_clear_memory(self):
        try:
            clear_memory()
            self.log("記憶體清理完成")
            QMessageBox.information(self, "成功", "記憶體清理完成")
        except Exception as e:
            self.log(f"記憶體清理失敗: {e}")
            QMessageBox.critical(self, "錯誤", f"記憶體清理失敗: {e}")

    def on_start_webcam(self):
        """啟動網路攝影機"""
        try:
            if self.webcam_running:
                QMessageBox.information(self, "提醒", "網路攝影機已在運行中")
                return
            
            self.log("正在啟動網路攝影機...")
            
            # 創建網路攝影機處理器
            self.webcam_processor = WebcamStreamProcessor(
                camera_index=0,  # 預設使用第一個攝影機
                yolo_weights=self.yolo_weights
            )
            
            # 設定自動上傳回調
            self.webcam_processor.set_auto_upload_callback(self.on_webcam_auto_upload)
            
            # 啟動串流
            if self.webcam_processor.start_stream():
                self.webcam_running = True
                self.log("✅ 網路攝影機啟動成功！")
                self.log("正在進行即時車輛偵測...")
                
                # 開始定時更新畫面
                self._start_webcam_display()
                
            else:
                self.log("❌ 網路攝影機啟動失敗")
                QMessageBox.critical(self, "錯誤", "無法啟動網路攝影機，請檢查設備連接")
                
        except Exception as e:
            self.log(f"啟動網路攝影機時發生錯誤: {e}")
            QMessageBox.critical(self, "錯誤", f"啟動失敗: {e}")

    def on_stop_webcam(self):
        """停止網路攝影機"""
        try:
            if not self.webcam_running:
                QMessageBox.information(self, "提醒", "網路攝影機未在運行")
                return
            
            self.log("正在停止網路攝影機...")
            
            if self.webcam_processor:
                self.webcam_processor.stop_stream()
                self.webcam_processor = None
            
            self.webcam_running = False
            self.log("✅ 網路攝影機已停止")
            
            # 恢復預設畫面
            self.image_label.setText("(No image)")
            
        except Exception as e:
            self.log(f"停止網路攝影機時發生錯誤: {e}")

    def on_webcam_settings(self):
        """網路攝影機設定"""
        try:
            if not self.webcam_processor:
                QMessageBox.warning(self, "提醒", "請先啟動網路攝影機")
                return
            
            # 使用 webcam_stream 模組的功能
            self._show_webcam_settings_dialog()
            
        except Exception as e:
            self.log(f"網路攝影機設定失敗: {e}")

    def _show_webcam_settings_dialog(self):
        """顯示網路攝影機設定對話框"""
        from PySide6.QtWidgets import QInputDialog, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QDoubleSpinBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("網路攝影機設定")
        dialog.resize(450, 350)
        
        layout = QVBoxLayout()
        
        # 攝影機資訊
        camera_info = self.webcam_processor.get_camera_info()
        info_label = QLabel(f"攝影機狀態: {camera_info.get('status', '未知')}")
        if 'resolution' in camera_info:
            info_label.setText(f"攝影機狀態: {camera_info['status']}\n解析度: {camera_info['resolution']}\nFPS: {camera_info['fps']}")
        layout.addWidget(info_label)
        
        # 自動上傳時間設定
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("自動上傳持續時間 (秒):"))
        duration_spin = QDoubleSpinBox()
        duration_spin.setRange(1.0, 10.0)
        duration_spin.setValue(self.webcam_processor.auto_upload_duration)
        duration_spin.setSingleStep(0.5)
        duration_layout.addWidget(duration_spin)
        layout.addLayout(duration_layout)
        
        # 偵測閾值設定
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("偵測閾值:"))
        threshold_spin = QDoubleSpinBox()
        threshold_spin.setRange(0.1, 1.0)
        threshold_spin.setValue(self.webcam_processor.detection_threshold)
        threshold_spin.setSingleStep(0.05)
        threshold_spin.setDecimals(2)
        threshold_layout.addWidget(threshold_spin)
        layout.addLayout(threshold_layout)
        
        # 按鈕
        button_layout = QHBoxLayout()
        
        btn_apply = QPushButton("套用設定")
        btn_apply.clicked.connect(lambda: self._apply_webcam_settings(duration_spin.value(), threshold_spin.value(), dialog))
        
        btn_stats = QPushButton("顯示統計")
        btn_stats.clicked.connect(self._show_webcam_stats)
        
        btn_reset = QPushButton("重置計數器")
        btn_reset.clicked.connect(self._reset_webcam_counters)
        
        btn_capture = QPushButton("手動拍攝")
        btn_capture.clicked.connect(self._manual_capture)
        
        button_layout.addWidget(btn_apply)
        button_layout.addWidget(btn_stats)
        button_layout.addWidget(btn_reset)
        button_layout.addWidget(btn_capture)
        
        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        
        dialog.exec()

    def _apply_webcam_settings(self, duration: float, threshold: float, dialog):
        """套用網路攝影機設定"""
        try:
            self.webcam_processor.set_auto_upload_duration(duration)
            self.webcam_processor.set_detection_threshold(threshold)
            self.log(f"設定已更新 - 自動上傳時間: {duration:.1f}秒, 偵測閾值: {threshold:.2f}")
            dialog.accept()
        except Exception as e:
            self.log(f"套用設定失敗: {e}")

    def _show_webcam_stats(self):
        """顯示網路攝影機統計"""
        try:
            summary = self.webcam_processor.get_vehicle_detection_summary()
            QMessageBox.information(self, "網路攝影機統計", summary)
        except Exception as e:
            self.log(f"顯示統計失敗: {e}")

    def _reset_webcam_counters(self):
        """重置網路攝影機計數器"""
        try:
            self.webcam_processor.reset_detection_counters()
            self.log("已重置車輛偵測計數器")
        except Exception as e:
            self.log(f"重置計數器失敗: {e}")

    def _manual_capture(self):
        """手動拍攝照片"""
        try:
            if not self.webcam_running:
                QMessageBox.warning(self, "提醒", "請先啟動網路攝影機")
                return
            
            filepath = self.webcam_processor.capture_single_frame()
            if filepath:
                self.log(f"手動拍攝完成: {filepath}")
                QMessageBox.information(self, "成功", f"照片已儲存: {filepath}")
            else:
                QMessageBox.warning(self, "失敗", "無法拍攝照片")
        except Exception as e:
            self.log(f"手動拍攝失敗: {e}")

    def on_test_yolo(self):
        """測試 YOLO 模型"""
        try:
            if not self.webcam_processor:
                # 創建一個臨時的處理器來測試
                self.webcam_processor = WebcamStreamProcessor(
                    camera_index=0,
                    yolo_weights=self.yolo_weights
                )
            
            # 獲取 YOLO 狀態
            status = self.webcam_processor.get_yolo_status()
            
            status_text = f"""
=== YOLO 模型狀態 ===
模型已載入: {'是' if status['model_loaded'] else '否'}
模型路徑: {status['model_path']}
偵測閾值: {status['detection_threshold']:.2f}
車輛類別: {', '.join(status['vehicle_classes'])}

正在測試模型...
"""
            
            # 顯示狀態
            QMessageBox.information(self, "YOLO 狀態", status_text)
            
            # 執行測試
            if self.webcam_processor.test_yolo_model():
                self.log("YOLO 模型測試成功！")
                QMessageBox.information(self, "測試結果", "YOLO 模型測試成功！")
            else:
                self.log("YOLO 模型測試失敗！")
                QMessageBox.critical(self, "測試結果", "YOLO 模型測試失敗！")
                
        except Exception as e:
            self.log(f"測試 YOLO 模型時發生錯誤: {e}")
            QMessageBox.critical(self, "錯誤", f"測試失敗: {e}")

    def on_toggle_display(self):
        """切換顯示模式（偵測框/原始畫面）"""
        if not self.webcam_running:
            QMessageBox.warning(self, "提醒", "請先啟動網路攝影機")
            return
        
        self.show_detection_boxes = not self.show_detection_boxes
        
        if self.show_detection_boxes:
            self.log("已切換到偵測框顯示模式")
            QMessageBox.information(self, "顯示模式", "已切換到偵測框顯示模式")
        else:
            self.log("已切換到原始畫面顯示模式")
            QMessageBox.information(self, "顯示模式", "已切換到原始畫面顯示模式")

    def on_webcam_auto_upload(self, upload_info):
        try:
            self.log(f"自動偵測到車輛: {upload_info['vehicle_type']}")
            self.log(f"   置信度: {upload_info['confidence']:.2f}")
            self.log(f"   持續時間: {upload_info['duration']:.1f}秒")
            self.log(f"   影像已儲存: {upload_info['filepath']}")
            
            self.source_path = upload_info['filepath']
            self.show_image(upload_info['filepath'])
        
            reply = QMessageBox.question(
                self, "提醒", 
                f"偵測到車輛 {upload_info['vehicle_type']}，是否要進行違規分析？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.log("開始自動分析...")
                self.on_run_detection()
                
        except Exception as e:
            self.log(f"處理自動上傳時發生錯誤: {e}")

    def _start_webcam_display(self):
        if not self.webcam_running:
            return
        
        try:
            if self.show_detection_boxes:
                frame = self.webcam_processor.get_frame()
            else:
                frame = self.webcam_processor.get_raw_frame()
            
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                
                from PySide6.QtGui import QImage
                qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                pixmap = QPixmap.fromImage(qt_image)
                self.image_label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio))
                
                status_info = self.webcam_processor.get_status_info()
                display_mode = "偵測框" if self.show_detection_boxes else "原始畫面"
                self.setWindowTitle(f"紅線違停辨識 - {status_info} [{display_mode}]")
            
            from PySide6.QtCore import QTimer
            QTimer.singleShot(100, self._start_webcam_display)
            
        except Exception as e:
            self.log(f"更新網路攝影機畫面時發生錯誤: {e}")

    def closeEvent(self, event):
        try:
            if self.webcam_running and self.webcam_processor:
                self.webcam_processor.stop_stream()
            
            stop_global_monitoring()
            self.log("記憶體監控已停止")
            
        except Exception as e:
            self.log(f"關閉程式時發生錯誤: {e}")
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    try:
        start_global_monitoring(interval=2.0)
        print("[DEBUG] 記憶體監控已啟動")
    except Exception as e:
        print(f"[DEBUG] 啟動記憶體監控失敗: {e}")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
