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

    def closeEvent(self, event):
        try:
            stop_global_monitoring()
            self.log("記憶體監控已停止")
        except Exception as e:
            self.log(f"停止記憶體監控失敗: {e}")
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
