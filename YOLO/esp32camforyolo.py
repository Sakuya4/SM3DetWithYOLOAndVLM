# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg
import cv2 as cv
import numpy as np
import time
from urllib.request import Request, urlopen

url="http://192.168.0.111:81/stream"
# url = "http://10.34.233.158:81/stream"
CAMERA_BUFFRER_SIZE = 16384  

print("準備建立連線...")
req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    stream = urlopen(req)
    print("已成功連線到 ESP32!")
except Exception as e:
    print("連線失敗:", e)
    exit(1)

bts = b''

# 載入 YOLO
print("載入 YOLO 模型...")
net = cv.dnn.readNetFromDarknet("YOLO/yolov3.cfg", "YOLO/yolov3.weights")
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
with open("YOLO/coco.names", 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
print("YOLO 模型載入完成！")

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    outLayers = net.getUnconnectedOutLayers()
    # 兼容 1D/2D array
    if len(outLayers.shape) == 2:
        return [layersNames[i[0] - 1] for i in outLayers]
    else:
        return [layersNames[i - 1] for i in outLayers]

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    confThreshold = 0.5
    nmsThreshold = 0.4
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for idx in indices:
        i = int(idx)  # 直接轉成 int，無論 1D/2D 都可
        box = boxes[i]
        left, top, width, height = box
        label = f"{classes[classIds[i]]}:{confidences[i]:.2f}"
        cv.rectangle(frame, (left, top), (left + width, top + height), (255, 178, 50), 3)
        cv.putText(frame, label, (left, top-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

cv.namedWindow("YOLO", cv.WINDOW_NORMAL)

# FPS 計算變數
fps_counter = 0
fps_start_time = time.time()
fps = 0

print("進入 while 迴圈...")
while True:
    try:
        bts += stream.read(CAMERA_BUFFRER_SIZE)
    except Exception as e:
        print("讀取資料失敗:", e)
        break
    
    jpghead = bts.find(b'\xff\xd8')
    jpgend = bts.find(b'\xff\xd9')
    
    if jpghead > -1 and jpgend > -1:
        jpg = bts[jpghead:jpgend+2]
        bts = bts[jpgend+2:]
        try:
            img = cv.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv.IMREAD_COLOR)
            if img is None or img.size == 0:
                continue
        except Exception as e:
            continue
        
        # 縮小影像以提高處理速度
        height, width = img.shape[:2]
        if width > 640:  # 如果影像太寬，進行縮放
            scale = 640 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv.resize(img, (new_width, new_height))
        
        # YOLO 偵測
        blob = cv.dnn.blobFromImage(img, 1/255, (320, 320), [0,0,0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))
        postprocess(img, outs)
        
        # 計算並顯示 FPS
        fps_counter += 1
        if fps_counter % 30 == 0:  # 每30幀更新一次FPS
            current_time = time.time()
            fps = 30 / (current_time - fps_start_time)
            fps_start_time = current_time
        
        # 在影像上顯示 FPS
        cv.putText(img, f"FPS: {fps:.1f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv.imshow("YOLO", img)
    
    if cv.waitKey(1) == 27:  # 按 ESC 離開
        print("離開 while 迴圈")
        break

cv.destroyAllWindows()
print("程式結束")