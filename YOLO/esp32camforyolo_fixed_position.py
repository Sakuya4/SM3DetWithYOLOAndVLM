# ESP32 專用修復版本的 YOLO 物件偵測程式
import cv2 as cv
import numpy as np
import time
from urllib.request import Request, urlopen

url="http://192.168.0.111:81/stream"
CAMERA_BUFFRER_SIZE = 16384

print("--- connection ---")
req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    stream = urlopen(req)
    print("--- connection success ---")
except Exception as e:
    print("--- connection failed ---", e)
    exit(1)

bts = b''

# 載入 YOLO
print("--- load YOLO ---")
net = cv.dnn.readNetFromDarknet("YOLO/yolov3.cfg", "YOLO/yolov3.weights")
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
with open("YOLO/coco.names", 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
print("--- YOLO loaded ---")

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    outLayers = net.getUnconnectedOutLayers()
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
        i = int(idx)
        box = boxes[i]
        left, top, width, height = box
        label = f"{classes[classIds[i]]}:{confidences[i]:.2f}"
        cv.rectangle(frame, (left, top), (left + width, top + height), (255, 178, 50), 3)
        cv.putText(frame, label, (left, top-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

cv.namedWindow("ESP32 YOLO", cv.WINDOW_NORMAL)

# 變數
frame_count = 0
start_time = time.time()
fps = 0
data_count = 0
last_frame_time = time.time()
consecutive_failures = 0

print("--- enter ESP32 processing loop ---")
print("--- press ESC to exit ---")

while True:
    try:
        data = stream.read(CAMERA_BUFFRER_SIZE)
        if not data:
            print("--- stream data end ---")
            break
        
        data_count += 1
        bts += data
        
        if len(bts) > 200000:
            last_start = bts.rfind(b'\xff\xd8')
            if last_start > 0:
                bts = bts[last_start:]
            else:
                bts = b''
        
        jpghead = bts.find(b'\xff\xd8')
        if jpghead >= 0:
            jpgend = bts.find(b'\xff\xd9', jpghead)
            if jpgend > jpghead:
                jpg = bts[jpghead:jpgend+2]
                bts = bts[jpgend+2:]
                
                if len(jpg) > 1000:
                    try:
                        img = cv.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv.IMREAD_COLOR)
                        if img is not None and img.size > 0:
                            frame_count += 1
                            current_time = time.time()
                            consecutive_failures = 0  # 重置失敗計數
                            
                            if frame_count % 30 == 0:
                                fps = 30 / (current_time - start_time)
                                start_time = current_time
                            
                            height, width = img.shape[:2]
                            if width > 320:
                                scale = 320 / width
                                new_width = int(width * scale)
                                new_height = int(height * scale)
                                img = cv.resize(img, (new_width, new_height))
                            
                            # YOLO 偵測 
                            blob = cv.dnn.blobFromImage(img, 1/255, (224, 224), [0,0,0], 1, crop=False)
                            net.setInput(blob)
                            outs = net.forward(getOutputsNames(net))
                            postprocess(img, outs)
                            
                            cv.putText(img, f"FPS: {fps:.1f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            cv.putText(img, f"Frame: {frame_count}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            cv.putText(img, f"Size: {img.shape[1]}x{img.shape[0]}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            cv.putText(img, f"Buffer: {len(bts)}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            cv.putText(img, f"Data: {data_count}", (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            
                            cv.imshow("ESP32 YOLO", img)
                            last_frame_time = current_time
                            
                            if frame_count % 10 == 0:
                                print(f"--- process {frame_count} frames, FPS: {fps:.1f}, buffer size: {len(bts)} bytes ---")
                            
                        else:
                            consecutive_failures += 1
                            print(f"--- image decode failed (consecutive failures: {consecutive_failures}) ---")
                            
                    except Exception as e:
                        consecutive_failures += 1
                        print(f"--- image processing error: {e} (consecutive failures: {consecutive_failures}) ---")
                        continue
                else:
                    consecutive_failures += 1
                    print(f"--- JPEG too small: {len(jpg)} bytes (consecutive failures: {consecutive_failures}) ---")
        
        if consecutive_failures > 50:
            print("--- too many consecutive failures, clear buffer ---")
            bts = b''
            consecutive_failures = 0
                
    except Exception as e:
        print(f"--- stream read error: {e} ---")
        break
    
    if cv.waitKey(1) == 27:
        print("--- exit processing loop ---")
        break

cv.destroyAllWindows()
print(f"--- processing end - total processed {frame_count} frames, received {data_count} data packets ---")
print("---done---") 