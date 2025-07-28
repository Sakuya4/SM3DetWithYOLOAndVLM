""" user import lib """
import cv2
from ultralytics import YOLO
import fiftyone as fo
import fiftyone.utils.huggingface as fouh
import numpy as np
import os 
import matplotlib.pyplot as plt
""" user import lib done"""

""" dataset and model define """
model = YOLO("yolo11n.pt")
if "JAN_bdd100k" in fo.list_datasets():
    fo.delete_dataset("JAN_bdd100k")
imgDir = r"C:\Users\user\Desktop\比賽\影像辨識+VLM偵測紅線違停\VisionProcess\bdd100k\bdd100k\images\100k\val" # change dir for yourself
dataset = fo.Dataset.from_dir(
    dataset_dir=imgDir,
    dataset_type=fo.types.ImageDirectory,
    name="JAN_bdd100k"
)
session = fo.launch_app(dataset)
""" dataset and model define done"""

""" image preprocess """
def caseComparePreprocess(img):
    clearImg=cv2.GaussianBlur(img,(1,1),0)
    badCaseImg=cv2.GaussianBlur(img,(51,51),0)
    return clearImg, badCaseImg

def SMT3DetUseImage(prev_img, next_img):
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

    if prev_gray.shape != next_gray.shape:
        next_gray = cv2.resize(next_gray, (prev_gray.shape[1], prev_gray.shape[0]))
    
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.1,
        levels=1,
        winsize=9,
        iterations=3,
        poly_n=5,
        poly_sigma=0.9,
        flags=0
    )
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(prev_img)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(np.sqrt(magnitude), None, 0, 255, cv2.NORM_MINMAX)
    flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_img

""" image preprocess done """


""" model inference """
for i, sample in enumerate(dataset):
    if i >= 5:
        break

    imagePath = sample.filepath
    img = cv2.imread(imagePath)
    
    if img is not None:
        clearImg, badCaseImg = caseComparePreprocess(img)
        flowImg = SMT3DetUseImage(clearImg, badCaseImg)
        clearResult = model(clearImg)
        clearAnnotated = clearResult[0].plot()

        badCaseResult = model(badCaseImg)
        badCaseAnnotated = badCaseResult[0].plot()

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(clearAnnotated, cv2.COLOR_BGR2RGB))
        plt.title("YOLOv11 clear case")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(badCaseAnnotated, cv2.COLOR_BGR2RGB))
        plt.title("YOLOv11 bad case")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(flowImg, cv2.COLOR_BGR2RGB))
        plt.title("YOLOv11 flowImg")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        input("press enter")
    else:
        print(f"fail to get image: {imagePath}")
print("detect done")

""" model inference done"""


