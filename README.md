# solveillegalparkingbyVLM
### 2025高通台灣AI黑客松比賽題目
YOLO做車輛檢測，經過NAFNet的模糊處理，最後提供給VLM去判斷是否違規

如果畫面無法辨識(Bad case)，會由SM3Det介入協助影像處理

本專案目的希望可以將SM3Det的無人機角度，改變成一般監視器之角度，以應用在交通違規處理


---
### 專案介紹

1. 攝影機及其輔助裝置優先採取邊緣運算的裝置設備，例如：ESP32, raspberry pi 等性能較低之設備

2. 架構上採混合雲，透過ESP32-CAM收集到的畫面，傳到雲端，經系統處理後，產生摘要，最後交由USER確定是否開立罰單

<img width="1146" height="643" alt="image" src="https://github.com/user-attachments/assets/9ed41312-32ab-4fab-a5f3-b95b308a0c0f" />
<img width="1146" height="644" alt="image" src="https://github.com/user-attachments/assets/9024c9d7-5e1e-459b-a2aa-6ae1dea65bc9" />

---
### Demo
<img width="1108" height="319" alt="image" src="https://github.com/user-attachments/assets/186939f9-38ed-4c36-b833-38ea244dd503" />

---
### 測試用GUI程式使用

1. 建虛擬環境
   
`python -m venv venv   //or use conda`

2. 啟動虛擬環境

PowerShell:

`.\venv\Scripts\Activate.ps1`

cmd:

`.\venv\Scripts\activate.bat`

3. 安裝需要的套件

`pip install -r requirements.txt`
