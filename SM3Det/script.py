import os
import subprocess

img_dir = 'data/archive/bdd100k/images/10k/val/'
config = 'configs/SM3Det/bdd100k_SM3Det.py'
checkpoint = 'checkpoint/iter_33468.pth'
out_dir = 'results/'

os.makedirs(out_dir, exist_ok=True)

count=0
for img_name in os.listdir(img_dir):
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(img_dir, img_name)
        out_path = os.path.join(out_dir, img_name)
        cmd = [
            'python', 'tools/image_demo.py',
            img_path, config, checkpoint,
            '--out-file', out_path,
            '--score-thr', '0.3'
        ]
        print(' '.join(cmd))
        subprocess.run(cmd)
        count+=1
        if count>=20:
            break
