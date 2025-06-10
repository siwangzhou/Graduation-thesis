import os
from PIL import Image

hr_path = "E:/DATASETS/Image-SR/hcd_visual/img024.png"
sr_path = "../bicubic_down_up"

# 读取图像
img_hr = Image.open(hr_path).convert('RGB')

W, H = img_hr.size
img_lr = img_hr.resize((int(W / 4), int(H / 4)), Image.BICUBIC)
w, h = img_lr.size
img_sr = img_lr.resize((int(w * 4), int(h * 4)), Image.BICUBIC)

sr_path_full = os.path.join(sr_path, "img024.png")
img_sr.save(sr_path_full)

