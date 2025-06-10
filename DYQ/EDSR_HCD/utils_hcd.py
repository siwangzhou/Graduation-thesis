import math
import os
import numpy as np
import logging
from PIL import Image
import torchvision.transforms.functional as TF


def setup_logging(log_path):
    """
    设置日志记录，日志同时保存到文件和控制台
    """
    # 创建日志文件所在目录
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # 获取指定名称的 logger 对象
    logger = logging.getLogger("MyLogger")
    logger.setLevel(logging.INFO)

    # 如果已存在 handler，先清空
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建文件 handler
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 创建控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def load_image(path):
    # 根据路径读取图片，并转换为Tensor
    mul = 32

    hr_img = Image.open(path).convert('RGB')
    target_img = hr_img

    w, h = hr_img.size
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
    padh = H - h if h % mul != 0 else 0
    padw = W - w if w % mul != 0 else 0
    hr_img = TF.pad(hr_img, (0, 0, padw, padh), padding_mode='reflect')

    hr_img = TF.to_tensor(hr_img)
    target_img = TF.to_tensor(target_img)

    hr_img = (hr_img - 0.5) * 2.0
    target_img = (target_img - 0.5) * 2.0

    hr_img = hr_img.unsqueeze(0)
    target_img = target_img.unsqueeze(0)

    return hr_img.cuda(), target_img.cuda()


def psnr_get(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)