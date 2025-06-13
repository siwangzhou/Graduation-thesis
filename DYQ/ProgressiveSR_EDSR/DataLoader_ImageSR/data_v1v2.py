import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from typing import Sequence
from numpy import random
import numpy as np
import os


class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class MyDataset(Dataset):
    def __init__(self, alist=None):
        super(MyDataset, self).__init__()
        if alist is None:
            alist = ["C:/DATASETS/Image-SR/DIV2K/DIV2K_train_HR", "C:/DATASETS/Image-SR/Flickr2K/Flickr2K"]       #
        self.hrimgs = []
        for path in alist:
            train_list = sorted(os.listdir(path))
            for name in train_list:
                img_path = path+"/"+name
                self.hrimgs.append(img_path)

        self.sizex = len(self.hrimgs)
        self.patch_size = 64
        self.down_scale = 8 / 32

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        patch_size = self.patch_size
        down_scale = self.down_scale
        index_ = index % self.sizex
        hr_path = self.hrimgs[index_]
        hr_img = Image.open(hr_path).convert('RGB')
        hr_img = TF.to_tensor(hr_img)

        hh, ww = hr_img.shape[1], hr_img.shape[2]
        rr = random.randint(0, hh - patch_size)
        cc = random.randint(0, ww - patch_size)

        # Crop patch
        hr_img = hr_img[:, rr:rr + patch_size, cc:cc + patch_size]

        lr_img = F.interpolate(
            hr_img.unsqueeze(0),  # 添加批次维度
            scale_factor=down_scale,
            mode='bicubic',
            align_corners=False
        ).squeeze(0)  # 移除批次维度

        aug = random.randint(0, 8)
        if aug == 1:
            hr_img = hr_img.flip(1)
            lr_img = lr_img.flip(1)
        elif aug == 2:
            hr_img = hr_img.flip(2)
            lr_img = lr_img.flip(2)
        elif aug == 3:
            hr_img = torch.rot90(hr_img, dims=(1, 2))
            lr_img = torch.rot90(lr_img, dims=(1, 2))
        elif aug == 4:
            hr_img = torch.rot90(hr_img, dims=(1, 2), k=2)
            lr_img = torch.rot90(lr_img, dims=(1, 2), k=2)
        elif aug == 5:
            hr_img = torch.rot90(hr_img, dims=(1, 2), k=3)
            lr_img = torch.rot90(lr_img, dims=(1, 2), k=3)
        elif aug == 6:
            hr_img = torch.rot90(hr_img.flip(1), dims=(1, 2))
            lr_img = torch.rot90(lr_img.flip(1), dims=(1, 2))
        elif aug == 7:
            hr_img = torch.rot90(hr_img.flip(2), dims=(1, 2))
            lr_img = torch.rot90(lr_img.flip(2), dims=(1, 2))

        hr_img = (hr_img - 0.5) * 2.0
        lr_img = (lr_img - 0.5) * 2.0

        return lr_img, hr_img


class Test(Dataset):
    def __init__(self, alist=None):
        super(Test, self).__init__()
        if alist is None:
            alist = ["C:/DATASETS/Image-SR/Set5/HR"]
        self.hrimgs = []
        for path in alist:
            test_list = sorted(os.listdir(path))
            for name in test_list:
                img_path = path+"/"+name
                self.hrimgs.append(img_path)

        self.sizex = len(self.hrimgs)
        self.mul = 32
        self.down_scale = 8 / 32

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        mul = self.mul
        down_scale = self.down_scale
        index_ = index % self.sizex

        hr_path = self.hrimgs[index_]
        hr_img = Image.open(hr_path).convert('RGB')
        target_img = hr_img

        w, h = hr_img.size
        H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
        padh = H - h if h % mul != 0 else 0
        padw = W - w if w % mul != 0 else 0
        hr_img = TF.pad(hr_img, (0, 0, padw, padh), padding_mode='reflect')

        hr_img = TF.to_tensor(hr_img)
        target_img = TF.to_tensor(target_img)
        filename = os.path.split(hr_path)[-1]

        lr_img = F.interpolate(
            hr_img.unsqueeze(0),  # 添加批次维度
            scale_factor=down_scale,
            mode='bicubic',
            align_corners=False
        ).squeeze(0)  # 移除批次维度

        hr_img = (hr_img - 0.5) * 2.0
        lr_img = (lr_img - 0.5) * 2.0
        target_img = (target_img - 0.5) * 2.0

        return lr_img, hr_img, target_img, filename


if __name__ == '__main__':
    a = MyDataset()
    n = a.__len__()
    a.__getitem__(1)