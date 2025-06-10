import math
import os
import time
from torch.utils.checkpoint import checkpoint
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_get
from ProgressiveSR_CARN_16x16_8x8.ops.CARN_16x16 import LR_SR_x4_v2_quant
from ProgressiveSR_CARN_16x16_8x8.DataLoader_ImageSR.data_v1v2 import Test
from torchvision.utils import save_image
from ProgressiveSR_CARN_16x16_8x8.utils import *


def psnr_get(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


kwards = {'scale': 2}
P2Pnet = LR_SR_x4_v2_quant(kwards=kwards).cuda()
chpoint= torch.load("../experiment_CARN_16x16/v2/CS_CARN_P2P_x4_best.pt")
P2Pnet.load_state_dict(chpoint['model_state_dict'])
test_epoch = chpoint['epoch']
P2Pnet.eval()
for param in P2Pnet.parameters():
    param.requires_grad = False

# Set5 Set14 Urban100 BSDS100 DIV2K/DIV2K_valid_HR manga109
data_test = Test(["C:/DATASETS/Image-SR/DIV2K/DIV2K_valid_HR"])
test_data = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=False, num_workers=0,
                                        pin_memory=True)
flag_metric = True
flag_visual = True

y_psnr_sum = 0
y_ssim_sum = 0
t = time.time()
count = 0

output_dir = '../results_16x16/DIV2K/out_v2/'
hr_path = os.path.join(output_dir, 'hr')
sr_path = os.path.join(output_dir, 'sr')
lr_path = os.path.join(output_dir, 'lr')
expand_path = os.path.join(output_dir, 'lr_expand')
mkdirs(hr_path)
mkdirs(sr_path)
mkdirs(lr_path)
mkdirs(expand_path)

for test_batch, (_, hr_img, target_img, filename) in enumerate(test_data, start=0):
    hr_img = hr_img.cuda()
    target_img = target_img.cuda()

    with torch.no_grad():
        lr_img, lr_expand, sr_img = P2Pnet(hr_img)

    h, w = target_img.shape[2], target_img.shape[3]
    sr_img = sr_img[:, :, :h, :w]

    i1 = target_img / 2 + 0.5
    i2 = sr_img / 2 + 0.5
    i3 = lr_img / 2 + 0.5
    i4 = lr_expand / 2 + 0.5

    i1 = i1.clamp(0, 1)
    i2 = i2.clamp(0, 1)
    i3 = i3.clamp(0, 1)
    i4 = i4.clamp(0, 1)

    if flag_visual:
        save_image(i1, os.path.join(hr_path, filename[0]))
        save_image(i2, os.path.join(sr_path, filename[0]))
        save_image(i3, os.path.join(lr_path, filename[0]))
        save_image(i4, os.path.join(expand_path, filename[0]))

    i1 = i1.cpu().detach().numpy()[0]
    i2 = i2.cpu().detach().numpy()[0]

    y_i1 = 65.481 * i1[0, :, :] + 128.553 * i1[1, :, :] + 24.966 * i1[2, :, :] + 16
    y_i2 = 65.481 * i2[0, :, :] + 128.553 * i2[1, :, :] + 24.966 * i2[2, :, :] + 16

    y_psnr = psnr_get(y_i1, y_i2)
    y_ssim = ssim_get(y_i1, y_i2, data_range=255)

    y_psnr_sum += y_psnr
    y_ssim_sum += y_ssim

    count += 1

print('avg Y PSNR:%.2f' % (y_psnr_sum / count))
print('avg Y SSIM:%.4f' % (y_ssim_sum / count))
print('time:%.3f s' % (time.time() - t))

if flag_metric:
    str_write = 'test_epoch:{0}    avg_Y_PSRN:{1:.2f}    avg_Y_SSIM:{2:.4f}    time={3:.3f} s '.format(test_epoch, (
                y_psnr_sum / count), (y_ssim_sum / count), (time.time() - t)) + '\n'
    fp = open(os.path.join(output_dir, 'test_log.txt'), 'a+')
    fp.write(str_write)
    fp.close()
