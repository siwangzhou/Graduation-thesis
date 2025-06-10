import math
import os
import time
from torch.utils.checkpoint import checkpoint
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_get

from ProgressiveSR_CARN.ops.CARN_general import LR_SR_x4_general
from ProgressiveSR_CARN.DataLoader_ImageSR.data_v1v2 import Test
from torchvision.utils import save_image
from ProgressiveSR_CARN.utils import *


def psnr_get(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)

kwards = {'scale': 2}
net_G = LR_SR_x4_general(kwards=kwards).cuda()

chpoint= torch.load("../experiment_CARN/general/CS_CARN_P2P_x4_best.pt")
net_G.load_state_dict(chpoint['model_state_dict'])
test_epoch = chpoint['epoch']
net_G.eval()
for param in net_G.parameters():
    param.requires_grad = False

# chpoint1 = torch.load("../experiment_CARN/v2/CS_CARN_P2P_x4_best.pt")
# v2_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint1['model_state_dict'].items() if
#                             k.startswith('layer1.')}
# v2_upsample_state_dict = {k.replace('layer3.', ''): v for k, v in chpoint1['model_state_dict'].items() if
#                           k.startswith('layer3.')}
# v2_carn_state_dict = {k.replace('layer4.', ''): v for k, v in chpoint1['model_state_dict'].items() if
#                           k.startswith('layer4.')}
# net_G.v2_downsample.load_state_dict(v2_downsample_state_dict)
# net_G.v2_upsample.load_state_dict(v2_upsample_state_dict)
# net_G.SRModel.load_state_dict(v2_carn_state_dict)
#
# chpoint2 = torch.load("../experiment_CARN/v4/CS_CARN_P2P_x4_best.pt")
# v4_conv1_state_dict = {k.replace('conv1.', ''): v for k, v in chpoint2['model_state_dict'].items() if
#                        k.startswith('conv1.')}
# v4_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint2['model_state_dict'].items() if
#                             k.startswith('layer1.')}
# v4_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint2['model_state_dict'].items() if
#                           k.startswith('layer2.')}
# net_G.conv1.load_state_dict(v4_conv1_state_dict)
# net_G.v4_downsample.load_state_dict(v4_downsample_state_dict)
# net_G.v4_upsample.load_state_dict(v4_upsample_state_dict)
#
# chpoint3 = torch.load("../experiment_CARN/v7/CS_CARN_P2P_x4_best.pt")
# v7_conv2_state_dict = {k.replace('conv2.', ''): v for k, v in chpoint3['model_state_dict'].items() if
#                        k.startswith('conv2.')}
# v7_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint3['model_state_dict'].items() if
#                             k.startswith('layer1.')}
# v7_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint3['model_state_dict'].items() if
#                           k.startswith('layer2.')}
# net_G.conv2.load_state_dict(v7_conv2_state_dict)
# net_G.v7_downsample.load_state_dict(v7_downsample_state_dict)
# net_G.v7_upsample.load_state_dict(v7_upsample_state_dict)
#
# chpoint4 = torch.load("../experiment_CARN/v9/CS_CARN_P2P_x4_best.pt")
# v9_conv3_state_dict = {k.replace('conv3.', ''): v for k, v in chpoint4['model_state_dict'].items() if
#                        k.startswith('conv3.')}
# v9_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint4['model_state_dict'].items() if
#                             k.startswith('layer1.')}
# v9_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint4['model_state_dict'].items() if
#                           k.startswith('layer2.')}
# net_G.conv3.load_state_dict(v9_conv3_state_dict)
# net_G.v9_downsample.load_state_dict(v9_downsample_state_dict)
# net_G.v9_upsample.load_state_dict(v9_upsample_state_dict)
#
# chpoint5 = torch.load("../experiment_CARN/v11/CS_CARN_P2P_x4_best.pt")
# v11_conv4_state_dict = {k.replace('conv4.', ''): v for k, v in chpoint5['model_state_dict'].items() if
#                         k.startswith('conv4.')}
# v11_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint5['model_state_dict'].items() if
#                              k.startswith('layer1.')}
# v11_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint5['model_state_dict'].items() if
#                            k.startswith('layer2.')}
# net_G.conv4.load_state_dict(v11_conv4_state_dict)
# net_G.v11_downsample.load_state_dict(v11_downsample_state_dict)
# net_G.v11_upsample.load_state_dict(v11_upsample_state_dict)
#
# chpoint6 = torch.load("../experiment_CARN/v13/CS_CARN_P2P_x4_best.pt")
# v13_conv5_state_dict = {k.replace('conv5.', ''): v for k, v in chpoint6['model_state_dict'].items() if
#                         k.startswith('conv5.')}
# v13_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint6['model_state_dict'].items() if
#                              k.startswith('layer1.')}
# v13_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint6['model_state_dict'].items() if
#                            k.startswith('layer2.')}
# net_G.conv5.load_state_dict(v13_conv5_state_dict)
# net_G.v13_downsample.load_state_dict(v13_downsample_state_dict)
# net_G.v13_upsample.load_state_dict(v13_upsample_state_dict)
#
# chpoint7 = torch.load("../experiment_CARN/v15/CS_CARN_P2P_x4_best.pt")
# v15_conv6_state_dict = {k.replace('conv6.', ''): v for k, v in chpoint7['model_state_dict'].items() if
#                         k.startswith('conv6.')}
# v15_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint7['model_state_dict'].items() if
#                              k.startswith('layer1.')}
# v15_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint7['model_state_dict'].items() if
#                            k.startswith('layer2.')}
# net_G.conv6.load_state_dict(v15_conv6_state_dict)
# net_G.v15_downsample.load_state_dict(v15_downsample_state_dict)
# net_G.v15_upsample.load_state_dict(v15_upsample_state_dict)
#
# chpoint8 = torch.load("../experiment_CARN/v17/CS_CARN_P2P_x4_best.pt")
# v17_conv7_state_dict = {k.replace('conv7.', ''): v for k, v in chpoint8['model_state_dict'].items() if
#                         k.startswith('conv7.')}
# v17_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint8['model_state_dict'].items() if
#                              k.startswith('layer1.')}
# v17_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint8['model_state_dict'].items() if
#                            k.startswith('layer2.')}
# net_G.conv7.load_state_dict(v17_conv7_state_dict)
# net_G.v17_downsample.load_state_dict(v17_downsample_state_dict)
# net_G.v17_upsample.load_state_dict(v17_upsample_state_dict)
#
# net_G.eval()
# for param in net_G.parameters():
#     param.requires_grad = False


# Set5 Set14 Urban100 BSDS100 DIV2K/DIV2K_valid_HR manga109
data_test = Test(["C:/DATASETS/Image-SR/DIV2K/DIV2K_valid_HR"])
test_data = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=False, num_workers=0,
                                        pin_memory=True)
flag_metric = True
flag_visual = True

y_psnr_sum_v2 = 0
y_ssim_sum_v2 = 0
y_psnr_sum_v4 = 0
y_ssim_sum_v4 = 0
y_psnr_sum_v7 = 0
y_ssim_sum_v7 = 0
y_psnr_sum_v9 = 0
y_ssim_sum_v9 = 0
y_psnr_sum_v11 = 0
y_ssim_sum_v11 = 0
y_psnr_sum_v13 = 0
y_ssim_sum_v13 = 0
y_psnr_sum_v15 = 0
y_ssim_sum_v15 = 0
y_psnr_sum_v17 = 0
y_ssim_sum_v17 = 0

t = time.time()
count = 0

output_dir = '../results_general/DIV2K/'
out_v2_path = os.path.join(os.path.join(output_dir,'out_v2'),'sr')
out_v4_path = os.path.join(os.path.join(output_dir,'out_v4'),'sr')
out_v7_path = os.path.join(os.path.join(output_dir,'out_v7'),'sr')
out_v9_path = os.path.join(os.path.join(output_dir,'out_v9'),'sr')
out_v11_path = os.path.join(os.path.join(output_dir,'out_v11'),'sr')
out_v13_path = os.path.join(os.path.join(output_dir,'out_v13'),'sr')
out_v15_path = os.path.join(os.path.join(output_dir,'out_v15'),'sr')
out_v17_path = os.path.join(os.path.join(output_dir,'out_v17'),'sr')

mkdirs(out_v2_path)
mkdirs(out_v4_path)
mkdirs(out_v7_path)
mkdirs(out_v9_path)
mkdirs(out_v11_path)
mkdirs(out_v13_path)
mkdirs(out_v15_path)
mkdirs(out_v17_path)

for test_batch, (_, hr_img, target_img, filename) in enumerate(test_data, start=0):
    hr_img = hr_img.cuda()
    target_img = target_img.cuda()

    with torch.no_grad():
        sr_img_list = net_G(hr_img)

    h, w = target_img.shape[2], target_img.shape[3]
    for i in range(0, len(sr_img_list)):
        sr_img_list[i] = sr_img_list[i][:, :, :h, :w]

    target_img = target_img / 2 + 0.5
    for i in range(0, len(sr_img_list)):
        sr_img_list[i] = sr_img_list[i] / 2 + 0.5
        sr_img_list[i] = sr_img_list[i].clamp(0, 1)

    if flag_visual:
        save_image(sr_img_list[0], os.path.join(out_v2_path, filename[0]))
        save_image(sr_img_list[1], os.path.join(out_v4_path, filename[0]))
        save_image(sr_img_list[2], os.path.join(out_v7_path, filename[0]))
        save_image(sr_img_list[3], os.path.join(out_v9_path, filename[0]))
        save_image(sr_img_list[4], os.path.join(out_v11_path, filename[0]))
        save_image(sr_img_list[5], os.path.join(out_v13_path, filename[0]))
        save_image(sr_img_list[6], os.path.join(out_v15_path, filename[0]))
        save_image(sr_img_list[7], os.path.join(out_v17_path, filename[0]))

    target_img = target_img.cpu().detach().numpy()[0]
    for i in range(0, len(sr_img_list)):
        sr_img_list[i] = sr_img_list[i].cpu().detach().numpy()[0]

    y_target_img = 65.481 * target_img[0, :, :] + 128.553 * target_img[1, :, :] + 24.966 * target_img[2, :, :] + 16
    for i in range(0, len(sr_img_list)):
        sr_img_list[i] = 65.481 * sr_img_list[i][0, :, :] + 128.553 * sr_img_list[i][1, :, :] + 24.966 * sr_img_list[i][2, :, :] + 16

    y_psnr_v2 = psnr_get(y_target_img, sr_img_list[0])
    y_ssim_v2 = ssim_get(y_target_img, sr_img_list[0], data_range=255)
    y_psnr_v4 = psnr_get(y_target_img, sr_img_list[1])
    y_ssim_v4 = ssim_get(y_target_img, sr_img_list[1], data_range=255)
    y_psnr_v7 = psnr_get(y_target_img, sr_img_list[2])
    y_ssim_v7 = ssim_get(y_target_img, sr_img_list[2], data_range=255)
    y_psnr_v9 = psnr_get(y_target_img, sr_img_list[3])
    y_ssim_v9 = ssim_get(y_target_img, sr_img_list[3], data_range=255)
    y_psnr_v11 = psnr_get(y_target_img, sr_img_list[4])
    y_ssim_v11 = ssim_get(y_target_img, sr_img_list[4], data_range=255)
    y_psnr_v13 = psnr_get(y_target_img, sr_img_list[5])
    y_ssim_v13 = ssim_get(y_target_img, sr_img_list[5], data_range=255)
    y_psnr_v15 = psnr_get(y_target_img, sr_img_list[6])
    y_ssim_v15 = ssim_get(y_target_img, sr_img_list[6], data_range=255)
    y_psnr_v17 = psnr_get(y_target_img, sr_img_list[7])
    y_ssim_v17 = ssim_get(y_target_img, sr_img_list[7], data_range=255)

    y_psnr_sum_v2 += y_psnr_v2
    y_ssim_sum_v2 += y_ssim_v2
    y_psnr_sum_v4 += y_psnr_v4
    y_ssim_sum_v4 += y_ssim_v4
    y_psnr_sum_v7 += y_psnr_v7
    y_ssim_sum_v7 += y_ssim_v7
    y_psnr_sum_v9 += y_psnr_v9
    y_ssim_sum_v9 += y_ssim_v9
    y_psnr_sum_v11 += y_psnr_v11
    y_ssim_sum_v11 += y_ssim_v11
    y_psnr_sum_v13 += y_psnr_v13
    y_ssim_sum_v13 += y_ssim_v13
    y_psnr_sum_v15 += y_psnr_v15
    y_ssim_sum_v15 += y_ssim_v15
    y_psnr_sum_v17 += y_psnr_v17
    y_ssim_sum_v17 += y_ssim_v17

    count += 1

psnr_list_8 = [y_psnr_sum_v2/count, y_psnr_sum_v4/count, y_psnr_sum_v7/count, y_psnr_sum_v9/count, y_psnr_sum_v11/count, y_psnr_sum_v13/count, y_psnr_sum_v15/count, y_psnr_sum_v17/count]
psnr_list_8_str = ', '.join([f'{psnr:.2f}' for psnr in psnr_list_8])

ssim_list_8 = [y_ssim_sum_v2/count, y_ssim_sum_v4/count, y_ssim_sum_v7/count, y_ssim_sum_v9/count, y_ssim_sum_v11/count, y_ssim_sum_v13/count, y_ssim_sum_v15/count, y_ssim_sum_v17/count]
ssim_list_8_str = ', '.join([f'{ssim:.4f}' for ssim in ssim_list_8])

print("avg Y PSNR:", psnr_list_8_str)
print("avg Y SSIM:", ssim_list_8_str)
print('time:%.3f s' % (time.time() - t))

if flag_metric:
    str_write = 'psnr_list:{0}, ssim_list:{1}'.format(psnr_list_8_str, ssim_list_8_str) + '\n'
    fp = open(os.path.join(output_dir, 'test_log.txt'), 'a+')
    fp.write(str_write)
    fp.close()
