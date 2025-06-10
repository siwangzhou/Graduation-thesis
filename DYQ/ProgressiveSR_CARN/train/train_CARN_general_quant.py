import torch
import numpy as np
import math
from torch.utils.checkpoint import checkpoint
from ProgressiveSR_CARN.ops.CARN_general import LR_SR_x4_general
from ProgressiveSR_CARN.DataLoader_ImageSR.data_v1v2 import MyDataset, Test
from ProgressiveSR_CARN.utils import *
from warmup_scheduler import GradualWarmupScheduler
import torch.nn.functional as F
import time
import random

# Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


def psnr_get(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


if __name__ == '__main__':
    kwards = {'scale': 2}

    net_G = LR_SR_x4_general(kwards=kwards).cuda()

    chpoint1 = torch.load("../experiment_CARN/v2/CS_CARN_P2P_x4_best.pt")
    v2_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint1['model_state_dict'].items() if
                                k.startswith('layer1.')}
    v2_upsample_state_dict = {k.replace('layer3.', ''): v for k, v in chpoint1['model_state_dict'].items() if
                              k.startswith('layer3.')}
    net_G.v2_downsample.load_state_dict(v2_downsample_state_dict)
    net_G.v2_upsample.load_state_dict(v2_upsample_state_dict)

    chpoint2 = torch.load("../experiment_CARN/v4/CS_CARN_P2P_x4_best.pt")
    v4_conv1_state_dict = {k.replace('conv1.', ''): v for k, v in chpoint2['model_state_dict'].items() if
                           k.startswith('conv1.')}
    v4_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint2['model_state_dict'].items() if
                                k.startswith('layer1.')}
    v4_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint2['model_state_dict'].items() if
                              k.startswith('layer2.')}
    net_G.conv1.load_state_dict(v4_conv1_state_dict)
    net_G.v4_downsample.load_state_dict(v4_downsample_state_dict)
    net_G.v4_upsample.load_state_dict(v4_upsample_state_dict)

    chpoint3 = torch.load("../experiment_CARN/v7/CS_CARN_P2P_x4_best.pt")
    v7_conv2_state_dict = {k.replace('conv2.', ''): v for k, v in chpoint3['model_state_dict'].items() if
                           k.startswith('conv2.')}
    v7_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint3['model_state_dict'].items() if
                                k.startswith('layer1.')}
    v7_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint3['model_state_dict'].items() if
                              k.startswith('layer2.')}
    net_G.conv2.load_state_dict(v7_conv2_state_dict)
    net_G.v7_downsample.load_state_dict(v7_downsample_state_dict)
    net_G.v7_upsample.load_state_dict(v7_upsample_state_dict)

    chpoint4 = torch.load("../experiment_CARN/v9/CS_CARN_P2P_x4_best.pt")
    v9_conv3_state_dict = {k.replace('conv3.', ''): v for k, v in chpoint4['model_state_dict'].items() if
                           k.startswith('conv3.')}
    v9_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint4['model_state_dict'].items() if
                                k.startswith('layer1.')}
    v9_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint4['model_state_dict'].items() if
                              k.startswith('layer2.')}
    net_G.conv3.load_state_dict(v9_conv3_state_dict)
    net_G.v9_downsample.load_state_dict(v9_downsample_state_dict)
    net_G.v9_upsample.load_state_dict(v9_upsample_state_dict)

    chpoint5 = torch.load("../experiment_CARN/v11/CS_CARN_P2P_x4_best.pt")
    v11_conv4_state_dict = {k.replace('conv4.', ''): v for k, v in chpoint5['model_state_dict'].items() if
                            k.startswith('conv4.')}
    v11_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint5['model_state_dict'].items() if
                                 k.startswith('layer1.')}
    v11_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint5['model_state_dict'].items() if
                               k.startswith('layer2.')}
    net_G.conv4.load_state_dict(v11_conv4_state_dict)
    net_G.v11_downsample.load_state_dict(v11_downsample_state_dict)
    net_G.v11_upsample.load_state_dict(v11_upsample_state_dict)

    chpoint6 = torch.load("../experiment_CARN/v13/CS_CARN_P2P_x4_best.pt")
    v13_conv5_state_dict = {k.replace('conv5.', ''): v for k, v in chpoint6['model_state_dict'].items() if
                            k.startswith('conv5.')}
    v13_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint6['model_state_dict'].items() if
                                 k.startswith('layer1.')}
    v13_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint6['model_state_dict'].items() if
                               k.startswith('layer2.')}
    net_G.conv5.load_state_dict(v13_conv5_state_dict)
    net_G.v13_downsample.load_state_dict(v13_downsample_state_dict)
    net_G.v13_upsample.load_state_dict(v13_upsample_state_dict)

    chpoint7 = torch.load("../experiment_CARN/v15/CS_CARN_P2P_x4_best.pt")
    v15_conv6_state_dict = {k.replace('conv6.', ''): v for k, v in chpoint7['model_state_dict'].items() if
                            k.startswith('conv6.')}
    v15_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint7['model_state_dict'].items() if
                                 k.startswith('layer1.')}
    v15_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint7['model_state_dict'].items() if
                               k.startswith('layer2.')}
    net_G.conv6.load_state_dict(v15_conv6_state_dict)
    net_G.v15_downsample.load_state_dict(v15_downsample_state_dict)
    net_G.v15_upsample.load_state_dict(v15_upsample_state_dict)

    chpoint8 = torch.load("../experiment_CARN/v17/CS_CARN_P2P_x4_best.pt")
    v17_conv7_state_dict = {k.replace('conv7.', ''): v for k, v in chpoint8['model_state_dict'].items() if
                            k.startswith('conv7.')}
    v17_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint8['model_state_dict'].items() if
                                 k.startswith('layer1.')}
    v17_upsample_state_dict = {k.replace('layer2.', ''): v for k, v in chpoint8['model_state_dict'].items() if
                               k.startswith('layer2.')}
    net_G.conv7.load_state_dict(v17_conv7_state_dict)
    net_G.v17_downsample.load_state_dict(v17_downsample_state_dict)
    net_G.v17_upsample.load_state_dict(v17_upsample_state_dict)

    for param in net_G.parameters():
        param.requires_grad = False
    for param in net_G.SRModel.parameters():
        param.requires_grad = True

    data_train = MyDataset()
    data_test = Test()
    train_data = torch.utils.data.DataLoader(dataset=data_train, batch_size=16, shuffle=True, num_workers=8,
                                             pin_memory=True)
    test_data = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=True, num_workers=0,
                                            pin_memory=True)

    model_dir = "../experiment_CARN/general"
    mkdirs(model_dir)

    start_epochs = 1
    epochs = 1500

    lossmse = torch.nn.MSELoss()
    lossL1 = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-8)
    warmup_epochs = 3
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, epochs - warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer_G, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    best_psnr = 0
    best_epoch_psnr = 0

    for epoch in range(start_epochs, epochs+1):
        start_time = time.time()
        sum_loss = 0

        net_G.train()
        for batch, (_, hr_img) in enumerate(train_data, start=0):
            hr_img = hr_img.cuda()

            optimizer_G.zero_grad()

            sr_img_list = net_G(hr_img)

            loss_v2 = lossL1(sr_img_list[0], hr_img)
            loss_v4 = lossL1(sr_img_list[1], hr_img)
            loss_v7 = lossL1(sr_img_list[2], hr_img)
            loss_v9 = lossL1(sr_img_list[3], hr_img)
            loss_v11 = lossL1(sr_img_list[4], hr_img)
            loss_v13 = lossL1(sr_img_list[5], hr_img)
            loss_v15 = lossL1(sr_img_list[6], hr_img)
            loss_v17 = lossL1(sr_img_list[7], hr_img)

            loss_G = (loss_v2 + loss_v4 + loss_v7 + loss_v9 + loss_v11 + loss_v13 + loss_v15 + loss_v17) / 8

            loss_G.backward()
            optimizer_G.step()
            sum_loss += loss_G.item()

        # 判断最高psnr并保存
        if epoch % 1 == 0:
            net_G.eval()
            psnr2_sum = 0
            psnr4_sum = 0
            psnr7_sum = 0
            psnr9_sum = 0
            psnr11_sum = 0
            psnr13_sum = 0
            psnr15_sum = 0
            psnr17_sum = 0
            psnr_sum = 0
            count = 0
            for test_batch, (_, hr_img, target_img, filename) in enumerate(test_data, start=0):
                hr_img = hr_img.cuda()
                target_img = target_img.cuda()

                with torch.no_grad():
                    sr_img_list = net_G(hr_img)

                h, w = target_img.shape[2], target_img.shape[3]
                for i in range(0, len(sr_img_list)):
                    sr_img_list[i] = sr_img_list[i][:, :, :h, :w]

                i1 = target_img.cpu().detach().numpy()[0]
                i2 = sr_img_list[0].cpu().detach().numpy()[0]
                i4 = sr_img_list[1].cpu().detach().numpy()[0]
                i7 = sr_img_list[2].cpu().detach().numpy()[0]
                i9 = sr_img_list[3].cpu().detach().numpy()[0]
                i11 = sr_img_list[4].cpu().detach().numpy()[0]
                i13 = sr_img_list[5].cpu().detach().numpy()[0]
                i15 = sr_img_list[6].cpu().detach().numpy()[0]
                i17 = sr_img_list[7].cpu().detach().numpy()[0]

                i1 = (i1 + 1.0) / 2.0
                i1 = np.clip(i1, 0.0, 1.0)
                i2 = (i2 + 1.0) / 2.0
                i2 = np.clip(i2, 0.0, 1.0)
                i4 = (i4 + 1.0) / 2.0
                i4 = np.clip(i4, 0.0, 1.0)
                i7 = (i7 + 1.0) / 2.0
                i7 = np.clip(i7, 0.0, 1.0)
                i9 = (i9 + 1.0) / 2.0
                i9 = np.clip(i9, 0.0, 1.0)
                i11 = (i11 + 1.0) / 2.0
                i11 = np.clip(i11, 0.0, 1.0)
                i13 = (i13 + 1.0) / 2.0
                i13 = np.clip(i13, 0.0, 1.0)
                i15 = (i15 + 1.0) / 2.0
                i15 = np.clip(i15, 0.0, 1.0)
                i17 = (i17 + 1.0) / 2.0
                i17 = np.clip(i17, 0.0, 1.0)

                i1 = 65.481 * i1[0, :, :] + 128.553 * i1[1, :, :] + 24.966 * i1[2, :, :] + 16
                i2 = 65.481 * i2[0, :, :] + 128.553 * i2[1, :, :] + 24.966 * i2[2, :, :] + 16
                i4 = 65.481 * i4[0, :, :] + 128.553 * i4[1, :, :] + 24.966 * i4[2, :, :] + 16
                i7 = 65.481 * i7[0, :, :] + 128.553 * i7[1, :, :] + 24.966 * i7[2, :, :] + 16
                i9 = 65.481 * i9[0, :, :] + 128.553 * i9[1, :, :] + 24.966 * i9[2, :, :] + 16
                i11 = 65.481 * i11[0, :, :] + 128.553 * i11[1, :, :] + 24.966 * i11[2, :, :] + 16
                i13 = 65.481 * i13[0, :, :] + 128.553 * i13[1, :, :] + 24.966 * i13[2, :, :] + 16
                i15 = 65.481 * i15[0, :, :] + 128.553 * i15[1, :, :] + 24.966 * i15[2, :, :] + 16
                i17 = 65.481 * i17[0, :, :] + 128.553 * i17[1, :, :] + 24.966 * i17[2, :, :] + 16

                psnr2 = psnr_get(i1, i2)
                psnr4 = psnr_get(i1, i4)
                psnr7 = psnr_get(i1, i7)
                psnr9 = psnr_get(i1, i9)
                psnr11 = psnr_get(i1, i11)
                psnr13 = psnr_get(i1, i13)
                psnr15 = psnr_get(i1, i15)
                psnr17 = psnr_get(i1, i17)

                psnr2_sum += psnr2
                psnr4_sum += psnr4
                psnr7_sum += psnr7
                psnr9_sum += psnr9
                psnr11_sum += psnr11
                psnr13_sum += psnr13
                psnr15_sum += psnr15
                psnr17_sum += psnr17
                psnr_sum += (psnr2 + psnr4 + psnr7 + psnr9 + psnr11 + psnr13 + psnr15 + psnr17) / 8
                count += 1

            avg_y_psnr = psnr_sum / count
            avg_psnr2 = psnr2_sum / count
            avg_psnr4 = psnr4_sum / count
            avg_psnr7 = psnr7_sum / count
            avg_psnr9 = psnr9_sum / count
            avg_psnr11 = psnr11_sum / count
            avg_psnr13 = psnr13_sum / count
            avg_psnr15 = psnr15_sum / count
            avg_psnr17 = psnr17_sum / count
            psnr_list_8 = [avg_psnr2,avg_psnr4,avg_psnr7,avg_psnr9,avg_psnr11,avg_psnr13,avg_psnr15, avg_psnr17]
            psnr_list_8_str = ', '.join([f'{psnr:.3f}' for psnr in psnr_list_8])
            if avg_y_psnr > best_psnr:
                best_psnr = avg_y_psnr
                best_epoch_psnr = epoch
                checkpoint_best = {
                    'epoch': epoch,
                    'model_state_dict': net_G.state_dict(),  # index 是当前最好的模型的索引
                    'optimizer_state_dict': optimizer_G.state_dict(),
                }
                torch.save(checkpoint_best, os.path.join(model_dir, 'CS_CARN_P2P_x4_best.pt'))
            print("[epoch %d PSNR: %.2f --- Best_Epoch %d Best_PSNR %.2f]" % (epoch, avg_y_psnr, best_epoch_psnr, best_psnr))
            print("psnr_list:{}".format(psnr_list_8))

        scheduler.step()
        print("[epoch %d Time: %.4f --- Loss: %.4f --- LearningRate: %.6f]" % (
        epoch, time.time() - start_time, sum_loss, optimizer_G.param_groups[0]['lr']))
        checkpoint_latest = {
            'epoch': epoch,
            'model_state_dict': net_G.state_dict(),
            'optimizer_state_dict': optimizer_G.state_dict(),
        }
        torch.save(checkpoint_latest, os.path.join(model_dir, 'CS_CARN_P2P_x4_latest.pt'))

        str_write = ('{0}|{1}---PSNR:{2:.2f}---Best_Epoch:{3:}---Best_PSNR:{4:.2f}---psnr_list:{5}---time:{6:.4f}'
                     .format(epoch, epochs, avg_y_psnr,best_epoch_psnr,best_psnr,psnr_list_8_str, time.time() - start_time) + '\n')
        fp = open(os.path.join(model_dir, 'CS_CARN_P2P_x4.txt'), 'a+')
        fp.write(str_write)
        fp.close()


