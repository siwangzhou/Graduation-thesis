import torch
import numpy as np
import math
from torch.utils.checkpoint import checkpoint
from ProgressiveSR_CARN_16x16_8x8.ops.CARN_8x8 import LR_SR_x4_v11_quant
from ProgressiveSR_CARN_16x16_8x8.DataLoader_ImageSR.data_v10v11 import MyDataset, Test
from ProgressiveSR_CARN_16x16_8x8.utils import *
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

    net_G = LR_SR_x4_v11_quant(kwards=kwards).cuda()

    chpoint = torch.load("../experiment_CARN_8x8/v2/CS_CARN_P2P_x4_best.pt")
    v2_downsample_state_dict = {k.replace('layer1.', ''): v for k, v in chpoint['model_state_dict'].items() if
                                k.startswith('layer1.')}
    net_G.v2_downsample.load_state_dict(v2_downsample_state_dict)
    for param in net_G.v2_downsample.parameters():
        param.requires_grad = False

    data_train = MyDataset()
    data_test = Test()
    train_data = torch.utils.data.DataLoader(dataset=data_train, batch_size=16, shuffle=True, num_workers=8,
                                             pin_memory=True)
    test_data = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=True, num_workers=0,
                                            pin_memory=True)

    model_dir = "../experiment_CARN_8x8/v11"
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
        for batch, (lr_bic, hr_img) in enumerate(train_data, start=0):
            lr_bic = lr_bic.cuda()
            hr_img = hr_img.cuda()
            expand_img = F.interpolate(hr_img, scale_factor=1/2, mode='bicubic', align_corners=False)

            optimizer_G.zero_grad()

            lr_img, lr_expand, sr_img = net_G(hr_img)

            loss1 = lossmse(lr_img, lr_bic)
            loss2 = lossmse(lr_expand, expand_img)
            loss3 = lossL1(sr_img, hr_img)

            loss_G = loss1 + loss2 + loss3

            loss_G.backward()
            optimizer_G.step()
            sum_loss += loss_G.item()

        # 判断最高psnr并保存
        if epoch % 1 == 0:
            net_G.eval()
            sum_y_psnr = 0
            avg_y_psnr = 0
            count = 0
            for test_batch, (_, hr_img, target_img, filename) in enumerate(test_data, start=0):
                hr_img = hr_img.cuda()
                target_img = target_img.cuda()

                with torch.no_grad():
                    _, _, sr_img = net_G(hr_img)

                    h, w = target_img.shape[2], target_img.shape[3]
                    sr_img = sr_img[:, :, :h, :w]

                i1 = target_img.cpu().detach().numpy()[0]
                i2 = sr_img.cpu().detach().numpy()[0]
                i1 = (i1 + 1.0) / 2.0
                i1 = np.clip(i1, 0.0, 1.0)

                i2 = (i2 + 1.0) / 2.0
                i2 = np.clip(i2, 0.0, 1.0)

                i1 = 65.481 * i1[0, :, :] + 128.553 * i1[1, :, :] + 24.966 * i1[2, :, :] + 16
                i2 = 65.481 * i2[0, :, :] + 128.553 * i2[1, :, :] + 24.966 * i2[2, :, :] + 16

                y_psnr = psnr_get(i1, i2)
                sum_y_psnr += y_psnr
                count += 1

            avg_y_psnr = sum_y_psnr / count
            if avg_y_psnr > best_psnr:
                best_psnr = avg_y_psnr
                best_epoch_psnr = epoch
                checkpoint_best = {
                    'epoch': epoch,
                    'model_state_dict': net_G.state_dict(),  # index 是当前最好的模型的索引
                    'optimizer_state_dict': optimizer_G.state_dict(),
                }
                torch.save(checkpoint_best, os.path.join(model_dir, 'CS_CARN_P2P_x4_best.pt'))
            print("[epoch %d PSNR: %.2f --- Best_Epoch %d Best_PSNR %.2f]" % (
            epoch, avg_y_psnr, best_epoch_psnr, best_psnr))

        scheduler.step()
        print("[epoch %d Time: %.4f --- Loss: %.4f --- LearningRate: %.6f]" % (
        epoch, time.time() - start_time, sum_loss, optimizer_G.param_groups[0]['lr']))
        checkpoint_latest = {
            'epoch': epoch,
            'model_state_dict': net_G.state_dict(),
            'optimizer_state_dict': optimizer_G.state_dict(),
        }
        torch.save(checkpoint_latest, os.path.join(model_dir, 'CS_CARN_P2P_x4_latest.pt'))

        str_write = '{0}|{1}---PSNR:{2:.2f}---Best_Epoch:{3:}---Best_PSNR:{4:.2f}---time:{5:.4f}'.format(epoch,
                                                                                                         epochs,
                                                                                                         avg_y_psnr,
                                                                                                         best_epoch_psnr,
                                                                                                         best_psnr,
                                                                                                         time.time() - start_time) + '\n'
        fp = open(os.path.join(model_dir, 'CS_CARN_P2P_x4.txt'), 'a+')
        fp.write(str_write)
        fp.close()


