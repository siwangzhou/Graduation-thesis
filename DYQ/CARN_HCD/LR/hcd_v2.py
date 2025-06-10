import random
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import structural_similarity as ssim_get

from CARN_HCD.utils import AverageMeter
from ProgressiveSR_CARN.ops.CARN import LR_SR_x4_v2_quant
from ProgressiveSR_CARN.DataLoader_ImageSR.data_v1v2 import Test
from CARN_HCD.utils_hcd import *
from torchvision.utils import save_image
from einops import rearrange


# Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


psnrs_lr = AverageMeter()
ssims_lr = AverageMeter()
base_psnrs = AverageMeter()
base_ssims = AverageMeter()


def optimize_lr(logger, net, y, gt, Nx, epsilon, alpha, output_dir):
    with torch.no_grad():
        x = net.layer1(y)

    delta_x = torch.zeros_like(x, requires_grad=True)
    optimizer_x = optim.Adam([delta_x], lr=alpha)
    mse_loss = nn.MSELoss()

    gt_w, gt_h = gt.shape[2], gt.shape[3]
    best_sr = gt
    best_x = gt
    best_pnsr = 0
    best_ssim = 0
    baseline = [0, 0]
    best_delta_x = delta_x

    for i in range(Nx):
        optimizer_x.zero_grad()

        x_prime = x + delta_x

        LR_expand = net.layer3(net.layer2(x_prime))
        SR = net.layer4(LR_expand)

        sr_prime = SR

        total_loss = mse_loss(sr_prime[:, :, :gt_w, :gt_h], gt)
        total_loss.backward()

        with torch.no_grad():
            # 计算 PSNR
            i1 = sr_prime[:, :, :gt_w, :gt_h] / 2 + 0.5
            i2 = gt / 2 + 0.5
            i1 = i1.clamp(0, 1)
            i2 = i2.clamp(0, 1)

            i1 = i1.cpu().detach().numpy()[0]
            i2 = i2.cpu().detach().numpy()[0]

            y_i1 = 65.481 * i1[0, :, :] + 128.553 * i1[1, :, :] + 24.966 * i1[2, :, :] + 16
            y_i2 = 65.481 * i2[0, :, :] + 128.553 * i2[1, :, :] + 24.966 * i2[2, :, :] + 16

            y_psnr = psnr_get(y_i1, y_i2)
            y_ssim = ssim_get(y_i1, y_i2, data_range=255)

            if y_psnr > best_pnsr:
                best_pnsr = y_psnr
                best_ssim = y_ssim
                best_sr = sr_prime
                best_x = x_prime
                best_delta_x = delta_x
            if i == 0:
                baseline[0] = y_psnr
                baseline[1] = y_ssim
                base_psnrs.update(baseline[0], 1)
                base_ssims.update(baseline[1], 1)
            logger.info(f"[LR_diff] Iteration {i + 1}/{Nx}, PSNR: {y_psnr:.2f}, SSIM: {y_ssim:.4f}")

        with torch.no_grad():
            grad_norm = torch.norm(delta_x.grad, p=2)
            if grad_norm != 0:
                delta_x.data = torch.clamp(delta_x - alpha * delta_x.grad / grad_norm, -epsilon, epsilon)
            else:
                delta_x.data = torch.clamp(delta_x, -epsilon, epsilon)

    # 保存最优的psnr对应的图像和gt
    best_sr = best_sr[:, :, :gt_w, :gt_h]

    torch.save(best_delta_x, output_dir + f"/best_delta_x.pt")
    torch.save(best_x, output_dir + f"/best_x.pt")

    best_sr = best_sr / 2 + 0.5
    best_x = best_x / 2 + 0.5

    best_sr = best_sr.clamp(0, 1)
    best_x = best_x.clamp(0, 1)

    save_image(best_sr, output_dir+f"/best_sr.png")
    save_image(best_x, output_dir+f"/best_x.png")
    logger.info(f"[HR] Iteration\tbaseline_[psnr, ssim]: [{baseline[0]:.2f}, {baseline[1]:.4f}]\t"
                f"best_[psnr, ssim]: [{best_pnsr:.2f}, {best_ssim:.4f}]")
    psnrs_lr.update(best_pnsr, 1)
    ssims_lr.update(best_ssim, 1)

    return y


if __name__ == '__main__':
    kwards = {'scale': 2}
    P2Pnet = LR_SR_x4_v2_quant(kwards=kwards).cuda()
    chpoint = torch.load("../../ProgressiveSR_CARN/experiment_CARN/v2/CS_CARN_P2P_x4_best.pt")
    P2Pnet.load_state_dict(chpoint['model_state_dict'])
    test_epoch = chpoint['epoch']
    P2Pnet.eval()
    for param in P2Pnet.parameters():
        param.requires_grad = False

    data_test = Test(["D:/DATASETS/Image-SR/Set5/HR"])
    test_data = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=False, num_workers=0,
                                            pin_memory=True)

    for test_batch, (_, hr_img, target_img, filename) in enumerate(test_data, start=0):
        output_dir = f"./results/hcd_v2/Set5/"
        path = os.path.join(output_dir, filename[0][:-4])
        os.makedirs(os.path.join(output_dir, filename[0][:-4]), exist_ok=True)

        hr_img = hr_img.cuda()
        target_img = target_img.cuda()

        logger = setup_logging(os.path.join(output_dir, 'log.txt'))

        y = optimize_lr(logger, P2Pnet, hr_img, target_img, 15, 0.3, 20 / 255, path)

        logger.info(f"End test, base_result: [{base_psnrs.avg:.2f} / {base_ssims.avg:.4f}] \t best_result: [{psnrs_lr.avg:.2f} / {ssims_lr.avg:.4f}]\t")
