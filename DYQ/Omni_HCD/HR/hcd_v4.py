import random
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import structural_similarity as ssim_get

from Omni_HCD.utils import AverageMeter
from ProgressiveSR_Omni.ops.OmniSR import LR_SR_x4_v4_quant
from ProgressiveSR_Omni.DataLoader_ImageSR.data_v3v4 import Test
from Omni_HCD.utils_hcd import *
from torchvision.utils import save_image
from einops import rearrange

# Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

psnrs_hr = AverageMeter()
ssims_hr = AverageMeter()
base_psnrs = AverageMeter()
base_ssims = AverageMeter()


def optimize_hr(logger, net, y, gt, lr_v2, Ny, epsilon, alpha, output_dir):
    delta_y = torch.zeros_like(y, requires_grad=True)
    optimizer_y = optim.Adam([delta_y], lr=alpha)
    mse_loss = nn.MSELoss()

    gt_w, gt_h = gt.shape[2], gt.shape[3]
    best_x = gt
    best_y = gt
    best_sr = gt
    best_pnsr = 0
    best_ssim = 0
    baseline = [0, 0]
    # best_delta_y = delta_y

    for i in range(Ny):
        optimizer_y.zero_grad()
        y_prime = y + delta_y

        LR = net.conv1(lr_v2)
        LR_diff = net.layer1(y_prime)
        LR_diff = net.v4_quant(LR_diff)

        LR_cat = torch.cat((LR, LR_diff), dim=1)
        new_LR = rearrange(LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=9, a2=9, c=3)

        LR_expand = net.layer2(new_LR)
        SR = net.layer3(LR_expand)

        x_prime, sr_prime = new_LR, SR

        total_loss = mse_loss(sr_prime[:, :, :gt_w, :gt_h], gt)  # 将SR裁剪成gt大小计算loss
        total_loss.backward()

        with torch.no_grad():
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
                best_y = y_prime
                best_x = x_prime
                best_sr = sr_prime
                # best_delta_y = delta_y
            if i == 0:
                baseline[0] = y_psnr
                baseline[1] = y_ssim
                base_psnrs.update(baseline[0], 1)
                base_ssims.update(baseline[1], 1)
            # logger.info(f"[HR] Iteration {i + 1}/{Ny}, PSNR: {y_psnr:.2f}, SSIM: {y_ssim:.4f}")

        with torch.no_grad():
            grad_norm = torch.norm(delta_y.grad, p=2)
            if grad_norm != 0:
                delta_y.data = torch.clamp(delta_y - alpha * delta_y.grad / grad_norm, -epsilon, epsilon)
            else:
                delta_y.data = torch.clamp(delta_y, -epsilon, epsilon)

    # 保存最优的psnr对应的图像和gt
    # best_delta_y = best_delta_y[:, :, :gt_w, :gt_h]
    best_sr = best_sr[:, :, :gt_w, :gt_h]
    best_y = best_y[:, :, :gt_w, :gt_h]
    best_x = best_x
    # torch.save(best_delta_y, output_dir + f"/best_delta_y.pt")
    torch.save(best_y, output_dir + f"/best_y.pt")
    torch.save(best_x, output_dir + f"/best_x.pt")

    # best_delta_y = best_delta_y / 0.6 + 0.5
    best_sr = best_sr / 2 + 0.5
    best_y = best_y / 2 + 0.5
    best_x = best_x / 2 + 0.5

    # best_delta_y = best_delta_y.clamp(0, 1)
    best_sr = best_sr.clamp(0, 1)
    best_y = best_y.clamp(0, 1)
    best_x = best_x.clamp(0, 1)

    # save_image(best_delta_y, output_dir+f"/best_delta_y.png")
    save_image(best_sr, output_dir+f"/best_sr.png")
    # save_image(best_y, output_dir+f"/best_y.png")
    save_image(best_x, output_dir+f"/best_x.png")
    # logger.info(f"[HR] Iteration\tbaseline_[psnr, ssim]: [{baseline[0]:.2f}, {baseline[1]:.4f}]\t"
    #             f"best_[psnr, ssim]: [{best_pnsr:.2f}, {best_ssim:.4f}]")
    psnrs_hr.update(best_pnsr, 1)
    ssims_hr.update(best_ssim, 1)

    return y


if __name__ == '__main__':
    kwards = {'upsampling': 2,
          'res_num': 5,
          'block_num': 1,
          'bias': True,
          'block_script_name': 'OSA_O',
          'block_class_name': 'OSA_Block',
          'window_size': 8,
          'pe': True,
          'ffn_bias': True, }
    P2Pnet = LR_SR_x4_v4_quant(kwards=kwards).cuda()
    chpoint = torch.load("../../ProgressiveSR_Omni/experiment_Omni/v4/CS_Omni_P2P_x4_best.pt")
    P2Pnet.load_state_dict(chpoint['model_state_dict'])
    test_epoch = chpoint['epoch']
    P2Pnet.eval()
    for param in P2Pnet.parameters():
        param.requires_grad = False

    # Set5 Set14 Urban100 BSDS100 DIV2K/DIV2K_valid_HR manga109
    data_name = "manga109/HR"
    data_test = Test([f"D:/DATASETS/Image-SR/{data_name}"])
    test_data = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=False, num_workers=0,
                                            pin_memory=True)

    output_dir = f"./results/hcd_v4_base_hcd_v2/{data_name}/"
    logger = setup_logging(os.path.join(output_dir, 'log.txt'))

    for test_batch, (_, hr_img, target_img, filename) in enumerate(test_data, start=0):

        path = os.path.join(output_dir,filename[0][:-4])
        os.makedirs(os.path.join(output_dir,filename[0][:-4]), exist_ok=True)

        hr_img = hr_img.cuda()
        target_img = target_img.cuda()
        _, lr_v2 = load_image(f"./results/hcd_v2/{data_name}/{filename[0][:-4]}/best_x.png")
        # _, lr_v2 = load_image(f"../../ProgressiveSR_Omni/results/{data_name}/out_v2/lr/{filename[0]}")

        y = optimize_hr(logger, P2Pnet, hr_img, target_img, lr_v2,15,0.3, 20 / 255, path)

    logger.info(f"End test, base_result: [{base_psnrs.avg:.2f} / {base_ssims.avg:.4f}] \t best_result: [{psnrs_hr.avg:.2f} / {ssims_hr.avg:.4f}]\t")