from PIL import Image

# GT
# img_dir = "E:/DATASETS/Image-SR/DIV2K/DIV2K_valid_HR/0825.png"
# out_dir = "E:/ProgressiveSR-visual-result/ProgressiveSR_CARN/results_visual_0825/DIV2K/out_v2/hr/0825_patch_b9.png"
# image = Image.open(img_dir)
#
# # 裁剪图像
# p = [1045, 910, 1135, 1000]
# cropped_image = image.crop((p[0], p[1], p[2], p[3]))
#
# # 保存裁剪后的图像
# cropped_image.save(out_dir)


# CARN
# data_dir = "E:/ProgressiveSR-visual-result/ProgressiveSR_CARN/results_visual_0825/DIV2K"
# path_list = ["out_v2", "out_v4", "out_v7", "out_v9", "out_v11", "out_v13", "out_v15", "out_v17"]
#
# for path in path_list:
#     img_dir = f"{data_dir}/{path}/sr/0825.png"
#     out_dir = f"{data_dir}/{path}/sr/0825_patch_b9.png"
#     image = Image.open(img_dir)
#
#     # 裁剪图像
#     p = [1045, 910, 1135, 1000]
#     cropped_image = image.crop((p[0], p[1], p[2], p[3]))
#
#     # 保存裁剪后的图像
#     cropped_image.save(out_dir)

# baby [138, 160, 168, 190]
# barbara [480, 286, 510, 316]
# img024 [545, 390, 575, 420]
# gt bicubic prog prog_hr_hcd
img_dir = "../hcd_visual_bicubic/barbara.png"
out_dir = "../hcd_visual_bicubic/barbara_patch.png"
image = Image.open(img_dir)

# 裁剪图像
p = [480, 286, 510, 316]
cropped_image = image.crop((p[0], p[1], p[2], p[3]))

# 保存裁剪后的图像
cropped_image.save(out_dir)