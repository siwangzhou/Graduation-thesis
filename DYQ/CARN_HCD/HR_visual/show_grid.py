import cv2
import numpy as np
import os

# 加载图像
image_path = '../hcd_visual_prog_v2_hcd30/img024.png'
out_path = '../hcd_visual_prog_v2_hcd30/img024_grid.png'

# 检查文件是否存在
if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found.")
    exit(1)

# 读取图像
image = cv2.imread(image_path)

# 检查图像是否成功加载
if image is None:
    print(f"Error: Failed to load image from '{image_path}'.")
    exit(1)

# 获取图像的高度和宽度
height, width, _ = image.shape

# 设置网格的大小
grid_size = 32

# 创建一个副本以绘制网格
image_with_grid = image.copy()

# 绘制垂直线
for x in range(0, width, grid_size):
    cv2.line(image_with_grid, (x, 0), (x, height), (0, 0, 255), 1)

# 绘制水平线
for y in range(0, height, grid_size):
    cv2.line(image_with_grid, (0, y), (width, y), (0, 0, 255), 1)

# 保存带网格的图像
# cv2.imwrite(out_path, image_with_grid)
# print(f"Grid image saved to '{out_path}'.")

# 如果需要显示图像，可以使用非 GUI 方法（如 Matplotlib）
try:
    import matplotlib.pyplot as plt

    # 使用 Matplotlib 显示图像
    plt.imshow(cv2.cvtColor(image_with_grid, cv2.COLOR_BGR2RGB))
    plt.title('Image with Grid')
    plt.axis('off')  # 关闭坐标轴
    plt.show()
except ImportError:
    print("Matplotlib is not installed. Skipping image display.")