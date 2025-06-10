import cv2


def draw_rectangle(image_path, top_left, bottom_right, color='red', thickness=2, output_path='output.png'):
    """
    在图像上绘制矩形框。

    参数:
    - image_path: 输入图像的路径 (str)。
    - top_left: 矩形框左上角坐标 (tuple)，例如 (x1, y1)。
    - bottom_right: 矩形框右下角坐标 (tuple)，例如 (x2, y2)。
    - color: 矩形框颜色 (str)，可选 'red' 或 'green'。
    - thickness: 线条粗细 (int)。
    - output_path: 输出图像保存路径 (str)。
    """
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 使用 IMREAD_UNCHANGED 保留透明度通道（如果有）
    if image is None:
        raise ValueError("无法读取图像，请检查路径是否正确！")

    # 定义颜色
    colors = {
        'red': (0, 0, 255),  # OpenCV中颜色格式为BGR
        'green': (0, 255, 0)
    }
    if color.lower() not in colors:
        raise ValueError("颜色必须是 'red' 或 'green'")

    # 绘制矩形框
    cv2.rectangle(image, top_left, bottom_right, colors[color.lower()], thickness)

    # 保存结果图像
    cv2.imwrite(output_path, image)
    print(f"图像已保存至: {output_path}")


# 示例用法
if __name__ == "__main__":
    # 输入图像路径
    input_image = "../hcd_visual_prog_v2_hcd30/img024.png"
    # 输出图像路径
    output_image = "../hcd_visual_prog_v2_hcd30/img024_rectangle.png"

    # baby [138, 160, 168, 190]
    # barbara [480, 286, 510, 316]
    # img024 [545, 390, 575, 420]
    top_left = (545, 390)
    bottom_right = (575, 420)
    # 矩形框颜色和线宽
    color = "red"  # 或 "green"
    thickness = 1
    # 调用函数
    draw_rectangle(input_image, top_left, bottom_right, color, thickness, output_image)