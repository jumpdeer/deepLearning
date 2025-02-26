import numpy as np
import os
import cv2

# 对于分割数据集中的annotation是彩色图像，将其转为0，1，2等标签

def color2gray(img_path, color_map, save_dir):
    # 读取图片
    color_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    gray_img = np.zeros(shape=(color_img.shape[0], color_img.shape[1]), dtype=np.uint8)
    for i in range(color_map.shape[0]):
        index = np.where(np.all(color_img == color_map[i], axis=-1))
        gray_img[index] = i

        # 保存图片
        dir, name = os.path.split(img_path)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_dir = os.path.join(save_dir, name)
        cv2.imwrite(save_dir, gray_img)


if __name__ == '__main__':


    cmap = np.array(
        [
            (0, 0, 0),  # 背景 Background     0
            (128, 0, 0),  # 帽子 Hat            1
            (255, 0, 0),  # 头发 Hair           2
            (0, 85, 0),  # 手套 Glove          3
            (170, 0, 51),  # 眼镜 Sunglasses     4
            (255, 85, 0),  # 上衣 Upper-clothes  5
            (0, 0, 85),  # 裙子 Skirt          6
            (0, 119, 221),  # 外套 Coat           7
            (85, 85, 0),  # 袜子 Socks          8
            (0, 85, 85),  # 裤子 Pants          9
            (85, 51, 0),  # 脖子 Neck           10
            (52, 86, 128),  # 围巾 Scarf          11
            (0, 128, 0),  # 裙裤 Dress          12
            (0, 0, 255),  # 脸  Face            13
            (51, 170, 221),  # 左臂 LeftArm        14
            (0, 255, 255),  # 右臂 RightArm       15
            (85, 255, 170),  # 左腿 Leftleg        16
            (170, 255, 85),  # 右腿 Rightleg       17
            (255, 255, 0),  # 左鞋 Leftshoe       18
            (255, 170, 0),  # 右鞋 Rightshoe      19
        ]
    )

    # 文件路径
    img_dir = ''
    save_dir = ''
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    for img in os.listdir(img_dir):
        if not img.endswith((".png", ".jpg")):
            continue
        img_path = os.path.join(img_dir, img)
        color2gray(img_path, cmap, save_dir)