import os

import cv2
import numpy as np


AnnotationPath = ""

color_map = np.array([
    255,   # 背景 0
    0,     # 视杯 1
    128    # 视盘 2
])

for root,dirs,files in os.walk('./data/REFUGE-Training400/mask'):
    for item in files:
        color_img = cv2.imread('./data/REFUGE-Training400/mask/'+item)

        gray_img = np.full(shape=(color_img.shape[0],color_img.shape[1]),fill_value=-1,dtype=np.uint8)
        for i in range(color_map.shape[0]):
            index = np.where(np.all(color_img == color_map[i],axis=-1))
            gray_img[index] = i

            save_path = os.path.join('./data/REFUGE-Training400/Mask-Training400/',item)
            cv2.imwrite(save_path,gray_img)

