import numpy as np
import os
import cv2
import time

def color2gray(img_path,color_map,save_dir):
    # 读取图片
    color_img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
    # 计算时间
    t0 = time.time()

    gray_img = np.zeros(shape=(color_img.shape[0],color_img.shape[1]),dtype=np.uint8)
    for i in range(color_map.shape[0]):
        index = np.where(np.all(color_img == color_map[i],axis=-1))
        gray_img[index] = i
        t1 = time.time()
        time_cost = round(t1-t0,3)

        print(f"colorlabel cost time {time_cost}")

        # 保存图片
        dir,name = os.path.split(img_path)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir,name)
        cv2.imwrite(save_path,gray_img)

if __name__ == '__main__':
    cmap = np.array(
        [
            (0,0,0),        # 背景 Background   0
            (128,0,0),      # 飞机 Aero plane   1
            (0,128,0),      # 自行车 Bicycle    2
            (128,128,0),    # 鸟 Bird          3
            (0,0,128),      # 船 Boat          4
            (128,0,128),    # 瓶子 Bottle      5
            (0,128,128),    # 公共汽车 Bus      6
            (128,128,128),  # 小汽车 Car       7
            (64,0,0),       # 猫 Cat          8
            (192,0,0),      # 椅子 Chair      9
            (64,128,0),     # 奶牛 Cow        10
            (192,128,0),    # 餐桌 Dining table 11
            (64,0,128),     # 狗 Dog          12
            (192,0,128),    # 马 Horse        13
            (64,128,128),   # 摩托车 Motorbike  14
            (192,128,128),  # 人 Person       15
            (0,64,0),       # 盆栽 Potted plant  16
            (128,64,0),     # 绵羊 Sheep        17
            (0,192,0),      # 沙发 Sofa         18
            (128,192,0),    # 列车 Train        19
            (0,64,128),     # 显示器 TV         20
            (224,224,192)   # 边界 Border       21
        ]
    )
    # 文件路径
    img_dir = 'archive/segment_voc2012/SegmentationClass/'
    save_dir = 'archive/segment_voc2012/mask_Segmentation/'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    for img in os.listdir(img_dir):
        if not img.endswith((".png",".jpg")):
            continue
        img_path = os.path.join(img_dir,img)
        color2gray(img_path,color_map=cmap,save_dir=save_dir)