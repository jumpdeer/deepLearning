import os

import torch
from PIL import Image
import numpy as np
from model import FCN8s
import torch.nn.functional as F
from torchvision.transforms import transforms
import torchvision
from albumentations import Resize



def Iou(target_all, pred_all,n_class):
    """
        target是真实标签，shape为(h,w)，像素值为0，1，2...
        pred是预测结果，shape为(h,w)，像素值为0，1，2...
        n_class:为预测类别数量
        """
    pred_all = pred_all.to('cpu')
    target_all = target_all.to('cpu')
    iou = []
    for i in range(target_all.shape[0]):
        pred = pred_all[i]
        target = target_all[i]

        h, w = target.shape
        # 转为one-hot，shape变为(h,w,n_class)
        target_one_hot = np.eye(n_class)[target]
        pred_one_hot = np.eye(n_class)[pred]

        target_one_hot[target_one_hot != 0] = 1
        pred_one_hot[pred_one_hot != 0] = 1
        join_result = target_one_hot * pred_one_hot

        join_sum = np.sum(np.where(join_result == 1))  # 计算相交的像素数量
        pred_sum = np.sum(np.where(pred_one_hot == 1))  # 计算预测结果非0得像素数
        target_sum = np.sum(np.where(target_one_hot == 1))  # 计算真实标签的非0得像素数

        iou.append(join_sum / (pred_sum + target_sum - join_sum + 1e-6))

    return np.mean(iou)

def rand_crop(data, high, width):  # high, width为裁剪后图像的固定宽高(320x480)
    im_width, im_high = data.size
    # 生成随机点位置
    left = np.random.randint(0, im_width - width)
    top = np.random.randint(0, im_high - high)
    right = left + width
    bottom = top + high
    # 图像随机裁剪(图像和标签一一对应)
    data = data.crop((left, top, right, bottom))

    return data

def main():
    cmap = np.array(
        [
            (0, 0, 0),  # 背景 Background   0
            (128, 0, 0),  # 飞机 Aero plane   1
            (0, 128, 0),  # 自行车 Bicycle    2
            (128, 128, 0),  # 鸟 Bird          3
            (0, 0, 128),  # 船 Boat          4
            (128, 0, 128),  # 瓶子 Bottle      5
            (0, 128, 128),  # 公共汽车 Bus      6
            (128, 128, 128),  # 小汽车 Car       7
            (64, 0, 0),  # 猫 Cat          8
            (192, 0, 0),  # 椅子 Chair      9
            (64, 128, 0),  # 奶牛 Cow        10
            (192, 128, 0),  # 餐桌 Dining table 11
            (64, 0, 128),  # 狗 Dog          12
            (192, 0, 128),  # 马 Horse        13
            (64, 128, 128),  # 摩托车 Motorbike  14
            (192, 128, 128),  # 人 Person       15
            (0, 64, 0),  # 盆栽 Potted plant  16
            (128, 64, 0),  # 绵羊 Sheep        17
            (0, 192, 0),  # 沙发 Sofa         18
            (128, 192, 0),  # 列车 Train        19
            (0, 64, 128),  # 显示器 TV         20
            (224, 224, 192)  # 边界 Border       21
        ]
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )
    #
    # net = FCN8s(22)
    # net.load_state_dict(torch.load('FCN8s.pth'))

    net = torchvision.models.segmentation.fcn_resnet50(pretrained=True, progress=True, num_classes=21,
                                                                aux_loss=None)

    pre_path = 'archive/segment_voc2012/image/'

    with open('archive/segment_voc2012/test.txt','r') as f:
        for row in f.readlines():
            img_name = row.split('\n')[0]+'.jpg'
            img_path =os.path.join(pre_path,img_name)
            img = Image.open(img_path)
            print(row)

            cropImg = img.resize((320,480))
            img = transform(cropImg)
            img = torch.unsqueeze(img, dim=0)

            output = net(img)
            predict = torch.argmax(F.softmax(output,dim=1),1)
            predict = predict.numpy()

            RGB_img = np.zeros(shape=(predict.shape[1],predict.shape[2],3),dtype=np.uint8)
            for i in range(RGB_img.shape[0]):
                for j in range(RGB_img.shape[1]):
                    index = predict[0,i,j]
                    RGB_img[i,j]=cmap[index]

            RGB_img = Image.fromarray(RGB_img)
            RGB_img = Image.blend(cropImg,RGB_img,0.4)
            RGB_img.save(f'./archive/testresult/{img_name}')

if __name__=='__main__':
    main()