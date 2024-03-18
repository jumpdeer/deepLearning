import cv2

from Unet import Unet
import numpy as np
import torch
import os
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import transforms
import torchvision.transforms.functional
from torchvision.utils import save_image

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

def cropOutputImage(image,ori_size):
    new_image = torchvision.transforms.functional.center_crop(image,ori_size)
    return new_image

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image

def main():
    colormap = np.array([
        (255,255,255),
        (255,0,0),
        (0,128,0)
    ])

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )

    net = Unet(in_channels=3,out_channels=3)
    net.load_state_dict(torch.load('Unet.pth'))

    for path, dir_list, file_list in os.walk('./data/REFUGE-Test400/Test400'):
        for item in file_list:
            img = Image.open(os.path.join('./data/REFUGE-Test400/Test400',item))
            img = letterbox_image(img, [2444,2444])

            img = transform(img)
            img = torch.unsqueeze(img,dim=0)

            output = net(img)
            output = cropOutputImage(output,[1634,1634])

            predict = torch.argmax(F.log_softmax(output, dim=1), 1)
            predict = predict.numpy()

            RGB_img = np.zeros(shape=(predict.shape[1],predict.shape[2],3),dtype=np.uint8)
            for i in range(colormap.shape[0]):
                index = np.where(np.all(predict == i, axis=-1))
                for j in range(RGB_img.shape[0]):
                    RGB_img[index,j] = colormap[i]

            RGB_img = Image.fromarray(RGB_img)
            RGB_img.show()

if __name__=='__main__':
    main()
