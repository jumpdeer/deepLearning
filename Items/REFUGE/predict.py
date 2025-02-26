import cv2
from Unet import Unet
import numpy as np
import torch
import os
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import transforms
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

def main():
    colormap = np.array([
        (255,255,0),
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
            img = cv2.imread(os.path.join('./data/REFUGE-Test400/Test400',item))

            img = Resize(p=1,width=2048,height=2048)(image=img)['image']

            img = transform(img)
            img = torch.unsqueeze(img,dim=0)

            output = net(img)

            predict = torch.argmax(F.softmax(output,dim=1), 1)
            predict = predict.numpy()

            print(np.sum(predict==0))
            print(np.sum(predict==1))
            print(np.sum(predict==2))

            RGB_img = np.zeros(shape=(predict.shape[1],predict.shape[2],3),dtype=np.uint8)
            for i in range(RGB_img.shape[0]):   # 遍历颜色表
                for j in range(RGB_img.shape[1]):
                    index = predict[0,i,j]
                    RGB_img[i,j]=colormap[index]

            print(RGB_img.shape)
            print(RGB_img)
            RGB_img = Image.fromarray(RGB_img)
            RGB_img.save('./pre1.jpg')


if __name__=='__main__':
    main()
