import os.path
import unittest
import imageio
from .model import FCN8s
from torchvision.transforms import transforms
import numpy as np
import torchvision.transforms
from PIL import Image
from torchvision import models
import torch.nn.functional as F
import torch


class MyTestCase(unittest.TestCase):
    def test_something(self):
        img = Image.open("1.png")
        img = torchvision.transforms.ToTensor(img)
        print(img)

    def test1(self):
        vgg = models.vgg16(True)
        feature = torch.nn.Sequential(*list(vgg.children())[:])
        print(feature)


    def testPredictOne(self):
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

        net = FCN8s(21)
        net.load_state_dict(torch.load('FCN8s.pth'))

        img = Image.open('./archive/segment_voc2012/image/2008_000011.jpg')
        cropImg = img.resize((320, 480))
        img = transform(cropImg)
        img = torch.unsqueeze(img, dim=0)

        output = net(img)
        predict = torch.argmax(F.softmax(output, dim=1), 1)
        predict = predict.numpy()

        RGB_img = np.zeros(shape=(predict.shape[1], predict.shape[2], 3), dtype=np.uint8)
        for i in range(RGB_img.shape[0]):
            for j in range(RGB_img.shape[1]):
                index = predict[0, i, j]
                RGB_img[i, j] = cmap[index]

        RGB_img = Image.fromarray(RGB_img)
        RGB_img = Image.blend(cropImg, RGB_img, 0.8)

        RGB_img.save('test1.jpg')


if __name__ == '__main__':
    unittest.main()
