import unittest
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from Items.REFUGE.Unet import Unet
import torchvision.transforms.functional
import torch.nn.functional as F

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

class MyTestCase(unittest.TestCase):
    def test_something(self):
        img = cv2.imread('1.jpg',cv2.IMREAD_UNCHANGED)
        print(np.sum(img==2))
        print(img)

    def test_cv(self):
        color_img = cv2.imread('test.bmp', cv2.IMREAD_UNCHANGED)

        gray_img = color_img.copy()

        print(f'没换值之前视杯像素数量{np.sum(gray_img == 0)}')
        gray_img[np.where(gray_img == 0)] = 1
        print(f'换值之后视杯像素数量{np.sum(gray_img == 1)}')

        print(f'没换值之前背景像素数量{np.sum(gray_img == 255)}')
        gray_img[np.where(gray_img == 255)] = 0
        print(f'换值之后视杯像素数量{np.sum(gray_img == 0)}')

        print(f'没换值之前视盘像素数量{np.sum(gray_img == 128)}')
        gray_img[np.where(gray_img == 128)] = 2
        print(f'换值之后视杯像素数量{np.sum(gray_img == 2)}')

        print(f'换值之后背景:{np.sum(gray_img==0)},视杯:{np.sum(gray_img==1)},视盘:{np.sum(gray_img==2)}')


    def test_np(self):
        colormap = np.array([
            (255, 255, 255),
            (255, 0, 0),
            (0, 128, 0)
        ])
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        )

        net = Unet(in_channels=3, out_channels=3)
        net.load_state_dict(torch.load('Unet.pth'))

        img = Image.open('./data/REFUGE-Training400/Glaucoma/g0001.jpg')
        img = letterbox_image(img,[2444,2444])

        img =transform(img)
        img = torch.unsqueeze(img,dim=0)

        output = net(img)
        output = cropOutputImage(output,[2056,2124])

        predict = F.log_softmax(output, dim=1)
        predict = torch.argmax(predict,dim=1)

        predict = predict.numpy()
        #
        print(np.sum(predict==0))
        print(np.sum(predict==1))
        print(np.sum(predict==2))
        #
        # print(predict.shape)
        # RGB_img = np.zeros(shape=(predict.shape[1],predict.shape[2],3),dtype=np.uint8)
        # for i in range(colormap.shape[0]):
        #     index = np.where(np.all(predict == i,axis=-1))
        #     for j in range(RGB_img.shape[0]):
        #         RGB_img[index]=colormap[i]
        #
        # img = Image.fromarray(RGB_img)
        # img.save('test.png')


    def testTensordot(self):
        arr1 = torch.tensor([[[1,2,0,1,1],
                             [2,1,0,0,0],
                             [0,1,1,0,0]]])
        # arr2 = torch.tensor([[[[0.1,0.2,0.3,0.4,0.5]]],[],[]])
        print(arr1.shape)
        arr2 = F.one_hot(arr1)
        print(arr2)
        print(arr2.shape)
        arr2 = arr2.permute(0,3,1,2)  # 交换维度，onehot编码之后的交换
        print(arr2)
        print(arr2.shape)
        arr2 = arr2.reshape(1,-1)
        print(arr2)
        print(arr2.shape)

    def testImg(self):
        img = cv2.imread('./data/Train400/Mask-Training400/g0001.png',cv2.COLOR_BGR2GRAY)
        print(np.sum(img==0))
        print(np.sum(img==1))
        print(np.sum(img==2))

if __name__ == '__main__':
    unittest.main()
