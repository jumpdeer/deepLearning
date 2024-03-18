import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os


class VOCset(Dataset):
    def __init__(self,data_dir,mask_dir,txt_file,num_classes):
        super().__init__()
        self.data_dir = data_dir
        self.txt_file = txt_file
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.datalist,self.labellist = self._load_data()
        self.high = 320
        self.width = 480
        self.datalist = self.filter(self.datalist)
        self.labellist = self.filter(self.labellist)

    def _load_data(self):
        datalist = []
        labellist = []
        with open(self.txt_file,'r') as f:

            for row in f.readlines():
                img = row.split('\n')[0]
                datalist.append(os.path.join(self.data_dir,img+'.jpg'))
                labellist.append(os.path.join(self.mask_dir,img+'.png'))

        return datalist,labellist



    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        img = Image.open(self.datalist[index])
        label = Image.open(self.labellist[index])

        img, label = img_transforms(img, label, 320, 480)

        return img,label

    def filter(self, images):
        return [im for im in images if (Image.open(im).size[1] > self.high and Image.open(im).size[0] > self.width)]

def rand_crop(data, label, high, width):  # high, width为裁剪后图像的固定宽高(320x480)
    im_width, im_high = data.size
    # 生成随机点位置
    left = np.random.randint(0, im_width - width)
    top = np.random.randint(0, im_high - high)
    right = left + width
    bottom = top + high
    # 图像随机裁剪(图像和标签一一对应)
    data = data.crop((left, top, right, bottom))
    label = label.crop((left, top, right, bottom))

    # 图像随机翻转(图像和标签一一对应)
    angle = np.random.randint(-15, 15)
    data = data.rotate(angle)  # 逆时针旋转
    label = label.rotate(angle)  # 逆时针旋转
    return data, label

    # 预处理
def img_transforms(data, label, high, width):
    data, label = rand_crop(data, label, high, width)
    data_tfs = transforms.Compose([
        transforms.ToTensor(),
        # 标准化，据说这6个参数是在ImageNet上百万张数据里提炼出来的，效果最好
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data = data_tfs(data)

    label = torch.from_numpy(np.array(label))
    label = label.long()

    return data, label