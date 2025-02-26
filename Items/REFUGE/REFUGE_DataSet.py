import torch
from torch.utils.data import Dataset
from albumentations import Resize
import cv2
import os


class REFUGE_Dataset(Dataset):
    def __init__(self,dataPath,maskPath,num_classes,transform):
        super().__init__()
        self.dataPath = dataPath
        self.maskPath = maskPath
        self.num_classes = num_classes
        self.datalist,self.labellist = self._load_data()
        self.transform = transform
        self.size = 2048

    def _load_data(self):
        datalist = []
        labellist = []

        for root,dirs,files in os.walk(self.dataPath):
            for item in files:
                datalist.append(os.path.join(self.dataPath,item))

        for root,dirs,files in os.walk(self.maskPath):
            for item in files:
                labellist.append(os.path.join(self.maskPath,item))

        return datalist,labellist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        img = cv2.imread(self.datalist[index])
        label = cv2.imread(self.labellist[index],cv2.IMREAD_UNCHANGED)

        result = Resize(height=self.size,width=self.size)(image=img,mask=label)

        img = result['image']
        label = result['mask']

        img = self.transform(img)
        label = torch.from_numpy(label).long()

        return img,label

    #添加图像灰度条 image为输入的图像， size为输入的图像尺寸(2444,2444)
    # def _letterbox_image(self, image, size):
    #     iw, ih = image.size
    #     w, h = size
    #     scale = min(w / iw, h / ih)
    #     nw = int(iw * scale)
    #     nh = int(ih * scale)
    #
    #     image = image.resize((nw, nh), Image.BICUBIC)
    #     new_image = Image.new('RGB', size, (128, 128, 128))
    #     new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    #     return new_image

    def _calculateSize_(self,w,h):
        maxNum = max(w,h)
        minNum = min(w,h)
        theClosest = 128
        theCake = 128

        while True:
            if abs(theCake*2 - maxNum) < abs(theClosest - maxNum):
                theClosest=theCake*2
                theCake = theCake*2
            elif abs(minNum - theCake*2) < abs(minNum - theCake):
                theClosest=theCake*2
                theCake=theCake*2
            else:
                return theClosest


