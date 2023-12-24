from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os


class VOCset(Dataset):
    def __init__(self,data_dir,mask_dir,txt_file,num_classes,transform):
        super().__init__()
        self.data_dir = data_dir
        self.txt_file = txt_file
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        data = []
        with open(self.txt_file,'r') as f:

            for row in f.readlines():
                img = row.split('\n')[0]
                data.append(img)

        return data



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        name = self.data[index]

        img = Image.open(os.path.join(self.data_dir,name+'.jpg'))

        label = Image.open(os.path.join(self.mask_dir,name+'.png'))

        if self.transform:
            img = self.transform(img)


        return img,label


    @staticmethod
    def collate_fn(batch):
        images,targets = list(zip(*batch))
        batched_imgs = cat_list(images,)



def cat_list(images,fill_value=0):
    # 计算该batch数据中,channel,h,w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    