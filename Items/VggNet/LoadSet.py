from torch.utils.data import Dataset
from PIL import Image
import os

class LoadSet(Dataset):
    def __init__(self,data_dir,transform):
        self.data_dir = data_dir
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        data = []
        paths = os.walk(self.data_dir)
        for path,dir_lst,file_lst in paths:
            for file_name in file_lst:
                img_filename = file_name
                if file_name[0] == 'n':
                    data.append([img_filename,0])  # 0代表正常路面
                else:
                    data.append([img_filename,1])  # 1代表坑洼路面

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_filename, label = self.data[item]

        img = Image.open(os.path.join(self.data_dir, img_filename))

        if self.transform:
            img = self.transform(img)

        return img, label
