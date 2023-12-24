import csv

import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class FlowerSet(Dataset):
    def __init__(self,data_dir,csv_file,transform):
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.transform = transform
        self.data = self._load_data()


    def _load_data(self):
        data = []
        with open(self.csv_file,'r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                img_filename = row['imageName']
                label = row['label']

                label = int(label)

                data.append([img_filename,label])

            return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_filename,label = self.data[index]

        img = Image.open(os.path.join(self.data_dir,img_filename))


        if self.transform:
            img = self.transform(img)

        return img,label