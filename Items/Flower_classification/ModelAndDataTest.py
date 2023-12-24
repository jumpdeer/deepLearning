import unittest

import torch

from Items.Flower_classification.model import AlexNet
from MyDataset import FlowerSet
import torchvision.transforms as transforms
import csv
from PIL import Image


class MyTestCase(unittest.TestCase):
    def test_something(self):
        transform = transforms.Compose(
            [transforms.Resize((227, 227)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
             ]
        )
        train_set = FlowerSet('Dataset/train_image/', './Dataset/train_label.csv', transform)
        print(train_set.__getitem__(4))


    def test_model(self):
        transform = transforms.Compose(
            [transforms.Resize((227, 227)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
             ]
        )

        net = AlexNet(num_classes=5)
        net.load_state_dict(torch.load('Alexnet.pth'))

        net.eval()
        all = 0.0
        right = 0.0
        with open('Dataset/test_label.csv', 'r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                img_filename = './Dataset/test_image/'+row['imageName']
                label = row['label']

                im = Image.open(img_filename)
                im = transform(im)
                im = torch.unsqueeze(im, dim=0)

                with torch.no_grad():
                    outputs = net(im)
                    predict = torch.max(outputs, dim=1)[1].data.numpy()

                    if int(label) == int(predict):
                        right +=1

                all+=1
        print('模型在测试集准确率为：'+str(right/all))






if __name__ == '__main__':
    unittest.main()
