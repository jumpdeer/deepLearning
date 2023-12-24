import unittest
import imageio
import numpy as np
import torchvision.transforms
from PIL import Image
from torchvision import models
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


if __name__ == '__main__':
    unittest.main()
