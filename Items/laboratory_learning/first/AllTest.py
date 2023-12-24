import unittest
import torchvision.transforms as transforms
import torch
import torchvision
from model import LeNet


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


    def test_dataset(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),  # 将图片转化为tensor
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 对像素进行归一化处理
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=False, transform=transform)
        print(train_set)

    def test_changeFile(self):
        model = LeNet()
        model.load_state_dict(torch.load("Lenet.pth"))
        model.eval()
        example = torch.rand(1,3,32,32)
        traced_script_module = torch.jit.trace(model,example)
        traced_script_module.save("model.pt")


if __name__ == '__main__':
    unittest.main()
