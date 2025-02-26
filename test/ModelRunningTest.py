import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from model.segmentation.BiSeNet import BiSeNet
import math


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


    def test_running(self):
        model = BiSeNet()
        model.eval()

        dummy_input = torch.randn(1,3,512,512)

        with torch.no_grad():
            outputs = model(dummy_input)

        print(outputs.shape)


    def test_backward(self):
        model = BiSeNet()
        model.train()  # 训练模式

        dummy_input = torch.randn(2, 3, 512, 512)  # batch size=2, 3通道, 256x256

        # 4) 构造一个假的标签 (B, H, W)，取值在 [0, 18] 间
        #    这里每个像素随机指定一个类别
        dummy_label = torch.randint(
            low=0, high=19,  # 19类
            size=(2, 512, 512)  # 与输入分辨率匹配
        )

        # 5) 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # 6) 前向传播
        outputs = model(dummy_input)
        print("outputs shape =", outputs.shape)
        # 预期: (2, 19, 256, 256)

        # 7) 计算 loss
        # CrossEntropyLoss需要 (B, C, H, W) 和 (B, H, W) 对应
        loss = criterion(outputs, dummy_label)
        print("Loss =", loss.item())

        # 8) 反向传播 + 更新
        optimizer.zero_grad()
        loss.backward()  # 检查是否能顺利 backward
        optimizer.step()  # 更新参数
        print("Backward & update success!")

    def test_math(self):
        print(2 ** math.ceil(math.log2(499)))

if __name__ == '__main__':
    unittest.main()
