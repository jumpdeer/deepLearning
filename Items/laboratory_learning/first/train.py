import torch
import torchvision
import torch.nn as nn
from Items.laboratory_learning.first.model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),     # 将图片转化为tensor
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   # 对像素进行归一化处理

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)
    
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()

    loss_function = nn.CrossEntropyLoss()    # 确定损失函数为交叉熵损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001)    # 优化0器为Adam，初始学习率为0.001

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data     # 数据和标签

            print(inputs,labels)

            # zero the parameter gradients
            optimizer.zero_grad()   # 将梯度矩阵清零，清空之前的梯度

            # forward + backward + optimize
            outputs = net(inputs)    # 将数据输入网络
            loss = loss_function(outputs, labels)  # 根据输出和标签计算损失函数
            loss.backward()         # 函数的反向传播
            optimizer.step()        # 执行参数更新

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:    # print every 500 mini-batches
                with torch.no_grad():
                    outputs = net(val_image)  # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')

    save_path = 'Lenet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()
