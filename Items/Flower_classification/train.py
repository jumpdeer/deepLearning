import torch.utils.data
import torchvision.transforms as transforms
from torch import nn, optim
from MyDataset import FlowerSet
import os

from AlexNet import AlexNet



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")

    transform = transforms.Compose(
        [transforms.Resize((227, 227)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
         ]
    )

    net = AlexNet(num_classes=5,init_weights=True)
    net.to(device)

    train_set = FlowerSet(os.path.join('Dataset','train_image/'), os.path.join('Dataset','train_label.csv'), transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=20, shuffle=True, num_workers=0, pin_memory=True)


    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lr=0.0002,params=net.parameters())

    for epoch in range(100):
        print(f"第{epoch}轮训练")
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):

            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            print(outputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


    save_path = 'Alexnet.pth'
    torch.save(net.state_dict(),save_path)

if __name__ == '__main__':
    main()