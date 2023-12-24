import torch.utils.data
import torchvision.transforms as transforms
from torch import nn, optim
import os

from Items.butterfly_classification.model import AlexNet
from Items.butterfly_classification.my_dataset import butterfly_set


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.Resize((224,224)),
        transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
         ]
    )



    train_set = butterfly_set(os.path.join('archive', 'train/'), os.path.join('archive', 'Training_set.csv'), transform)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=20,shuffle=True,num_workers=0,pin_memory=True)

    net = AlexNet()
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.000005)

    for epoch in range(100):

        running_loss = 0.0
        for step,data in enumerate(train_loader,start=0):
            print(step)
            inputs,labels = data
            print(type(inputs),type(labels))
            inputs,labels = inputs.to(device),labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            print(outputs)
            loss = loss_function(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    save_path = 'Alexnet.pth'
    torch.save(net.state_dict(),save_path)

if __name__ == '__main__':
    main()