import torch.utils.data
import torchvision.transforms as transforms
from torch import nn,optim
from classification.model.VggNet import vgg19
from LoadSet import LoadSet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ]
    )

    net = vgg19(num_classes=2,init_weight=True)
    net.to(device)

    train_set = LoadSet('./data',transform)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=5,shuffle=True,num_workers=0,pin_memory=True)

    loss_function = nn.CrossEntropyLoss()    # 交叉熵损失函数
    optimizer = optim.Adam(lr=0.0000005,params=net.parameters())

    for epoch in range(5):
        net.train()
        running_loss = 0.0
        for step,data in enumerate(train_loader,start=0):

            inputs,labels = data

            inputs,labels = inputs.to(device),labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            print(outputs)

            loss = loss_function(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print(running_loss)
    save_path = 'vgg19Net.pth'
    torch.save(net.state_dict(),save_path)

if __name__ == '__main__':
    main()