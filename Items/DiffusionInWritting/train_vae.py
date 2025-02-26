import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from vae_model import VAE
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from datasets import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
batch_size = 32
learning_rate = 1e-3
num_epochs = 100
image_size = 512
latent_dim = 4

dataset = load_dataset("svjack/pokemon-blip-captions-en-zh", split="train")


preprocess = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(), # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])])

def transform(example):
    images = [preprocess(image.convert("RGB")) for image in example["image"]]
    en_texts = example["en_text"]
    return {"images": images, "en_texts": en_texts}

dataset.set_transform(transform)

train_dataset = dataset.select(range(0, 600))
val_dataset = dataset.select(range(600, 800))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# 初始化模型
vae = VAE(in_channels=3, latent_dim=latent_dim, image_size=image_size).to(device)

# 从保存的模型中加载并继续训练 | 如果第一次运行结束并保存模型后希望继续训练，则可以使用下面的代码继续训练
# vae.load_state_dict(torch.load('vae_model.pth', weights_only=True))
# learning_rate = 5e-4 # 如有必要，更新学习率继续训练
# batch_size = 24  # 如有必要，更新小批次继续训练

# 优化器和学习率调度器
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=5e-5)

# 损失函数
def vae_loss_function(recon_x, x, mu, logvar):
    MSE= nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

# 创建保存结果的目录
os.makedirs("vae_results", exist_ok=True)

# 训练循环
best_loss = float('inf')
early_stopping_patience = 30
no_improvement = 0

for epoch in range(num_epochs):
    vae.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_dataloader):

        data = batch["images"].to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = vae(data)
        loss = vae_loss_function(recon_batch, data, mu, logvar)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_dataloader.dataset)}] '
                  f'({100. * batch_idx / len(train_dataloader):.0f} %)]\tLoss: {loss.item()/len(data):.6f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss/len(train_dataloader.dataset):.4f} lr: {optimizer.param_groups[0]["lr"]}')


    # 验证
    vae.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            data = batch["images"].to(device)
            recon_batch, mu, logvar = vae(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar)
            val_loss += loss.item()

    val_loss /= len(val_dataloader.dataset)
    print(f'====> Validation set loss: {val_loss:.4f}')

    # 学习率调度
    scheduler.step(val_loss)

    # 验证和可视化
    if epoch % 10 == 0:
        with torch.no_grad():
            # 获取实际的批次大小
            actual_batch_size = data.size(0)
            # 重构图像
            n = min(actual_batch_size, 8)
            comparison = torch.cat([data[:n], recon_batch.view(actual_batch_size, 3, image_size, image_size)[:n]])
            comparison = (comparison * 0.5) + 0.5 # 将[-1, 1]转换回[0, 1]
            save_image(comparison.cpu, f'vae_results/reconstruction_{epoch}.png', nrow=n)

torch.save(vae.state_dict(), f'vae_model.pth')
print("Training completed.")