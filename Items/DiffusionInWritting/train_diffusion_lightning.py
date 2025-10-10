import os
from PIL import Image
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from modelscope.models.multi_modal.diffusion.model import DiffusionModel
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from datasets import load_dataset

from diffusion_model import UNet_Transformer, NoiseScheduler, sample_cfg
from transformers import CLIPTokenizer, CLIPTextModel

import pytorch_lightning as pl
import wandb

class DiffusionModelPL(pl.LightningModule):
    def __init__(self, in_channels=3, lr=1e-4, n_epochs=1000, num_timesteps=1000, image_size=64):
        super().__init__()
        # 将超参数保存下来，方便后续访问
        self.save_hyperparameters()

        # 初始化模型和噪声调度器
        self.diffusion_model = UNet_Transformer(in_channels=in_channels)
        self.noise_scheduler = NoiseScheduler(num_timesteps=num_timesteps, device=self.device)

        # 初始化 CLIP 模型 (注意：这里我们不在模型移动到GPU，Lightning会处理)
        model_local_path = "./offline_assets/openai-clip-vit-base-patch32"
        self.tokenizer = CLIPTokenizer.from_pretrained(model_local_path)
        self.text_encoder = CLIPTextModel.from_pretrained(model_local_path)

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        text  = batch["text"]

        text_inputs = self.tokenizer(text, padding="max_length", max_length=self.tokenizer.model_max_length,
                                     truncation=True, return_tensors="pt")
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device)).last_hidden_state

        timesteps = torch.randint(0, self.hparams.num_timesteps, (images.shape[0],), device=self.device).long()
        noisy_images, noise = self.noise_scheduler.add_noise(images, timesteps)
        noise_pred = self.diffusion_model(noisy_images, timesteps, text_embeddings)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # 使用self.log 记录日志
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch):
        # 验证逻辑
        images = batch["images"]
        text = batch["text"]

        text_inputs = self.tokenizer(text, padding="max_length", max_length=self.tokenizer.model_max_length,truncation=True, return_tensors="pt")
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device)).last_hidden_state

        timesteps = torch.randint(0, self.hparams.num_timesteps, (images.shape[0],), device=self.device).long()
        noisy_images, noise = self.noise_scheduler.add_noise(images, timesteps)
        noise_pred = self.diffusion_model(noisy_images, timesteps, text_embeddings)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        self.log("val_loss", loss, on_epoch=True)

    def configure_optimizers(self):
        # 定义优化器和学习率调度器
        optimizer = AdamW(self.diffusion_model.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.n_epochs)
        return [optimizer], [scheduler]


class ImageGenerationCallback(Callback):
    def __init__(self, save_interval):
        super().__init__()
        self.save_interval = save_interval
        os.makedirs('diffusion_results_pl', exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.save_interval == 0:
            print(f"\n--- Generating images at epoch {epoch + 1} ---")
            pl_module.eval()
            with torch.no_grad():
                sample_text = ["a water type pokemon", "a red pokemon with a red fire tail"]
                text_input = pl_module.tokenizer(sample_text, padding="max_length",
                                                 max_length=pl_module.tokenizer.model_max_length, truncation=True,
                                                 return_tensors="pt")
                text_embeddings = pl_module.text_encoder(text_input.input_ids.to(pl_module.device)).last_hidden_state

                sampled_images = sample_cfg(pl_module.diffusion_model, pl_module.noise_scheduler, len(sample_text),
                                            pl_module.hparams.in_channels, text_embeddings,
                                            image_size=pl_module.hparams.image_size, guidance_scale=3.0)

                # 保存并记录图像
                for i, img in enumerate(sampled_images):
                    img = img * 0.5 + 0.5
                    img = img.detach().cpu().permute(1, 2, 0).numpy()
                    img = (img * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img)
                    img_pil.save(f"diffusion_results_pl/generated_image_epoch_{epoch+1}_sample_{i}.png")

                    # 使用 trainer.logger 来记录 wandb
                    trainer.logger.experiment.log({f"generated_image_epoch_{epoch+1}_{i}": wandb.Image(img_pil)})
            pl_module.train()  # 恢复训练模式

# 创建一个LightningDataModule，封装数据加载
class PokemonDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=80, image_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.pokemon_local_path = "./offline_assets/pokemon-blip-captions-en-zh"
        self.preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def setup(self, stage=None):
        # 下载、切分数据集等操作
        dataset = load_dataset(self.pokemon_local_path, split="train")
        dataset.set_transform(self._transform)
        self.train_dataset = dataset.select(range(0, 600))
        self.val_dataset = dataset.select(range(600, 800))

    def _transform(self, examples):
        images = [self.preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images, "text": examples["en_text"]}

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False, drop_last=False)

if __name__ == "__main__":
    N_EPOCHS = 1000
    BATCH_SIZE = 80
    LR = 1e-4
    NUM_TIMESTEPS = 1000
    SAVE_CHECKPOINT_INTERVAL = 100
    IMAGE_SIZE = 64
    IN_CHANNELS = 3

    # --- 初始化模块 ---
    model = DiffusionModelPL(
        in_channels=IN_CHANNELS,
        lr=LR,
        n_epochs=N_EPOCHS,
        num_timesteps=NUM_TIMESTEPS,
        image_size=IMAGE_SIZE  # 传入image_size给callback用
    )
    datamodule = PokemonDataModule(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
    image_callback = ImageGenerationCallback(save_interval=SAVE_CHECKPOINT_INTERVAL)

    # --- 配置 Logger (用于 WandB) ---
    wandb_logger = pl.loggers.WandbLogger(project="diffusion_from_scratch_pl", mode="offline")

    # --- 配置并启动 Trainer ---
    trainer = pl.Trainer(
        accelerator="gpu",  # 使用 'gpu' 或 'cpu'
        devices=[1,2],  # !!! 在这里设置你想使用的 GPU 数量 !!!
        strategy="ddp",  # 使用 DDP 策略进行多卡训练 (比 DP 更快)
        max_epochs=N_EPOCHS,
        logger=wandb_logger,
        callbacks=[image_callback],  # 添加我们的图片生成回调
        # checkpointing 会自动开启，并保存在 logger 的目录里
    )

    trainer.fit(model,datamodule)