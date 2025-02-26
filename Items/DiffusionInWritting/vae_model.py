import torch
import torch.nn as nn


# VAE model
class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4, image_size=512): # 3x512x512 -> 4x64x64
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.image_size = image_size


        # Encoder
        # 3 x 512 x 512 -> 4 x 64 x 64
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, 32), # 32 x 256 x 256
            self._conv_block(32, 64), # 64 x 128 x 128
            self._conv_block(64, 128), # 128 x 64 x 64
        )

        # Encoder output
        self.fc_mu = nn.Conv2d(128, latent_dim, 1)  # 4 x 64 x 64
        self.fc_var = nn.Conv2d(128, latent_dim, 1) # 4 x 64 x 64

        # Decoder
        # 4 x 64 x 64 -> 3 x 512 x 512
        self.decoder_input = nn.Conv2d(latent_dim, 128, 1) # 128 x 64 x 64
        self.decoder = nn.Sequential(
            self._conv_transpose_block(128, 64), # 64 x 128 x 128
            self._conv_transpose_block(64, 32),  # 32 x 256 x 256
            self._conv_transpose_block(32, in_channels),  # 1 x 512 x 512
        )

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.LeakyReLU()  # use LeakyReLU for non-linearity effectiveness
        )

    def _conv_transpose_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.LeakyReLU()
        )

    def encode(self, input):
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        # result = self.sigmoid(self) # if the original images are normalized to [0, 1]
        result = self.tanh(result) # if the original images are normalized to [-1, 1]
        return result.view(-1, self.in_channels, self.image_size, self.image_size)


    # 由于从分布中采样（例如直接调用
    # torch.randn）会导致采样操作不可导，进而无法反向传播。重参数化技巧通过将随机性从采样中分离出来，使得模型整体可导。
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input):
        """
        Return 4 values:
        reconstruction, input, mu, log_var
        :param input:
        :return:
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var) # 潜在空间表达 Latent Space | 4 x 64 x 64
        return self.decode(z), input, mu, log_var  # 1.预测值 2.输入值 3.均值 4.方差
