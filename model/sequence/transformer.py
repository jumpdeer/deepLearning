import torch
import torch.nn as nn
import math
import torch.optim as optim

class Attention(nn.Module):
    # d_model代表隐特征维度，d_head表示每个头的维度，n_heads表示有多少个头, dff表示前馈神经网络的维度, dropout表示dropout的概率
    def __init__(self, config):
        super(Attention, self).__init__()

        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.d_head = self.d_model // config["n_heads"]

        self.W_Q = nn.Linear(self.d_model, self.d_model)
        self.W_K = nn.Linear(self.d_model, self.d_model)
        self.W_V = nn.Linear(self.d_model, self.d_model)

        self.softmax = nn.Softmax(dim=-1)  # 沿着列的方向去做softmax，即对于每一行，其所有列的数加起来为1

        self.norm = nn.LayerNorm(self.d_model)

        self.W_O = nn.Linear(self.d_model, self.d_model)

        self.is_mask = config["is_mask"]
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):

        batch_size = x.shape[0]
        N = x.shape[1]

        # 计算Q、K、V矩阵
        Q = self.W_Q(x).reshape(batch_size, N, self.n_heads, self.d_head).transpose(1,2)
        K = self.W_K(x).reshape(batch_size, N, self.n_heads, self.d_head).transpose(1,2)
        V = self.W_V(x).reshape(batch_size, N, self.n_heads, self.d_head).transpose(1,2) # [batch_size, n_heads, N, d_head]


        # 计算QK的分数
        QK_scores = Q @ K.transpose(-2,-1)  # [batch_size, n_heads, N, N]

        # 根据公式进行缩放
        scaled_scores = QK_scores / math.sqrt(self.d_head)

        # 如果需要计算mask
        if self.is_mask:
            # 计算上三角掩码
            up_triangle_mask = torch.triu(torch.ones(N, N, device=x.device) * float('-inf'), diagonal=1)  # 需要注意设备问题，需要创建在与x同在的设备上
            masked_scores = scaled_scores + up_triangle_mask
        else:
            masked_scores = scaled_scores

        # 进行Softmax操作
        attention_weights = self.softmax(masked_scores)

        # 对注意力权重进行一个dropout
        attention_weights = self.dropout(attention_weights)

        # 进行最后的注意力分数计算操作
        attention_scores = attention_weights @ V  # [batch_size, n_heads, N, d_head]

        # 对多头注意力进行拼接
        # attention_scores = attention_scores.transpose(-2, -3).reshape(batch_size, N, self.d_model)
        # 为什么不适用上面这一行，因为transpose操作同长不会在内存中移动数据，而是改变张量的"步长"，告诉pytorch如何索引数据。这可能导致张量在内存中的布局变得不连续
        # view()要求张量必须是内存连续的才能执行操作，不然会报错。而reshape()会暗中创建一个数据的副本，使其变得连续，然后再执行变形
        # 这样可以避免一些难以察觉的性能问题或bug
        attention_scores = attention_scores.transpose(-2, -3).contiguous().view(batch_size, N, self.d_model) # [batch_size, N, d_model]

        # 使用一个投影层将不同子空间的信息进行融合混合
        attention_scores = self.W_O(attention_scores)

        # 对最后的注意力分数进行残差连接
        output_scores = attention_scores + x

        output_scores = self.norm(output_scores)

        return output_scores  # [batch_size, N, d_model]


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.d_model = config["d_model"]
        self.dff = config["dff"]   # dff通常是d_model的四倍

        self.feed1 = nn.Linear(self.d_model, self.dff)
        self.activation = getattr(nn, config["activation"])()  # 可以从config中直接获取类，采用反射机制, 需要在后面再加一个括号,
        self.feed2 = nn.Linear(self.dff, self.d_model)         # 返回的才是一个激活函数实例，不然返回的是类名, 在下面使用forward的语句相当于给x赋予一个激活函数实例而不是调用

    def forward(self, x):
        x = self.feed1(x)
        x = self.activation(x)
        x = self.feed2(x)

        return x




