import torch
import torch.nn as nn


# 表示图中的一个模块
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        初始化 LSTMCell
        :param input_size: 输入特征的维度
        :param hidden_size: 隐状态的维度
        """
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 定义两个线性层，用于计算输入和隐状态的线性变换，输出维度为 4*hidden_size，
        # 分别对应输入门、遗忘门、候选状态和输出门，即图中蓝、绿、黄、紫四个单元框
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, hidden):
        """
        前向传播
        :param x: 当前时间步输入，形状 [batch_size, input_size]
        :param hidden: 包含 (h, c)，h 和 c 的形状均为 [batch_size, hidden_size]
        :return: 当前时间步的输出 h 和更新后的细胞状态 c
        """
        hx, cx = hidden   # hx：代表隐状态（短期记忆）， cx：代表细胞状态（长期记忆）
        # 计算输入和隐状态的线性变换后相加
        gates = self.x2h(x) + self.h2h(hx)
        # 将结果均分为四部分，分别对应输入门、遗忘门、候选状态和输出门
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        # 应用激活函数
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        # 更新细胞状态
        cy = forgetgate * cx + ingate * cellgate
        # 计算当前隐状态
        hy = outgate * torch.tanh(cy)
        return hy, cy


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        """
        初始化多层 LSTM 模块
        :param input_size: 输入特征的维度
        :param hidden_size: 隐状态的维度
        :param num_layers: LSTM 层数
        """
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # 使用 ModuleList 存储每一层的 LSTMCell
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cur_input_size = input_size if i == 0 else hidden_size
            self.cells.append(LSTMCell(cur_input_size, hidden_size))

    def forward(self, x, hidden=None):
        """
        前向传播
        :param x: 输入序列，形状 [batch_size, seq_len, input_size]
        :param hidden: 初始隐状态和细胞状态，若为 None 则自动初始化为零
                        hidden = (h0, c0)，其中 h0 和 c0 均为列表，每个元素形状为 [batch_size, hidden_size]
        :return: 输出序列 outputs 以及最终的 (h, c)
                 outputs 的形状为 [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.size()
        # 初始化隐状态和细胞状态
        if hidden is None:
            h0 = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
            c0 = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        else:
            h0, c0 = hidden

        outputs = []
        # 对于序列中的每个时间步
        for t in range(seq_len):
            input_t = x[:, t, :]  # 当前时间步输入，形状 [batch_size, input_size]
            # 多层堆叠
            for i, cell in enumerate(self.cells):
                h, c = cell(input_t, (h0[i], c0[i]))
                h0[i] = h  # 更新当前层的隐状态
                c0[i] = c  # 更新当前层的细胞状态
                input_t = h  # 当前层的输出作为下一层的输入
            outputs.append(h)

        # 将所有时间步的输出堆叠起来，形状 [batch_size, seq_len, hidden_size]
        outputs = torch.stack(outputs, dim=1)
        return outputs, (h0, c0)
