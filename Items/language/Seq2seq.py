import torch
from LSTM import LSTM
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embed_size, hidden_size, num_layers=1):
        """
        Encoder模块：对输入序列进行编码
        :param input_vocab_size: 输入词汇表大小
        :param embed_size: 词嵌入维度
        :param hidden_size: LSTM隐状态和细胞状态维度
        :param num_layers: LSTM层数
        """
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_size)
        self.lstm = LSTM(embed_size, hidden_size, num_layers)

    def forward(self, src):
        """
        :param src: [batch_size, src_seq_len]，源序列（词索引）
        :return: outputs: [batch_size, src_seq_len, hidden_size]
                 hidden: (h, c) 编码后的最后隐状态和细胞状态（每层都是一个张量）
        """
        embedded = self.embedding(src)  # [batch_size, src_seq_len, embed_size]
        outputs, (h, c) = self.lstm(embedded)
        return outputs, (h, c)


class Decoder(nn.Module):
    def __init__(self, output_vocab_size, embed_size, hidden_size, num_layers=1):
        """
        Decoder模块：根据编码器的状态逐步生成输出序列
        :param output_vocab_size: 目标词汇表大小
        :param embed_size: 词嵌入维度
        :param hidden_size: LSTM隐状态和细胞状态维度
        :param num_layers: LSTM层数
        """
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_vocab_size, embed_size)
        self.lstm = LSTM(embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, input, hidden):
        """
        :param input: [batch_size, 1]，当前时间步输入（词索引）
        :param hidden: (h, c)，上一时刻的隐状态和细胞状态
        :return: predictions: [batch_size, output_vocab_size]，当前时刻的预测分布
                 hidden: 更新后的 (h, c)
        """
        embedded = self.embedding(input)  # [batch_size, 1, embed_size]
        # 注意：这里传入的embedded张量时间步为1
        output, hidden = self.lstm(embedded, hidden)  # output: [batch_size, 1, hidden_size]
        predictions = self.fc(output.squeeze(1))  # [batch_size, output_vocab_size]
        return predictions, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        """
        Seq2Seq模型，由编码器和解码器组成
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        :param src: [batch_size, src_seq_len]，源序列
        :param trg: [batch_size, trg_seq_len]，目标序列（标签），第一列通常为<sos>起始符
        :param teacher_forcing_ratio: teacher forcing的比例
        :return: outputs: [batch_size, trg_seq_len, output_vocab_size]
        """
        batch_size, trg_len = trg.shape
        output_vocab_size = self.decoder.fc.out_features

        # 用于存储每个时间步的预测结果
        outputs = torch.zeros(batch_size, trg_len, output_vocab_size).to(self.device)

        # 编码器前向传播，获取上下文信息
        encoder_outputs, hidden = self.encoder(src)

        # encoder_outputs: [batch_size, seq_len, hidden_size]
        # hidden = (h0, c0)
        # h0, c0: [batch_size, hidden_size]

        # 解码器的初始输入为目标序列的第一个token（通常是<sos>标记）
        input = trg[:, 0]  # [batch_size]

        # 逐步生成目标序列
        for t in range(1, trg_len):
            # 注意这里每次只传入一个时间步的数据，所以需要unsqueeze维度变为 [batch_size, 1]
            output, hidden = self.decoder(input.unsqueeze(1), hidden)
            # output: [batch_size, 1, hidden_size]
            # hidden = (h0, c0)
            # h0, c0: [batch_size, hidden_size]

            outputs[:, t, :] = output
            # 是否采用teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)  # 当前时间步预测的token索引
            input = trg[:, t] if teacher_force else top1

        return outputs
