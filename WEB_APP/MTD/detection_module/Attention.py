import torch
import torch.nn as nn
import math


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        """
        :param hidden: 解码器的隐藏状态 (num_layers, batch_size, hidden_size)
        :param encoder_outputs: 编码器的所有隐藏状态 (batch_size, sequence_length, hidden_size)
        :return: 上下文向量 (batch_size, hidden_size)
        """
        # 调整 hidden 的维度顺序以匹配 encoder_outputs
        # print('hidden: ', hidden)
        hidden = hidden.permute(1, 0, 2)  # 调整为 (batch_size, num_layers, hidden_size)
        hidden = hidden[-1]  # 取最后一层的隐藏状态 (batch_size, hidden_size)

        # 重复 hidden 以匹配 encoder_outputs 的长度
        hidden = hidden.unsqueeze(1)  # 调整为 (batch_size, 1, hidden_size)
        hidden = hidden.repeat(1, encoder_outputs.size(1), 1)  # 调整为 (batch_size, sequence_length, hidden_size)

        # 计算注意力权重
        energy = torch.tanh(
            self.attn(torch.cat([hidden, encoder_outputs], 2)))  # (batch_size, sequence_length, hidden_size)
        energy = energy.transpose(2, 1)  # (batch_size, hidden_size, sequence_length)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # (batch_size, 1, hidden_size)
        attn_energies = torch.bmm(v, energy).squeeze(1)  # (batch_size, sequence_length)

        # 应用 softmax 函数计算注意力权重
        attn_weights = torch.softmax(attn_energies, dim=1)  # (batch_size, sequence_length)

        # 使用注意力权重对 encoder_outputs 进行加权求和
        context = attn_weights.unsqueeze(1).bmm(encoder_outputs)  # (batch_size, 1, hidden_size)
        context = context.squeeze(1)  # (batch_size, hidden_size)

        return context
