import torch.nn as nn
import torch
import math


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, output):
        """
        :param output: RNN 的所有时间步输出 (batch_size, sequence_length, hidden_size)
        :return: 加权后的上下文向量 (batch_size, hidden_size)
        """
        # 计算能量值
        energy = torch.tanh(self.attn(output))  # (batch_size, sequence_length, hidden_size)
        energy = energy.permute(0, 2, 1)  # (batch_size, hidden_size, sequence_length)

        # 计算注意力权重
        v = self.v.repeat(output.size(0), 1).unsqueeze(1)  # (batch_size, 1, hidden_size)
        attn_energies = torch.bmm(v, energy).squeeze(1)  # (batch_size, sequence_length)

        # 应用 softmax 函数计算注意力权重
        attn_weights = torch.softmax(attn_energies, dim=1)  # (batch_size, sequence_length)

        # 使用注意力权重对输出进行加权求和
        context = attn_weights.unsqueeze(1).bmm(output)  # (batch_size, 1, hidden_size)
        context = context.squeeze(1)  # (batch_size, hidden_size)

        return context
