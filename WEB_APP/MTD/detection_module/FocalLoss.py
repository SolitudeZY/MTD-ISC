import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        # 计算原始的交叉熵损失
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 获取预测概率
        pt = torch.exp(-BCE_loss)

        # 计算焦点损失
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # 根据 reduction 参数返回损失
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


# Train the Model



