# -*- coding: utf-8 -*-
import json
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from FocalLoss import FocalLoss
from utils import data_pre_process
from Attention import Attention

# torch.manual_seed(1)    # reproducible
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper Parameters
sequence_length = 16
input_size = 3 * 16 * 16
hidden_size = 128
num_layers = 2

# num_classes = 12  # TODO 根据数据集类型改变

batch_size = 128


class BIRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(BIRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.elu = nn.ELU()  # 添加 ELU 激活函数

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)

        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 3 * 16 * 16)  # X变为(batch_size, seq_len, input_size)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])
        out = self.elu(out)  # 应用 ELU 激活函数
        return out


# class BIRNNWithAttention(nn.Module):
#     def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
#         super(BIRNNWithAttention, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.layer_dim = layer_dim
#         self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_dim * 2, output_dim)
#         self.elu = nn.ELU()  # 添加 ELU 激活函数
#         self.attention = Attention(hidden_dim)  # 添加注意力机制
#
#     def forward(self, x):
#         h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)
#         c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)
#
#         x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 3 * 16 * 16)
#
#         out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
#
#         # 应用注意力机制
#         # 使用最后一个隐藏状态作为解码器的隐藏状态
#         # 注意：这里我们使用最后一个隐藏状态 hn[-1] 作为解码器的隐藏状态
#         # 这是因为在双向 LSTM 中，hn[-1] 包含了整个序列的信息
#         context = self.attention(hn[-1].unsqueeze(1), out)  # 应用注意力机制
#
#         # 使用上下文向量作为特征表示
#         out = self.fc(context)
#         out = self.elu(out)  # 应用 ELU 激活函数
#         return out


def calculate_tpr_fpr(json_filepath, true_labels, pred_labels):
    """计算总体TPR和FPR，使用json文件中的'Normal'类别索引"""
    # 从json文件中加载类别索引
    with open(json_filepath, 'r') as json_file:
        cla_dict = json.load(json_file)
        # 找到值为 "Normal" 的索引
        normal_index = None
        for idx, class_name in cla_dict.items():
            if class_name == "Normal":
                normal_index = int(idx)
                print(f"Found 'Normal' class at index: {idx}")
                break
        if normal_index is None:
            raise ValueError("Class 'Normal' not found in the JSON file.")
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)

    # 真正例（TP）是恶意流量被正确预测的总和（除了'Normal'索引的行和列）
    TP = np.sum(cm) - np.sum(cm[normal_index, :]) - np.sum(cm[:, normal_index]) + cm[normal_index, normal_index]
    # 假负例（FN）是恶意流量错误预测为'Normal'的数量
    FN = cm[:, normal_index].sum() - cm[normal_index, normal_index]
    # 假正例（FP）是'Normal'错误预测为恶意流量的数量
    FP = cm[normal_index, :].sum() - cm[normal_index, normal_index]
    # 真负例（TN）是'Normal'被正确预测的数量
    TN = cm[normal_index, normal_index]

    # 计算总体TPR和FPR
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}")
    return TPR, FPR


if __name__ == '__main__':

    directory_path = os.path.join(os.path.abspath(os.getcwd()), "../pre-processing/4_Png_16_CTU/Train")
    entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    num_classes = folder_count

    # 初始化模型和优化器
    # BiLSTM = BIRNNWithAttention(input_size, hidden_size, num_layers, num_classes).to(DEVICE)
    learning_rate = 0.003

    best_acc = 0.0
    EP = 10
    num_epochs = 6
    for ep in range(1,EP):
        BiLSTM = BIRNN(input_size, hidden_size, num_layers, num_classes).to(DEVICE)
        # 使用FocalLoss损失函数
        criterion = FocalLoss(alpha=0.75, gamma=4).to(DEVICE)  # 使用自定义的alpha和gamma
        # 使用交叉熵损失函数
        # criterion = nn.CrossEntropyLoss().to(DEVICE)
        optimizer = torch.optim.Adagrad(BiLSTM.parameters(), lr=learning_rate)

        # 生成数据加载器
        train_loader, validate_loader, _, _, _, dataset_name = data_pre_process(os.path.join(os.getcwd(), "../"),
                                                                                "4_Png_16_CTU",
                                                                                None)
        for epoch in range(num_epochs):
            best_acc = 0.0
            print("Epoch_" + str(epoch + 1))
            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=True)

            for i, (images, labels) in enumerate(pbar):
                images = images.squeeze(1).float().to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = BiLSTM(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                pbar.set_description(f'Epoch {epoch + 1}/{num_epochs} Loss: {loss.item():.4f}')
            # Test the Model
            # 测试模型
            with torch.no_grad():
                test_loss = 0
                correct = 0
                total = 0
                target_num = torch.zeros((1, num_classes)).to(DEVICE)
                predict_num = torch.zeros((1, num_classes)).to(DEVICE)
                acc_num = torch.zeros((1, num_classes)).to(DEVICE)
                # 新增变量用于统计TP, FP, TN, FN
                tp = torch.zeros((1, num_classes)).to(DEVICE)
                fp = torch.zeros((1, num_classes)).to(DEVICE)
                tn = torch.zeros((1, num_classes)).to(DEVICE)
                fn = torch.zeros((1, num_classes)).to(DEVICE)

                # 新增变量用于累积所有的真实标签和预测标签
                all_labels = []
                all_pred = []

                for images, labels in validate_loader:
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)

                    outputs = BiLSTM(images)
                    test_loss += criterion(outputs, labels).item() * images.size(0)

                    # 确保 pred 在与 outputs 相同的设备上
                    pred = outputs.argmax(dim=1, keepdim=True).to(DEVICE)
                    correct += pred.eq(labels.view_as(pred)).sum().item()

                    # 更新 total 的值
                    total += labels.size(0)  # 更新 total 的值

                    pre_mask = torch.zeros(outputs.size()).to(DEVICE).scatter_(1, pred, 1.)
                    predict_num += pre_mask.sum(0)
                    tar_mask = torch.zeros(outputs.size()).to(DEVICE).scatter_(1, labels.view_as(pred), 1.)
                    target_num += tar_mask.sum(0)
                    acc_mask = pre_mask * tar_mask
                    acc_num += acc_mask.sum(0)
                    # 将当前批次的真实标签和预测标签添加到累积列表中
                    all_labels.extend(labels.view(-1).tolist())
                    all_pred.extend(pred.view(-1).tolist())

                    # 计算TP, FP, TN, FN
                    for i in range(num_classes):
                        tp[0][i] += ((pred == i) * (labels == i)).sum()
                        fp[0][i] += ((pred == i) * (labels != i)).sum()
                        fn[0][i] += ((pred != i) * (labels == i)).sum()
                        # 计算TN时，需要排除所有预测为当前类别的样本
                        tn[0][i] += ((pred != i) * (labels != i)).sum()

                test_loss /= len(validate_loader.dataset)

                # 计算 recall, precision 和 F1-score 时，加入一个小的正数来避免除以零的情况
                epsilon = 1e-7
                recall = acc_num / (target_num + epsilon)
                precision = acc_num / (predict_num + epsilon)
                F1 = 2 * recall * precision / (recall + precision + epsilon)
                accuracy = acc_num.sum(1) / (target_num.sum(1) + epsilon)

                # 计算TPR和FPR
                tpr = tp / (tp + fn + epsilon)  # TPR = TP / (TP + FN)
                fpr = fp / (fp + tn + epsilon)  # FPR = FP / (FP + TN)
                # 计算所有数据的FPR/TPR
                # 使用累积的所有真实标签和预测标签计算混淆矩阵
                all_labels_tensor = torch.tensor(all_labels).to(DEVICE)
                all_pred_tensor = torch.tensor(all_pred).to(DEVICE)

                # 将张量从 GPU 移到 CPU 并转换为 NumPy 数组
                all_labels_cpu = all_labels_tensor.cpu().numpy()
                all_pred_cpu = all_pred_tensor.cpu().numpy()
                # print(f"target: {all_labels_tensor}, predicted: {all_pred_tensor}")
                TPR, FPR = calculate_tpr_fpr(json_filepath=dataset_name+"class_indices.json",
                                             true_labels=all_labels_cpu,
                                             pred_labels=all_pred_cpu)
                # 精度调整
                recall = (recall.cpu().numpy()[0] * 100).round(5)
                precision = (precision.cpu().numpy()[0] * 100).round(5)
                F1 = (F1.cpu().numpy()[0] * 100).round(5)
                accuracy = (accuracy.cpu().numpy()[0] * 100).round(5)
                tpr = (tpr.cpu().numpy()[0] * 1).round(5)
                fpr = (fpr.cpu().numpy()[0] * 1).round(5)

                # 打印格式方便复制
                print('testSize : {}'.format(len(validate_loader.dataset)))
                print('recall（%）   ', " ".join('%s' % id for id in recall))
                print('precision（%）', " ".join('%s' % id for id in precision))
                print('F1（%）       ', " ".join('%s' % id for id in F1))
                print('accuracy（%） ', accuracy)
                print('Test Accuracy of the BiLSTM on the  test images: %d %%' % (100 * correct / total))
                print('TPR          ', " ".join('%s' % id for id in tpr))
                print('FPR          ', " ".join('%s' % id for id in fpr))
                print(f"Total TPR:{TPR:.6f}, Total FPR:{FPR:.6f}")
            if best_acc <= accuracy:
                best_acc = accuracy
                save_filename = "../models/CTU/" + "BiLSTM_best_CTU_" + str(ep) + ".pt"
                torch.save(BiLSTM, save_filename)
                print("saving best model at epoch: ", epoch,
                      "\nsaving at", save_filename)
        # 指定保存模型的目录
        save_dir = '../models/CTU'
        os.makedirs(save_dir, exist_ok=True)
        # save_filename = os.path.join(save_dir, 'lstm-model.pt')
        save_filename = os.path.join(save_dir, 'BiLSTM_final_CTU_' + str(ep) + '.pt')
        # 保存模型
        torch.save(BiLSTM, save_filename)
        print('Saved as %s' % save_filename)
