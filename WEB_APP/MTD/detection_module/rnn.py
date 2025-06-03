import json
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm
from utils import data_pre_process, calculate_tpr_fpr, calculate_tpr_fpr_multiclass
from collections import Counter
from FocalLoss import FocalLoss

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


# class RNN(nn.Module):
#     def __init__(self, num_Classes, input_size):
#         super(RNN, self).__init__()
#         self.rnn = nn.RNN(
#             input_size=input_size,  # 输入的图像尺寸为32*32的RGB图像
#             hidden_size=64,
#             num_layers=1,
#             batch_first=True,
#         )
#         self.attention = Attention(hidden_size=64)  # 添加注意力机制
#         self.out_layer = nn.Linear(64, num_Classes)  # 输出层
#
#     def forward(self, x):
#         output, _ = self.rnn(x)
#         # 使用注意力机制
#         # attn_output = self.attention(output)
#         prediction = self.out_layer(attn_output)
#         return prediction


class RNN(nn.Module):
    def __init__(self, num_classes, input_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
        )
        self.elu = nn.ELU()  # 添加ELU层
        self.dropout = nn.Dropout(0.01)  # 添加Dropout层
        # self.attention = Attention(hidden_size=64)  # 添加注意力机制
        self.out_layer = nn.Linear(64, num_classes)  # 输出层

    def forward(self, x):
        output, _ = self.rnn(x)
        # 关注最后一个时间步的输出
        last_output = output[:, -1, :]
        elu_output = self.elu(last_output)  # 应用ELU激活函数
        dropout_output = self.dropout(elu_output)  # 应用Dropout层

        prediction = self.out_layer(dropout_output)
        return prediction


def prepare_data(batch_images, batch_labels, input_size):
    # 准备数据
    batch_images = batch_images.squeeze(1)  # 去掉多余的维度
    batch_images = batch_images.float().to(device)  # 转换为浮点型并移到GPU
    batch_labels = batch_labels.to(device)
    # 展平为 [batches_size, seq_len, Input_size]
    batch_size = batch_images.size(0)
    seq_len = 16  # 图像的高度或宽度
    input_size = input_size  # 输入的维度 16*16*3
    batch_images = batch_images.permute(0, 2, 3, 1).contiguous()  # 调整维度顺序
    batch_images = batch_images.view(batch_size, seq_len, input_size)
    return batch_images, batch_labels


def get_accuracy(model, data_loader, num_class, dataset_name: str):
    model.eval()
    predictions = []
    true_labels = []
    correct = 0
    total = 0
    target_counts = torch.zeros(num_class).to(next(model.parameters()).device)  # 存储每个类别的实际出现次数
    predicted_counts = torch.zeros(num_class).to(next(model.parameters()).device)  # 存储每个类别的预测出现次数
    accurate_counts = torch.zeros(num_class).to(next(model.parameters()).device)  # 存储每个类别正确预测的次数

    total_samples = len(data_loader.dataset)
    start_time = time.perf_counter()
    with torch.no_grad():
        for image, label in data_loader:
            image, label = prepare_data(image, label, input_size=16 * 3)
            output = model(image)
            _, predicted = torch.max(output.data, 1)

            total += label.size(0)
            correct += (predicted == label).sum().item()
            predictions.extend(predicted.cpu().numpy())
            # 收集预测标签和真实标签
            true_labels.extend(label.cpu().numpy())
            # 更新计数器
            predicted_mask = torch.zeros(output.size()).to(next(model.parameters()).device).scatter_(1,
                                                                                                     predicted.unsqueeze(
                                                                                                         1),
                                                                                                     1.).squeeze()
            predicted_counts += predicted_mask.sum(0)
            target_mask = torch.zeros(output.size()).to(next(model.parameters()).device).scatter_(1, label.unsqueeze(1),
                                                                                                  1.).squeeze()
            target_counts += target_mask.sum(0)
            accurate_mask = predicted_mask * target_mask
            accurate_counts += accurate_mask.sum(0)

    test_time = time.perf_counter() - start_time
    print("testTime:", test_time)

    # 确保分母不为零
    eps = 1e-7  # 一个很小的正数，用来防止除以零
    # 计算指标
    recall = accurate_counts / (target_counts + eps)
    precision = accurate_counts / (predicted_counts + eps)
    f1_score = 2 * recall * precision / (recall + precision + eps)
    accuracy = accurate_counts.sum() / (target_counts.sum() + eps)

    # 计算每个类别的真负样本数
    true_negative_counts = total_samples - (predicted_counts + (target_counts - accurate_counts))
    true_negative_counts = torch.max(true_negative_counts, torch.tensor([eps]).to(next(model.parameters()).device))

    # FP 假阳性的次数
    false_positive_counts = predicted_counts - accurate_counts

    # 计算每个类别的假阳性率（FPR）
    fpr = false_positive_counts / (false_positive_counts + true_negative_counts + eps)

    TPR, FPR = calculate_tpr_fpr_multiclass(true_labels, predictions, num_classes)
    # 精度调整
    recall = (recall.cpu().numpy() * 100).round(3)
    precision = (precision.cpu().numpy() * 100).round(3)
    f1_score = (f1_score.cpu().numpy() * 100).round(3)
    accuracy = (accuracy.cpu().numpy() * 100).round(3)

    # 打印格式方便复制
    print('recall(TPR) (%)', " ".join(f'{value:.5f}' for value in recall))
    print('precision (%)  ', " ".join(f'{value:.5f}' for value in precision))
    print('F1 score  (%)  ', " ".join(f'{value:.5f}' for value in f1_score))
    print('accuracy  (%)  ', accuracy)
    print(" TPR:", TPR, '\t FPR:', FPR)
    acc = correct / total_samples
    print('correct={}, Test ACC:{:.5f}'.format(correct, acc))

    report = classification_report(true_labels, predictions, output_dict=False)
    print(report)
    report_dict = classification_report(true_labels, predictions, output_dict=True)
    f1_score = report_dict['weighted avg']['f1-score']
    ACC = report_dict['accuracy']
    print(f'Accuracy: {ACC:.4f}')
    print(f'F1-score: {f1_score:.4f}')

    return acc


def save_model(model, model_name: str = 'rnn-model.pt', save_dir='models'):
    # 指定保存模型的目录   save_dir
    # 创建目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    # 构建完整的文件路径
    save_file = os.path.join(save_dir, model_name)
    print("model saved as", save_file)
    # 保存模型
    torch.save(model, save_file)


if __name__ == "__main__":
    PNG_PATH = '4_Png_16_USTC'
    # 获得文件夹数量（分类数量）
    directory_path = os.path.join(os.path.abspath(os.getcwd()), F"./pre-processing/{PNG_PATH}/Train")
    entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    num_classes = folder_count
    # 训练轮数
    epochs = 5
    Input_Size = 3 * 16
    EP = 10
    train_loader, test_loader, _, _, _, cla_name = data_pre_process(os.getcwd(), PNG_PATH)
    for ep in range(1):  # TRAIN
        print(f"第{ep + 1}次训练模型")
        model = RNN(num_classes, input_size=Input_Size)
        model = model.to(device)
        # 定义优化器和损失函数

        # 使用加权交叉熵损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # weight_decay=1e-4 权重衰减系数，相当于L2正则化项的系数，有助于防止过拟合。
        # criterion = nn.CrossEntropyLoss()
        criterion = FocalLoss(gamma=2, alpha=0.75)

        best_accuracy = 0.0  # 初始化最佳准确率为0
        # 训练模型
        model.train()
        for current_epoch in range(epochs):  # train
            print('epoch:{}'.format(current_epoch + 1))
            running_loss = 0.0  # 累积loss
            start_train_time = time.perf_counter()
            pbar = tqdm(train_loader, desc=f'Epoch {current_epoch + 1}/{epochs} ')
            # 使用tqdm显示训练进度
            for images, labels in pbar:
                images, labels = prepare_data(images, labels, input_size=16 * 3)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 累积loss值
                running_loss += loss.item()
                # if (batch_idx + 1) % 200 == 0:
                #     print(f'epoch:{current_epoch + 1}, batch:{batch_idx + 1}, loss:{running_loss / (batch_idx + 1)}')
                pbar.set_postfix(loss=loss.item())
            avg_loss = running_loss / len(train_loader)
            train_time = time.perf_counter() - start_train_time
            print(f"Epoch {current_epoch + 1} average loss: {avg_loss:.4f}")
            print("trainTime:", train_time)

            # 测试模型
            model.eval()  # 设置模型为评估模式

            acc = get_accuracy(model, test_loader, num_classes, dataset_name=cla_name)

            model.train()  # 重新设置模型为训练模式

            # 保存最佳模型
            # if acc > best_accuracy:
            #     best_accuracy = acc
            #     save_model(model=model, model_name='RNN_best_' + cla_name + str(ep) + '.pt',
            #                save_dir='./models/ISAC')
            #     print("Model saved with best accuracy: {:.6f}".format(best_accuracy))

            # 保存最终模型
        # save_model(model=model, model_name='RNN_final_' + cla_name + str(ep) + '.pt', save_dir='./models/ISAC')

# 用于计算RGB图的均值和标准差，加快训练速度，避免每次都计算
# def calculate_weighted_mean_and_std(dataset, class_weights):
#     # 初始化均值和标准差
#     mean = torch.zeros(3)  # 对于RGB图像
#     std = torch.zeros(3)
#     total_images = 0
#
#     # 使用 DataLoader 遍历数据集
#     data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
#
#     for images, labels in tqdm(data_loader, desc="Calculating weighted mean and std"):
#         batch_samples = images.size(0)  # 当前批次的图像数量
#         images = images.view(batch_samples, images.size(1), -1)  # 将图像展平为 (batch_size, channels, height * width)
#
#         # 计算每个通道的均值和标准差
#         for c in range(3):  # 对于每个通道
#             weighted_mean = (images[:, c, :].mean(dim=1) * class_weights[labels]).sum()
#             mean[c] += weighted_mean.item()
#
#         for c in range(3):  # 对于每个通道
#             weighted_std = (images[:, c, :].std(dim=1) * class_weights[labels]).sum()
#             std[c] += weighted_std.item()
#
#         total_images += batch_samples
#
#     mean /= total_images  # 平均化均值
#     std /= total_images  # 平均化标准差
#
#     return mean, std
#
#
# def data_load(device):
#     print("torch.cuda.is_available():", torch.cuda.is_available())
#     print("device:", device)
#     print("using {} device.".format(device))
#
#     # 数据集路径
#     data_root = os.path.abspath(os.getcwd())
#     image_path = os.path.join(data_root, "pre-processing", "4_Png_16_CIC")
#     assert os.path.exists(image_path), f"{image_path} path does not exist."
#
#     # 创建训练集
#     train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=transforms.ToTensor())
#
#     # 计算各个类别的样本数量
#     class_counts = np.bincount(train_dataset.targets)
#     total_samples = len(train_dataset)
#
#     # 计算类别权重
#     class_weights = total_samples / (len(class_counts) * class_counts)
#
#     # 计算均值和标准差
#     mean, std = calculate_weighted_mean_and_std(train_dataset, class_weights)
#     print(f"Calculated Mean: {mean.tolist()}, Calculated Std: {std.tolist()}")
#
#     data_transforms = {
#         "train": transforms.Compose([
#             transforms.Resize((16, 16)),  # 确保提供的是高度和宽度
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean.tolist(), std.tolist())  # 使用计算出的均值和标准差
#         ]),
#         "test": transforms.Compose([
#             transforms.Resize((16, 16)),
#             transforms.CenterCrop(16),
#             transforms.ToTensor(),
#             transforms.Normalize(mean.tolist(), std.tolist())
#         ])
#     }
#
#     batch_size = 64
#     num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
#     print('Using {} dataloader workers every process'.format(num_workers))
#
#     # 使用 WeightedRandomSampler 进行过采样
#     weights = class_weights[train_dataset.targets]
#     sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
#
#     Train_Loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
#     test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"), transform=data_transforms["test"])
#     Test_Loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     val_num = len(test_dataset)
#
#     print("using {} images for training, {} images for validation.".format(len(train_dataset), val_num))
#     return Train_Loader, Test_Loader
