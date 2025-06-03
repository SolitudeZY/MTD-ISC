import json

import torch
import time
import os
import numpy as np
import argparse
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

from tqdm import tqdm
from tcn import TemporalConvNet
from utils import data_pre_process, calculate_tpr_fpr_multiclass
from sklearn.metrics import confusion_matrix, classification_report
from FocalLoss import FocalLoss

parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
# batches_size: 设置每批数据的大小，默认为128。
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
# --cuda: 是否启用CUDA加速，默认为True。
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
# --Dropout: 设置Dropout的概率，默认为0.05。
parser.add_argument('--dropout', type=float, default=0.05,
                    help='Dropout applied to layers (default: 0.05)')
# --clip: 设置梯度裁剪的阈值，默认为-1（表示不裁剪）。
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
# --epochs: 设置训练的最大轮数，默认为10轮。
parser.add_argument('--epochs', type=int, default=5,
                    help='upper epoch limit (default: 5)')
# --ksize: 设置卷积核大小，默认为7。
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
# --levels: 设置卷积层数，默认为8层。
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
# --log-interval 用于设置日志报告的间隔，默认值为100。
parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                    help='report interval (default: 100')
# 学习比率
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 2e-3)')
# 梯度下降算法（优化方法）
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
# 每层隐藏单元数
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer (default: 32)')
# 随机数种子
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
# --permute: 是否使用随机排列的MNIST数据集，默认为false。
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
args = parser.parse_args()

# 设计随机初始化种子，保证初始化都为固定
torch.manual_seed(args.seed)


class TCN_model(nn.Module):
    # 初始化所有的层
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN_model, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    # 定义模型的运算过程(前向传播的过程）
    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)


# torch.Tensor是一种包含单一数据类型元素的多维矩阵。
# Permutation()函数的意思的打乱原来数据中元素的顺序
permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()

input_channels = 3 * 16
seq_length = 16

# 训练数据丢进神经网络次数
# TODO：预计是用于计算当前训练到的位置
steps = 0


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


def train(model, ep, optimizer, train_loader, loss_function, dataset_name):
    print("start training...")
    steps = 0
    train_loss = 0
    # 在train模式下，dropout网络层会按照设定的参数p设置保留激活单元的概率（保留概率=p); batchnorm层会继续计算数据的mean和var等参数并更新
    model.train()
    startTrain = time.perf_counter()
    true_labels, pred_labels = [], []
    log_interval = 800
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.cuda(), target.cuda()  # data为img, target为label
        # view是改变tensor的形状，view中的-1是自适应的调整
        data = data.view(-1, input_channels, seq_length)
        # 调换Tensor中各维度的顺序
        if args.permute:
            data = data[:, :, permute]
        # 变量梯度设置为0
        optimizer.zero_grad()
        output = model(data)
        # 计算损失数值
        loss = loss_function(output, target)
        # F.nll_loss(output, target)
        # 进行反向传播
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # 进行梯度优化
        optimizer.step()
        # 计算损失值
        train_loss += loss.item()  # 使用loss.item()获取标量值
        # 更新步数
        steps += data.size(0)  # 基于批次大小更新步数
        # 将真实标签添加到列表中
        true_labels.extend(target.cpu().numpy())
        # 获取预测标签（最高得分对应的索引）
        _, predicted = torch.max(output.data, 1)
        pred_labels.extend(predicted.cpu().numpy())

        # 判断是否到达判断间隔组
        if batch_idx > 0 and batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * data.size(0), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), train_loss / log_interval, steps))
            train_loss = 0

    # TPR, FPR = calculate_tpr_fpr(dataset_name + "class_indices.json", true_labels=true_labels,
    # pred_labels=pred_labels)
    TPR, FPR = calculate_tpr_fpr_multiclass(true_labels, pred_labels, n_classes)
    print(f"TPR : {TPR}, FPR : {FPR}")
    trainTime = (time.perf_counter() - startTrain)
    print("trainTime:", trainTime)


def test(model, test_loader, class_nums, dataset_name):
    print("start testing", end="\n")
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    target_num = torch.zeros((1, class_nums))
    predict_num = torch.zeros((1, class_nums))
    acc_num = torch.zeros((1, class_nums))
    startTest = time.perf_counter()

    # 初始化累积列表
    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        conf_matrix = torch.zeros(class_nums, class_nums)
        for data, label in test_loader:  # target就是label
            if args.cuda:
                data, label = data.cuda(), label.cuda()
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            # data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, label, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
            total += label.size(0)

            # 累积 true_labels 和 pred_labels
            all_true_labels.extend(label.cpu().numpy().flatten().tolist())
            all_pred_labels.extend(pred.cpu().numpy().flatten().tolist())

            pre_mask = torch.zeros(output.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
            predict_num += pre_mask.sum(0)
            tar_mask = torch.zeros(output.size()).scatter_(1, label.data.cpu().view_as(pred), 1.)
            target_num += tar_mask.sum(0)
            acc_mask = pre_mask * tar_mask
            acc_num += acc_mask.sum(0)

            # 使用 target 和 pred 计算混淆矩阵
            # conf_matrix += torch.tensor(
            #     confusion_matrix(target.cpu().numpy(), pred.cpu().numpy(), labels=list(range(classnum))))

        test_loss /= len(test_loader.dataset)
        recall = acc_num / target_num
        precision = acc_num / predict_num
        F1 = 2 * recall * precision / (recall + precision)
        accuracy = acc_num.sum(1) / target_num.sum(1)
        recall = (recall.numpy()[0] * 100).round(3)
        precision = (precision.numpy()[0] * 100).round(3)
        F1 = (F1.numpy()[0] * 100).round(3)
        accuracy = (accuracy.numpy()[0] * 100).round(3)

        # 计算 TPR 和 FPR
        true_labels_tensor = torch.tensor(all_true_labels).cpu()
        pred_labels_tensor = torch.tensor(all_pred_labels).cpu()

        # 将张量从 GPU 移到 CPU 并转换为 NumPy 数组
        true_labels_cpu = true_labels_tensor.numpy()
        pred_labels_cpu = pred_labels_tensor.numpy()
        # print(f'target.cpu().numpy(): {label.cpu().numpy()}, pred.cpu().numpy(): {pred.cpu().numpy()}')
        TPR, FPR = calculate_tpr_fpr(dataset_name + 'class_indices.json', true_labels=true_labels_cpu,
                                     pred_labels=pred_labels_cpu)
        tpr, fpr = calculate_tpr_fpr_multiclass(true_labels_cpu, pred_labels_cpu, n_classes)
        print(f'Total TPR: {TPR}, FPR: {FPR}')
        print(f"weighted tpr : {tpr};  fpr : {fpr}\n")

        testTime = (time.perf_counter() - startTest)
        print("testTime:", testTime)
        print('testSize: {}'.format(len(test_loader.dataset)))
        print('recall：', " ".join('%s' % id for id in recall))
        print('precision：', " ".join('%s' % id for id in precision))
        print('F1：', " ".join('%s' % id for id in F1))
        print('accuracy：', accuracy)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        #
        # # 输出所有指标在同一行
        # print('\n Recall:', " ".join('%s' % id for id in recall), "\n",
        #       'Precision:', " ".join('%s' % id for id in precision), "\n",
        #       'F1:', " ".join('%s' % id for id in F1), "\n",
        #       'Accuracy :', accuracy)

        report = classification_report(true_labels_cpu, pred_labels_cpu, output_dict=True, zero_division=1)
        print("accuracy:", report['accuracy'])
        print("F1:", report.get('weighted avg').get('f1-score'))
        print("recall:", report.get('weighted avg').get('recall'))

        return accuracy


def get_model(num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置通道数 = 隐藏层数*每层隐藏单元数
    channel_sizes = [args.nhid] * args.levels  # [32] * 8
    # 卷积核大小
    kernel_size = args.ksize
    batch_size = args.batch_size

    model = TCN_model(input_channels, num_classes, channel_sizes, kernel_size=kernel_size, dropout=0.05)
    lr = 0.001
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # loss_function = FocalLoss()
    model = model.to(device)
    return model


def main():
    EP = 10
    num_epochs = 5
    for ep in range(1):
        print('EP {}/{}'.format(ep + 1, EP))
        best_acc = 0.0
        train_loader, test_loader, _, _, _, dataset_name = data_pre_process(os.getcwd(), PNG_PATH, None)
        # 设置损失函数、梯度和学习率
        lr = 0.001
        model = TCN_model(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=0.01)
        # optimizer = nn.optm.SGD(model.parameters(), lr=lr)  # getattr(optim, 'Adam')
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # loss_function = nn.CrossEntropyLoss()
        # loss_function = F.nll_loss
        loss_function = FocalLoss(alpha=4, gamma=0.75)
        model = model.to(device)

        for epoch in range(num_epochs):  # train model
            print(f"Epoch: {epoch}")
            train(ep=epoch, train_loader=train_loader, optimizer=optimizer, model=model, loss_function=loss_function,
                  dataset_name=dataset_name)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            accuracy = test(model=model, test_loader=test_loader, class_nums=n_classes, dataset_name=dataset_name)

            # 保存最佳模型
        #     save_dir = './models/CTU'
        #     if best_acc < accuracy:
        #         os.makedirs(save_dir, exist_ok=True)
        #         save_file = 'TCN_model_best_' + dataset_name + str(ep) + ".pt"
        #         print('Saving at best accuracy of ', save_file)
        #         save_filename = os.path.join(save_dir, save_file)
        #         torch.save(model, save_filename)
        # save_filename = './models/TCN_model_final_' + dataset_name + str(ep) + '.pt'
        # torch.save(model, save_filename)
        # print('Saved as %s' % save_filename)


if __name__ == "__main__":
    # PNG_PATH = "4_Png_16_CTU"
    PNG_PATH = "4_Png_16_ISAC"

    # 设置通道数 = 隐藏层数*每层隐藏单元数
    channel_sizes = [args.nhid] * args.levels  # [32] * 8
    # 卷积核大小
    kernel_size = args.ksize
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 由训练集路径计算类别数量
    directory_path = os.path.join(os.path.abspath(os.getcwd()), f"./pre-processing/{PNG_PATH}/Train")
    entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    n_classes = folder_count
    model = get_model(folder_count)
    main()
