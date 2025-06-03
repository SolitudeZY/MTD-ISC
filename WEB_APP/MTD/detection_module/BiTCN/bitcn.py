# -*- coding: utf-8 -*-
import json
import sys
import time

from torch import nn
from torch.utils.data import TensorDataset
import argparse
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report

from FocalLoss import FocalLoss
from utils import data_pre_process
from model import TCN
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append("../../")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--Dropout', type=float, default=0.05,
                    help='Dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')  # 设定阈值
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')  # 内核大小
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')  # 层数
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')

parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')  # 初始学习率

parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')  # 使用的优化器

parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')  # 每层隐藏单元数

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')  # 随机种子
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
args = parser.parse_args()

torch.manual_seed(args.seed)  # 随机初始化种子

batch_size = 64  # 一次训练尺寸

input_channels = 3  # 输入通道
seq_length = int(768 / input_channels)  # 序列长度
steps = 0

# torch.Tensor是一种包含单一数据类型元素的多维矩阵。
# Permutation()函数的意思的打乱原来数据中元素的顺序
permute = torch.Tensor(np.random.permutation(768).astype(np.float64)).long()

channel_sizes = [args.nhid] * args.levels
# 一个TCN基本块包含的通道数及层数 这里为[25,25,25,25,25,25,25,25]，即[25]*8
kernel_size = args.ksize
hidden_size = args.nhid


def count_time(func):
    def wrapper(*paras, **kwargs):
        start = time.time()
        result = func(*paras, **kwargs)  # 传递所有参数给原始函数
        end = time.time()
        print(f"\n {func.__name__} takes {end - start} seconds \n")
        return result  # 返回原始函数的结果

    return wrapper  # 返回 wrapper 函数


@count_time
def train(model, train_loader, epoch, loss_function, optimizer, flip_data=False):
    """
    Train the BiTCN for one epoch.

    Parameters:
    - epoch: Current training epoch.
    - flip_data: Whether to flip the input data.
    """
    # train_loader = data_pre_process(os.getcwd(), "4_Png_16_USTC", None)[0]
    # loss_function = torch.nn.CrossEntropyLoss()
    global steps
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target.long()
        data = data.view(-1, input_channels, seq_length)

        if flip_data:
            data = torch.flip(data, dims=[2])

        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target.long())
        loss = loss_function(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss.item()

        steps += seq_length
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_loss = 0


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

    TP = np.sum(cm) - np.sum(cm[normal_index, :]) - np.sum(cm[:, normal_index]) + cm[normal_index, normal_index]
    FN = cm[:, normal_index].sum() - cm[normal_index, normal_index]
    FP = cm[normal_index, :].sum() - cm[normal_index, normal_index]
    TN = cm[normal_index, normal_index]

    # 计算总体TPR和FPR
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}")
    return TPR, FPR


def test(model, validate_loader, dataset_name, flip_data=False):
    """
    Test the BiTCN on the validation set.

    Parameters:
    - flip_data: Whether to flip the input data.
    return
    """
    # validate_loader = data_pre_process(os.getcwd(), "4_Png_16_USTC", None)[1]
    model.eval()
    test_loss = 0
    correct = 0
    forward_output_list = []
    forward_result_list = []
    all_true_labels = []
    all_pred_labels = []
    with torch.no_grad():
        for data, label in validate_loader:
            data, label = data.to(device), label.to(device)
            data = data.view(-1, input_channels, seq_length)

            if flip_data:
                data = torch.flip(data, dims=[2])

            output = model(data)
            forward_output_list.append(output)
            forward_result_list.append(label)
            test_loss += F.nll_loss(output, label.long(), reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

            # 收集所有的真实标签和预测标签
            all_true_labels.extend(label.cpu().numpy())
            all_pred_labels.extend(pred.view(-1).cpu().numpy())

    # 计算混淆矩阵并计算TPR和FPR
    TPR, FPR = calculate_tpr_fpr(json_filepath=dataset_name + 'class_indices.json', true_labels=all_true_labels,
                                 pred_labels=all_pred_labels)
    report = classification_report(all_true_labels, all_pred_labels, zero_division=1, output_dict=True)
    recall = report['macro avg']['recall']
    precision = report['weighted avg']['precision']
    f1_score = report['weighted avg']['f1-score']
    print(f'Precision: {precision:.6f}, '
          f'Recall: {recall:.6f}, '
          f'F1 Score: {f1_score:.6f}, ')

    test_loss /= len(validate_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%), TPR: {:.6f}, FPR: {:.6f}\n'.format(
        test_loss, correct, len(validate_loader.dataset),
        100. * correct / len(validate_loader.dataset), TPR, FPR))
    return test_loss, forward_output_list, forward_result_list, TPR, FPR


def main():
    list_time = []
    directory_path = os.path.join(os.path.abspath(os.getcwd()), "../pre-processing/4_Png_16_ISAC/Train")
    entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    n_classes = folder_count  # 分类数

    for frequency in range(10):
        best_accuracy = 0.0
        print("frequency: ")
        print(frequency)
        lr = 0.001
        train_loader, validate_loader, _, _, _, dataset_name = data_pre_process(os.getcwd(), "4_Png_16_ISAC", None)

        BiTCN = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.Dropout)

        BiTCN.cuda()
        optimizer = optim.Adam(BiTCN.parameters(), lr=lr)
        # loss_function = FocalLoss()
        loss_function = F.nll_loss

        epochs = 2
        for epoch in range(epochs):
            print(f"\nepoch:{epoch + 1} ", end="")
            correct = 0
            total = 0
            target_num = torch.zeros((1, n_classes))
            predict_num = torch.zeros((1, n_classes))
            acc_num = torch.zeros((1, n_classes))

            # 记录时间戳，正向开始训练时间
            train_start = time.time()
            list_time.append(train_start)
            print("forward training")
            train(model=BiTCN, train_loader=train_loader, epoch=epoch, optimizer=optimizer, loss_function=loss_function,
                  flip_data=False)

            # 记录时间戳，正向结束训练时间
            train_end = time.time()
            list_time.append(train_end)

            test_loss, out_list, label_list, TPR, FPR = test(model=BiTCN, validate_loader=validate_loader,
                                                             dataset_name=dataset_name)

            # 记录时间戳，反向开始训练时间
            train_flipped_start = time.time()
            list_time.append(train_flipped_start)
            print("backward training")
            train(model=BiTCN, train_loader=train_loader, epoch=epoch, optimizer=optimizer, loss_function=loss_function,
                  flip_data=True)

            # 记录时间戳，反向结束训练时间
            train_flipped_end = time.time()
            list_time.append(train_flipped_end)

            flx_test_loss, fout_list, all_true_labels, TPR, FPR = test(model=BiTCN, validate_loader=validate_loader,
                                                                       dataset_name=dataset_name,
                                                                       flip_data=True)

            if epoch % 10 == 0:  # 修改学习率
                lr /= 10
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            print("双向训练结果")
            result = []
            loss = 0
            for i in range(len(out_list)):
                result.append(out_list[i] + fout_list[i])
            for j in range(len(label_list)):
                loss += F.nll_loss(result[j], label_list[j].long(), reduction='sum').item()
                pred = result[j].argmax(dim=1, keepdim=True)
                correct += pred.eq(label_list[j].view_as(pred)).sum().item()

                total += label_list[j].size(0)
                pre_mask = torch.zeros(result[j].size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
                predict_num += pre_mask.sum(0)  # TP+FP
                tar_mask = torch.zeros(result[j].size()).scatter_(1, label_list[j].cpu().view(-1, 1).long(), 1.)
                target_num += tar_mask.sum(0)  # TP+FN
                acc_mask = pre_mask * tar_mask
                acc_num += acc_mask.sum(0)  # TP

            test_loss /= len(validate_loader.dataset)
            recall = acc_num / target_num
            precision = acc_num / predict_num
            F1 = 2 * recall * precision / (recall + precision)
            accuracy = acc_num.sum(1) / target_num.sum(1)

            # 精度调整
            recall = (recall.numpy()[0] * 100).round(3)
            precision = (precision.numpy()[0] * 100).round(3)
            F1 = (F1.numpy()[0] * 100).round(3)
            accuracy = (accuracy.numpy()[0] * 100).round(3)

            # 打印格式方便复制
            print('testSize ：{}'.format(len(validate_loader.dataset)))
            print('Recall (%)  ', " ".join('%s' % id for id in recall))
            print('Precision(%)', " ".join('%s' % id for id in precision))
            print('F1 (%)      ', " ".join('%s' % id for id in F1))
            print('TPR(Recall) ', TPR)
            print('FPR         ', FPR)
            print('accuracy(%) ', accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # 保存模型
                save_dir = '../models/ISAC'
                os.makedirs(save_dir, exist_ok=True)
                best_model_filename = os.path.join(save_dir, 'BiTCN_best_ISAC_' + str(frequency) + '.pt')
                torch.save(BiTCN, best_model_filename)
                print(f'Saved new best model with accuracy {best_accuracy:.3f}% as {best_model_filename}')

        # 保存模型
        # 指定保存模型的目录
        save_dir = '../models/ISAC'
        os.makedirs(save_dir, exist_ok=True)
        save_filename = os.path.join(save_dir, 'BiTCN_final_ISAC_' + str(frequency) + '.pt')
        # 保存模型的
        torch.save(BiTCN, save_filename)
        print('Saved as %s' % save_filename)


if __name__ == "__main__":
    main()


def train_model(train_loader, epochs, model, device, class_nums, validate_loader, dataset_name):
    BiTCN = TCN(input_channels, class_nums, channel_sizes, kernel_size=kernel_size, dropout=args.Dropout)
    lr = 0.001
    BiTCN.cuda()
    optimizer = optim.Adam(BiTCN.parameters(), lr=lr)
    # loss_function = FocalLoss()
    loss_function = F.nll_loss

    for epoch in range(epochs):
        print(f"\nepoch:{epoch + 1} ", end="")
        correct = 0
        total = 0
        target_num = torch.zeros((1, class_nums))
        predict_num = torch.zeros((1, class_nums))
        acc_num = torch.zeros((1, class_nums))

        # 记录时间戳，正向开始训练时间
        print("forward training")
        train(model=BiTCN, train_loader=train_loader, epoch=epoch, optimizer=optimizer, loss_function=loss_function,
              flip_data=False)
        test_loss, out_list, label_list, TPR, FPR = test(model=BiTCN, validate_loader=validate_loader,
                                                         dataset_name=dataset_name)

        # 记录时间戳，反向开始训练时间
        print("backward training")
        train(model=BiTCN, train_loader=train_loader, epoch=epoch, optimizer=optimizer, loss_function=loss_function,
              flip_data=True)
        # 记录时间戳，反向结束训练时间
        flx_test_loss, fout_list, all_true_labels, TPR, FPR = test(model=BiTCN, validate_loader=validate_loader,
                                                                   dataset_name=dataset_name,
                                                                   flip_data=True)

        if epoch % 10 == 0:  # 修改学习率
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        print("双向训练结果")
        result = []
        loss = 0
        for i in range(len(out_list)):
            result.append(out_list[i] + fout_list[i])
        for j in range(len(label_list)):
            loss += F.nll_loss(result[j], label_list[j].long(), reduction='sum').item()
            pred = result[j].argmax(dim=1, keepdim=True)
            correct += pred.eq(label_list[j].view_as(pred)).sum().item()

            total += label_list[j].size(0)
            pre_mask = torch.zeros(result[j].size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
            predict_num += pre_mask.sum(0)  # TP+FP
            tar_mask = torch.zeros(result[j].size()).scatter_(1, label_list[j].cpu().view(-1, 1).long(), 1.)
            target_num += tar_mask.sum(0)  # TP+FN
            acc_mask = pre_mask * tar_mask
            acc_num += acc_mask.sum(0)  # TP

        test_loss /= len(validate_loader.dataset)
        recall = acc_num / target_num
        precision = acc_num / predict_num
        F1 = 2 * recall * precision / (recall + precision)
        accuracy = acc_num.sum(1) / target_num.sum(1)

        # 精度调整
        recall = (recall.numpy()[0] * 100).round(3)
        precision = (precision.numpy()[0] * 100).round(3)
        F1 = (F1.numpy()[0] * 100).round(3)
        accuracy = (accuracy.numpy()[0] * 100).round(3)

        # 打印格式方便复制
        print('Recall (%)  ', " ".join('%s' % id for id in recall))
        print('Precision(%)', " ".join('%s' % id for id in precision))
        print('F1 (%)      ', " ".join('%s' % id for id in F1))
        print('TPR(Recall) ', TPR)
        print('FPR         ', FPR)
        print('accuracy(%) ', accuracy)
    return model
