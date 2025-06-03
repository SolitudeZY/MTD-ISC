import csv
from os.path import exists

import sklearn.metrics
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn, autocast
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
from sklearn.metrics import classification_report
from tqdm import tqdm
import tcn_train_and_test
from FocalLoss import FocalLoss
from tcn import TemporalConvNet
import os
import numpy as np
import argparse
import time
from torch.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, recall_score, f1_score
from utils import data_pre_process, calculate_tpr_fpr, calculate_tpr_fpr_multiclass
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
# batches_size: 设置每批数据的大小，默认为128。
parser.add_argument('--batches_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
# --cuda: 是否启用CUDA加速，默认为True。
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
# --Dropout: 设置Dropout的概率，默认为0.05。
parser.add_argument('--Dropout', type=float, default=0.05,
                    help='Dropout applied to layers (default: 0.05)')
# --clip: 设置梯度裁剪的阈值，默认为-1（表示不裁剪）。
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
# --epochs: 设置训练的最大轮数，默认为10轮。
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit (default: 10)')
# --ksize: 设置卷积核大小，默认为7。
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
# --levels: 设置卷积层数，默认为8层。
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
# --log-interval 用于设置日志报告的间隔，默认值为100。
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
# 学习比率
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
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

# TODO 超参数定义

num_hid = 32  # 每层隐藏单元数
levels = 8  # 卷积层数
channel_sizes = [num_hid] * levels  # 设置通道数 = 隐藏层数*每层隐藏单元数
kernel_size = 7  # 卷积核大小
dropout = 0.05
clip = -1  # 设置梯度裁剪的阈值，默认为-1（表示不裁剪）。
# torch.Tensor是一种包含单一数据类型元素的多维矩阵。
permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()

# 每次扔进神经网络训练的数据个数
batch_size = 128
input_channels = 3 * 16
seq_length = 16
# 训练数据丢进神经网络次数
epochs = 10
# TODO：预计是用于计算当前训练到的位置
steps = 0

if torch.cuda.is_available():
    permute = permute.cuda()


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


def fit(stacking_train: np.ndarray, labels: np.ndarray, input_size: int, num_classes: int, dataset_name, epochs=5,
        device='cuda', Epoch=10):
    # 训练模型
    # 将数据转换为 PyTorch 张量
    stacking_train_tensor = torch.tensor(stacking_train, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

    # 创建数据集和数据加载器
    dataset = TensorDataset(stacking_train_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    scaler = GradScaler()  # 创建 GradScaler 对象
    # dataset_name = dataset_name[:-1]

    for ep in range(0, Epoch):
        print(f'L Epoch: {ep}')
        model = TCN_model(input_size, num_classes, channel_sizes, kernel_size, dropout).to(device)

        # 定义损失函数和优化器
        # criterion = nn.CrossEntropyLoss()
        criterion = FocalLoss(gamma=4, alpha=0.8)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
        # optimizer = optim.Adamax(model.parameters(), lr=0.0005)
        best_acc = 0.0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            model.train()
            running_loss = 0.0
            # 使用 tqdm 包裹 dataloader
            for inputs, targets in tqdm(dataloader, desc=f'Training Epoch {epoch + 1}/{epochs}', leave=False):
                optimizer.zero_grad()

                with autocast("cuda"):  # 使用 autocast 上下文管理器
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()  # 缩放损失并反向传播
                scaler.step(optimizer)  # 更新权重
                scaler.update()  # 更新缩放因子

                running_loss += loss.item()

            print(f'\n Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f} \n')

            # 评估模型
            model.eval()
            all_outputs = []
            all_labels = []
            all_predictions = []
            print('validating TCN')
            with torch.no_grad():
                # 使用 tqdm 包裹 dataloader
                for inputs, targets in tqdm(dataloader, desc=f'validating Epoch {epoch + 1}/{epochs}', leave=False):
                    with autocast("cuda"):  # 使用 autocast 上下文管理器
                        outputs = model(inputs)
                        probs = F.softmax(outputs, dim=1)
                        all_outputs.append(probs.cpu().numpy())
                        all_labels.append(targets.cpu().numpy())

                        # 获取每个样本最大概率对应的类别索引
                        predictions = torch.argmax(outputs, dim=1)
                        all_predictions.append(predictions.cpu().numpy())

            all_outputs = np.concatenate(all_outputs, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            all_predictions = np.concatenate(all_predictions, axis=0)

            # 计算准确率
            correct_predictions = (all_predictions == all_labels).sum()
            accuracy = correct_predictions / len(all_labels)

            # 计算其他指标
            TPR, FPR = calculate_tpr_fpr_multiclass(all_labels, all_predictions, n_classes=num_classes)
            print(f"TPR: {TPR}, FPR: {FPR}")
            report = classification_report(all_labels, all_predictions, output_dict=True, zero_division=1)
            precision = report['weighted avg']['precision']
            recall = report['macro avg']['recall']
            f1_score = report['weighted avg']['f1-score']

            print(
                f'Precision: {precision:.6f}, Recall: {recall:.6f}, F1 Score: {f1_score:.6f}, Accuracy: {accuracy:.6f}')

            save_dir = f'D:/Python Project/Deep-Traffic/models/{dataset_name}/'
            file_name = f'Meta_TCN_best_{dataset_name}_{ep}.pth'
            save_file = os.path.join(save_dir, file_name)

            if best_acc < accuracy:
                best_acc = accuracy
                torch.save(model.state_dict(), save_file)
                print(f"save best model at {save_file} at epoch {epoch}")


def load_model(model_path, device):
    # 加载模型
    # BiTCN = BiTCN(input_channel, num_classes, channel_size, kernel_size=kernel_sizes, dropout=Dropout)
    # 加载模型
    model = torch.load(model_path, map_location=device)
    # 将模型移动到指定设备
    model = model.to(device)

    # 设置模型为评估模式
    model.eval()

    return model


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def test(model, test_loader, class_nums):
    print("start testing", end="\n")
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    thresholds = [0] * class_nums  # class_nums 是你的类别数量

    # 获取模型参数所在的设备
    device = next(model.parameters()).device
    target_num = torch.zeros(class_nums, device=device)
    predict_num = torch.zeros(class_nums, device=device)
    acc_num = torch.zeros(class_nums, device=device)
    startTest = time.perf_counter()

    all_preds = []
    true_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.to(device), target.to(device)
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]

            output = model(data)
            # 在测试阶段应用 softmax 函数
            probs = torch.nn.functional.softmax(output, dim=1)

            # 获取每个样本最大概率对应的类别索引
            max_probs, preds = torch.max(probs, dim=1)
            # 应用阈值
            for i in range(len(preds)):
                if max_probs[i] < thresholds[preds[i]]:
                    preds[i] = -1  # 将不确定的预测标记为-1或其他未定义类别

            all_preds.extend(preds.cpu().numpy().flatten())
            true_labels.extend(target.cpu().numpy())

            test_loss += F.nll_loss(output, target, reduction='sum').item()

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total += target.size(0)

            pre_mask = torch.zeros_like(output).scatter_(1, pred, 1.)
            predict_num += pre_mask.sum(0)
            tar_mask = torch.zeros_like(output).scatter_(1, target.view(-1, 1), 1.)
            target_num += tar_mask.sum(0)
            acc_mask = pre_mask * tar_mask
            acc_num += acc_mask.sum(0)

        test_loss /= len(test_loader.dataset)

        # 防止除零错误
        recall = acc_num / (target_num + 1e-7)
        precision = acc_num / (predict_num + 1e-7)
        F1 = 2 * recall * precision / (recall + precision + 1e-7)
        accuracy = correct / total

        # 调整精度并转换为百分比形式
        recall = (recall.cpu().numpy() * 1).round(3)
        precision = (precision.cpu().numpy() * 1).round(3)
        F1 = (F1.cpu().numpy() * 1).round(3)
        # TPR, FPR = calculate_tpr_fpr("class_indices.json", true_labels=true_labels, pred_labels=all_preds, )
        TPR, FPR = calculate_tpr_fpr_multiclass(true_labels, all_preds, class_nums)

        print(f'testSize: {len(test_loader.dataset)}')
        print(f'recall: {" ".join(map(str, recall))}')
        print(f'precision: {" ".join(map(str, precision))}')
        print(f'F1: {"  ".join(map(str, F1))}')
        print(f'accuracy: {accuracy}%')
        print(f'TPR: {TPR}, FPR: {FPR}')
        # 计算相关指标
        report = classification_report(true_labels, all_preds, output_dict=True, zero_division=0)
        precision_weighted = report['weighted avg']['precision']
        recall_macro = report['macro avg']['recall']
        f1_score_weighted = report['weighted avg']['f1-score']
        print(
            f'Precision: {precision_weighted:.4f}, Recall: {recall_macro:.4f}, F1 Score: {f1_score_weighted:.4f}, '
            f'Accuracy: {accuracy:.4f}')

        print(
            f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy})')

    return all_preds, recall_macro, precision_weighted, f1_score_weighted, TPR, FPR, accuracy


def train_model(train_loader, model, device, class_nums, epochs=10):
    input_size = 16
    output_size = class_nums
    num_channels = [32] * 8
    log_interval = 1000
    train_loss = 0

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_function = FocalLoss()
    scaler = GradScaler()  # 初始化 GradScaler

    for ep in range(epochs):
        model.train()
        all_preds = []
        all_targets = []

        for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.cuda(), target.cuda()  # data为img, target为label
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            optimizer.zero_grad()

            # 使用 autocast 上下文管理器进行前向传播
            with autocast("cuda"):
                output = model(data)
                loss = loss_function(output, target)

            # 使用 scaler 缩放损失并进行反向传播
            scaler.scale(loss).backward()

            if args.clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            # 使用 scaler 更新模型参数
            scaler.step(optimizer)
            scaler.update()

            # 计算损失值
            train_loss += loss.item()  # 使用 loss.item() 获取标量值
            if batch_idx > 0 and batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                    ep, batch_idx * data.size(0), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), train_loss / log_interval, steps))
                train_loss = 0

            # 保存预测结果和目标值
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

        # 计算并打印准确率、召回率和F1分数
        accuracy = accuracy_score(all_targets, all_preds)
        recall = recall_score(all_targets, all_preds, average='macro')
        f1 = f1_score(all_targets, all_preds, average='macro')

        print(f'Epoch {ep} - Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    return model


def meta_predict(stacking_train, labels, input_size, num_classes, device, dataset_name):
    stacking_train_tensor = stacking_train
    # 将数据转换为 PyTorch 张量
    stacking_train_tensor = torch.tensor(stacking_train, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

    # 创建数据集和数据加载器
    dataset = TensorDataset(stacking_train_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    Epoch = 10
    # 初始化存储指标的列表
    accuracies = []
    f1_scores = []
    tprs = []
    fprs = []
    for ep in range(Epoch):  # delete  1 3 4
        print(f'L Epoch: {ep}')
        # model_path = f'./models/{dataset_name}/Meta_tcn_best_{dataset_name}_' + str(ep) + '.pth'
        model_path = f"./models/{dataset_name}/Meta_TCN_best_{dataset_name}_{ep}.pth"
        model_state_dict = torch.load(model_path, map_location=device)
        model = TCN_model(input_size, num_classes, channel_sizes, kernel_size, dropout).to(device)
        model.load_state_dict(model_state_dict)

        # 评估模型
        model.eval()
        all_outputs = []
        all_labels = []
        all_predictions = []
        print('validating TCN')
        with torch.no_grad():
            # 使用 tqdm 包裹 dataloader
            for inputs, targets in tqdm(dataloader, desc=f'validating Epoch {ep + 1}/{Epoch}', leave=False):
                with autocast("cuda"):  # 使用 autocast 上下文管理器
                    outputs = model(inputs)
                    probs = F.softmax(outputs, dim=1)
                    all_outputs.append(probs.cpu().numpy())
                    all_labels.append(targets.cpu().numpy())

                    # 获取每个样本最大概率对应的类别索引
                    predictions = torch.argmax(outputs, dim=1)
                    all_predictions.append(predictions.cpu().numpy())

        all_outputs = np.concatenate(all_outputs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        if ep == 0:
            # 计算混淆矩阵
            cm = sk_confusion_matrix(all_labels, all_predictions)
            # 打印混淆矩阵
            print("Confusion Matrix:")
            print(cm)
            # 保存混淆矩阵到文件
            cm_file_path = f'{dataset_name}_TCN_confusion_matrix.csv'
            np.savetxt(cm_file_path, cm, delimiter=',', fmt='%d')
            print(f"Confusion matrix saved to {cm_file_path}")

        # 计算准确率
        correct_predictions = (all_predictions == all_labels).sum()
        accuracy = correct_predictions / len(all_labels)

        # 计算其他指标
        TPR, FPR = calculate_tpr_fpr_multiclass(all_labels, all_predictions, n_classes=num_classes)
        print(f"TPR: {TPR}, FPR: {FPR}")
        report = classification_report(all_labels, all_predictions, output_dict=True, zero_division=1)
        precision = report['weighted avg']['precision']
        recall = report['macro avg']['recall']
        f1_score = report['weighted avg']['f1-score']

        print(
            f'Precision: {precision:.6f}, Recall: {recall:.6f}, F1 Score: {f1_score:.6f}, Accuracy: {accuracy:.6f}')

        accuracies.append(accuracy)
        f1_scores.append(f1_score)
        tprs.append(TPR)
        fprs.append(FPR)

    accuracies = np.array(accuracies)
    f1_scores = np.array(f1_scores).flatten()
    fprs = np.array(fprs).flatten()
    tprs = np.array(tprs).flatten()
    print('Mean accuracy  (%)  ', f'{np.mean(accuracies):.4f}±{np.std(accuracies):.4f}')
    print('Mean FPR            ', f'{np.mean(fprs):.4f}±{np.std(fprs):.4f}')
    print('Mean TPR            ', f'{np.mean(tprs):.4f}±{np.std(tprs):.4f}')
    print('Mean F1 score  (%)  ', f'{np.mean(f1_scores):.4f}±{np.std(f1_scores):.4f}')

    csv_file_path = f'./predict/{dataset_name}_Meta_TCN.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration'] + [str(i + 1) for i in range(10)])
        writer.writerow(['Accuracy'] + [f"{acc:.4f}" for acc in accuracies])
        writer.writerow(['TPR'] + [f"{tpr:.4f}" for tpr in tprs])
        writer.writerow(['FPR'] + [f"{fpr:.4f}" for fpr in fprs])
        writer.writerow(['F1 Score'] + [f"{f1:.4f}" for f1 in f1_scores])

    print(f"Results saved to {csv_file_path}")
    return all_predictions


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PNG_PATH = '4_Png_16_USTC'
    EP = 10
    recalls = []
    precisions = []
    f1_scores = []
    tprs = []
    fprs = []
    accuracies = []

    directory_path = os.path.join(os.path.abspath(os.getcwd()), f"../pre-processing/{PNG_PATH}/Train")
    entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    n_classes = folder_count
    _, test_load, _, _, _, dataset_name = data_pre_process(os.getcwd(), PNG_PATH)

    class_nums = n_classes

    for i in range(EP):
        print("\nIteration : ", i)
        model_path = f"../models/{dataset_name[:-1]}/TCN_model_best_{dataset_name}" + str(i) + ".pt"
        model = load_model(model_path, device)

        _, recall, precision, f1_score, tpr, fpr, accuracy = test(model, test_load, class_nums)

        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1_score)
        tprs.append(tpr)
        fprs.append(fpr)
        accuracies.append(accuracy)

    f1_scores = np.array(f1_scores)
    tprs = np.array(tprs)
    fprs = np.array(fprs)
    accuracies = np.array(accuracies)

    all_f1_scores = f1_scores.flatten()
    all_tprs = tprs.flatten()
    all_fprs = fprs.flatten()

    mean_f1_score = np.mean(all_f1_scores)
    std_f1_score = np.std(all_f1_scores)
    mean_tpr = np.mean(all_tprs)
    std_tpr = np.std(all_tprs)
    mean_fpr = np.mean(all_fprs)
    std_fpr = np.std(all_fprs)
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    # 打印结果
    print("accuracies", accuracies)
    print("TPRs", all_tprs)
    print("FPRs", all_fprs)
    print("F1 scores", all_f1_scores)

    print('Mean accuracy  (%)  ', f'{mean_accuracy:.4f}±{std_accuracy :.4f}')
    print('Mean TPR            ', f'{mean_tpr :.4f}±{std_tpr :.4f}')
    print('Mean FPR            ', f'{mean_fpr :.4f}±{std_fpr:.4f}')
    print('Mean F1 score  (%)  ', f'{mean_f1_score :.4f}±{std_f1_score:.4f}')

    csv_file_path = f'{dataset_name}TCN.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration'] + [str(i + 1) for i in range(10)])
        writer.writerow(['Accuracy'] + [f"{acc:.4f}" for acc in accuracies])
        writer.writerow(['TPR'] + [f"{tpr:.4f}" for tpr in all_tprs])
        writer.writerow(['FPR'] + [f"{fpr:.4f}" for fpr in all_fprs])
        writer.writerow(['F1 Score'] + [f"{f1:.4f}" for f1 in all_f1_scores])

    print(f"Results saved to {csv_file_path}")


def main_train():
    print('training TCN')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_hid = 32  # 每层隐藏单元数
    levels = 8  # 卷积层数
    channel_sizes = [num_hid] * levels  # 设置通道数 = 隐藏层数*每层隐藏单元数
    kernel_size = 7  # 卷积核大小
    dropout = 0.05
    input_size = 16
    # 获得数据集的类别数量  4_Png_16_USTC   4_Png_16_CTU  4_Png_16_ISAC
    directory_path = os.path.join(os.path.join(os.getcwd()), "../pre-processing/4_Png_16_ISAC/Train")
    entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    class_nums = folder_count  # 此参数为分类的数量，需要根据实际情况（数据集）修改

    model = tcn_train_and_test.get_model(class_nums)

    train_loader, test_loader, _, _, _, dataset_name = data_pre_process(os.path.join(os.getcwd(), "../"),
                                                                        '4_Png_16_ISAC')
    model = train_model(train_loader=train_loader, epochs=20, model=model, device=device, class_nums=class_nums)
    acc = test(model, test_loader, class_nums)[-1]
    print("Test Accuracy:", acc)
    model_path = f"../models/ISAC/TCN_for_meta.pt"
    torch.save(model, model_path)
    print("Model saved to {}".format(model_path))


if __name__ == "__main__":
    main()
