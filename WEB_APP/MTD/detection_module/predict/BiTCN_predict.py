import csv
import json
import sys
import time
from ..BiTCN.BiTCN_zy import BiTCN
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn.functional as F
from torch import nn, optim
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from ..FocalLoss import FocalLoss
from ..tcn import TemporalConvNet
from ..model import TCN
from ..utils import data_pre_process, calculate_tpr_fpr, calculate_tpr_fpr_multiclass


class meta_BiTCN(nn.Module):
    def __init__(self, input_size, num_classes, num_channels, kernel_size=3, dropout=0.05):
        super(meta_BiTCN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_size if i == 0 else num_channels[i - 1],
                      out_channels=num_channels[i],
                      kernel_size=kernel_size,
                      padding=(kernel_size // 2))
            for i in range(len(num_channels))
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # 双向卷积
        for conv in self.convs:
            x = F.elu(conv(x))
            x = self.dropout(x)
        x = x.mean(dim=2)  # 全局平均池化
        x = self.fc(x)
        return x


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append("../../")

# 定义参数
args = {
    'batches_size': 64,  # 批次大小
    'cuda': True,  # 是否使用 CUDA
    'Dropout': 0.05,  # Dropout 概率
    'clip': -1,  # 梯度裁剪阈值，-1 表示不进行裁剪
    'epochs': 10,  # 最大训练轮数
    'ksize': 7,  # 卷积核大小
    'levels': 8,  # 网络层数
    'log_interval': 500,  # 日志输出间隔
    'lr': 0.001,  # 学习率
    'optim': 'Adam',  # 优化器名称
    'nhid': 25,  # 每层的隐藏单元数
    'seed': 1111,  # 随机种子
    'permute': False,  # 是否使用打乱的数据
}

torch.manual_seed(args['seed'])  # 随机初始化种子

batch_size = args['batches_size']  # 一次训练尺寸

# input_channels = 3  # 输入通道
#
# seq_length = int(768 / input_channels)  # 序列长度
epochs = args['epochs']

# 一个TCN基本块包含的通道数及层数 这里为[25,25,25,25,25,25,25,25]，即25*8

# 设置随机种子
torch.manual_seed(args['seed'])

# 设置设备
device = torch.device("cuda" if args['cuda'] and torch.cuda.is_available() else "cpu")


def load_model(model_path, device):
    """
    Loads a trained BiTCN from a file.

    Args:
        model_path (str): Path to the saved BiTCN file.
        device (torch.device): Device to load the BiTCN onto.

    Returns:
        nn.Module: Loaded BiTCN.
    """
    model = torch.load(model_path)
    model.to(device)
    return model


def predict(model, data_loader, device, input_channel=3, seq_length=int(768 / 3), flip_data=True,
            meta=False,
            num_classes=None):
    """
    Predicts labels for a given DataLoader using the provided BiTCN and calculates Precision, Recall, F1 Score, and Accuracy.

    Args:
        model (nn.Module): Trained BiTCN.
        data_loader (DataLoader): DataLoader for the dataset to predict.
        device (str or torch.device): Device to run the prediction on ('cuda' or 'cpu').
        input_channel (int): Number of input channels (e.g., 3 for RGB).
        seq_length (int): Sequence length.
        num_classes (int): Number of classes.
        flip_data(bool): Whether to flip the data
        dataset_name (str): Name of the dataset
        meta (bool): Whether to load the meta
    Returns:
        List[int]: Predicted labels.
        float: Accuracy.
        List[float]: Precision.
        List[float]: Recall.
        List[float]: F1 Score.
    """
    model.eval()
    predictions = []
    all_labels = []  # 存储所有真实标签
    correct = 0  # 初始化正确预测的数量
    total = 0  # 初始化总的预测数量
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            labels = labels.to(device)
            if not meta:
                data = data.view(-1, input_channel, seq_length)

            if flip_data:
                data = torch.flip(data, dims=[2])

            output = model(data)
            pred = output.argmax(dim=1).tolist()
            predictions.extend(pred)
            all_labels.extend(labels.cpu().numpy().tolist())
            total += labels.size(0)
            correct += (torch.tensor(pred) == labels.cpu()).sum().item()

    accuracy = correct / total  # 计算准确率

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, predictions)

    # 计算 Precision、Recall 和 F1 Score
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    # 处理除零错误
    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0
    f1_score[np.isnan(f1_score)] = 0

    # 计算 TPR 和 FPR
    # TPR, FPR = calculate_tpr_fpr(json_filepath='class_indices.json', true_labels=all_labels, pred_labels=predictions)
    TPR, FPR = calculate_tpr_fpr_multiclass(all_labels, predictions, n_classes=num_classes)
    report = classification_report(all_labels, predictions, zero_division=1, output_dict=True)
    F1_score = report['weighted avg']['f1-score']
    print(f"TPR: {TPR}, FPR: {FPR}")
    print("Precision: \n", precision)
    print("Recall: \n", recall)
    print("F1 Score: \n", f1_score)
    print("Accuracy: \n", accuracy)

    return predictions, accuracy, TPR, FPR, F1_score


def train(model, train_loader, epoch, loss_function, optimizer, input_channels, seq_length, flip_data=False):
    """
    Train the BiTCN for one epoch.

    Parameters:
    - epoch: Current training epoch.
    - flip_data: Whether to flip the input data.
    """
    # train_loader = data_pre_process(os.getcwd(), "4_Png_16_USTC", None)[0]
    # loss_function = torch.nn.CrossEntropyLoss()
    train_loss = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
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
        if args['clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss.item()

        if batch_idx > 0 and batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_loss = 0
    return model


def test(model, validate_loader, dataset_name, input_channels, seq_length, flip_data=False):
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


def fit(stacking_train, labels, input_size, num_classes, epochs, dataset_name):
    # 将NumPy数组转换为Tensor并创建DataLoader
    stacking_train_tensor = torch.tensor(stacking_train, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(stacking_train_tensor, labels_tensor, test_size=0.2,
                                                        random_state=args['seed'])

    # 转换为Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 创建DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args['batches_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args['batches_size'], shuffle=False)
    for ep in range(10):
        print(f'L Epoch: {ep}')
        model_meta = meta_BiTCN(input_size=input_size, num_classes=num_classes, num_channels=8 * [25], dropout=0.05,
                                kernel_size=3)
        model_meta = model_meta.to(device)
        meta_optimizer = torch.optim.AdamW(model_meta.parameters(), lr=0.001)
        # meta_criterion = FocalLoss(0.75, 3)
        meta_criterion = nn.CrossEntropyLoss()
        # meta_criterion = F.nll_loss

        best_acc = 0.0
        print("Forward Training")
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}')
            print('*' * 10)
            correct = 0
            total = 0
            target_num = torch.zeros((1, num_classes))
            predict_num = torch.zeros((1, num_classes))
            acc_num = torch.zeros((1, num_classes))

            # 正常训练
            model_meta.train()
            running_loss = 0.0
            running_acc = 0.0
            all_preds = []
            all_labels = []
            output_list = []
            start_train_time = time.perf_counter()
            label_list = []
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                label_list.append(labels)
                # 前向传播
                outputs = model_meta(inputs)
                output_list.append(outputs)

                loss = meta_criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

                # 计算准确率
                _, preds = torch.max(outputs, 1)
                running_acc += (preds == labels).sum().item()

                # 反向传播
                meta_optimizer.zero_grad()
                loss.backward()
                meta_optimizer.step()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # 正向验证集验证结果
            pred, Acc, TPR, FPR, f1_score = predict(model_meta, test_loader, device, input_size, num_classes,
                                                    meta=True)

            train_time = (time.perf_counter() - start_train_time)
            print(f"Train Time (Forward): {train_time:.2f}s")
            print(
                f'Finish Forward Training {epoch + 1} epoch, Loss: {running_loss / len(train_loader.dataset):.6f},'
                f' Acc: {running_acc / len(train_loader.dataset):.6f}')
            print(f"\n正向验证集验证结果： acc:{Acc:.4f}, TPR:{TPR:.4f}, FPR:{FPR:.4f}, F1:{f1_score:.4f} \n")

            # # 计算指标
            # report = classification_report(all_labels, all_preds, output_dict=True, zero_division=1)
            # precision = report['weighted avg']['precision']
            # recall = report['macro avg']['recall']
            # f1_score = report['weighted avg']['f1-score']
            # TPR, FPR = calculate_tpr_fpr(json_filepath='class_indices.json', true_labels=all_labels,
            #                              pred_labels=all_preds)
            # # 打印格式方便复制
            # print(
            #     f'Precision (Forward): {precision:.6f}, Recall (Forward): {recall:.6f}, F1 Score (Forward): {f1_score:.6f}')
            # print(f"TPR (Forward): {TPR:.6f}, FPR (Forward): {FPR:.6f}\n")

            # 翻转数据并再次训练
            model_meta.train()
            running_loss = 0.0
            running_acc = 0.0
            all_preds = []
            all_labels = []
            start_train_time = time.perf_counter()
            Reverse_output = []
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs = torch.flip(inputs, dims=[2])  # 翻转数据

                # 前向传播
                outputs = model_meta(inputs)
                Reverse_output.append(outputs)
                loss = meta_criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
                # 计算准确率
                _, preds = torch.max(outputs, 1)
                running_acc += (preds == labels).sum().item()
                # 反向传播
                meta_optimizer.zero_grad()
                loss.backward()
                meta_optimizer.step()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # 翻转数据集验证集验证结果
            train_time = (time.perf_counter() - start_train_time)
            print(f"Train Time (Backward): {train_time:.2f}s")
            print(
                f'Finish Backward Training {epoch + 1} epoch, Loss: {running_loss / len(train_loader.dataset):.6f}, Acc: {running_acc / len(train_loader.dataset):.6f}')

            pred, Acc, TPR, FPR, f1_score = predict(model_meta, test_loader, device, input_size, num_classes,
                                                    meta=True)
            print(f"\n翻转数据集验证集验证结果： acc:{Acc:.4f}, TPR:{TPR:.4f}, FPR:{FPR:.4f}, F1:{f1_score:.4f} \n")

            # # 计算指标
            # report = classification_report(all_labels, all_preds, output_dict=True, zero_division=1)
            # precision = report['weighted avg']['precision']
            # recall = report['macro avg']['recall']
            # f1_score = report['weighted avg']['f1-score']
            # TPR, FPR = calculate_tpr_fpr(json_filepath='class_indices.json', true_labels=all_labels,
            #                              pred_labels=all_preds)
            # print(
            # f'Precision (Backward): {precision:.6f}, Recall (Backward): {recall:.6f}, F1 Score (Backward): {f1_score:.6f}')
            # print(f"TPR (Backward): {TPR:.6f}, FPR (Backward): {FPR:.6f} \n")

            #  结合正向和反向信息
            result = []
            loss = 0.0
            true_labels = []  # 存放真实标签
            predicted_labels = []  # 存放预测标签
            for i in range(len(output_list)):  # 结合正向训练和反向训练
                result.append(output_list[i] + Reverse_output[i])
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

                # 收集真实标签和预测标签
                true_labels.extend(label_list[j].cpu().numpy())
                predicted_labels.extend(pred.squeeze().cpu().numpy())

            recall = acc_num / target_num
            precision = acc_num / predict_num
            F1 = 2 * recall * precision / (recall + precision)
            accuracy = acc_num.sum(1) / target_num.sum(1)
            # 精度调整
            recall = (recall.numpy()[0]).round(4)
            precision = (precision.numpy()[0]).round(4)
            F1 = (F1.numpy()[0]).round(4)
            accuracy = (accuracy.numpy()[0]).round(6)
            tpr, fpr = calculate_tpr_fpr('class_indices.json', true_labels, predicted_labels)
            # 打印格式方便复制
            print('*' * 10)
            print(f'双向训练结果如下：')
            print('Recall (%)  ', " ".join('%s' % id for id in recall))
            print('Precision(%)', " ".join('%s' % id for id in precision))
            print('F1 (%)      ', " ".join('%s' % id for id in F1))
            print('TPR(Recall) ', tpr)
            print('FPR         ', fpr)
            print('accuracy(%) ', accuracy)

            save_dir = './models/USTC/Meta_BiTCN_best_USTC_' + str(ep) + '.pth'
            if best_acc < Acc:
                best_acc = Acc
                print(f"saving best model to {save_dir}, accuracy {Acc}")
                torch.save(model_meta.state_dict(), save_dir)

        final_dir = './models/USTC/Meta_BiTCN_model_final_' + str(ep) + '.pth'
        torch.save(model_meta.state_dict(), final_dir)
        print(f"finish train,best accuracy {best_acc}")


def fit_bitcn(stacking_train, labels, input_size, num_classes, dataset_name, epochs):
    list_time = []
    # 将NumPy数组转换为Tensor并创建DataLoader
    stacking_train_tensor = torch.tensor(stacking_train, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

    # 检查两个张量的形状
    if stacking_train_tensor.shape[0] != labels_tensor.shape[0]:
        min_length = min(stacking_train_tensor.shape[0], labels_tensor.shape[0])
        stacking_train_tensor = stacking_train_tensor[:min_length]
        labels_tensor = labels_tensor[:min_length]
    # 打印形状以确认
    print(f"stacking_train_tensor shape: {stacking_train_tensor.shape}")
    print(f"labels_tensor shape: {labels_tensor.shape}")
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(stacking_train_tensor, labels_tensor, test_size=0.2,
                                                        random_state=args['seed'])

    # 转换为Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 创建DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args['batches_size'], shuffle=True, drop_last=True)
    validate_loader = DataLoader(test_dataset, batch_size=args['batches_size'], shuffle=False, drop_last=True)

    input_channels = 5
    seq_length = 1
    EP = 10
    for frequency in range(EP):
        best_accuracy = 0.0
        print("frequency: ", frequency)
        lr = 0.002

        # BiTCN = TCN(input_channels, num_classes, channel_sizes, kernel_size=kernel_size, dropout=args.Dropout)
        # BiTCN = BiTCN(input_size=input_size, output_size=num_classes, num_channels=8 * [25], dropout=0.005,
        #               kernel_size=3)
        model_meta = meta_BiTCN(input_size=input_size, num_classes=num_classes, num_channels=8 * [25], dropout=0.05,
                                kernel_size=3)
        model_meta.cuda()
        optimizer = optim.Adam(model_meta.parameters(), lr=lr)
        # loss_function = FocalLoss()
        # loss_function = F.nll_loss
        loss_function = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            print(f"\nepoch:{epoch + 1} ", end="")
            correct = 0
            total = 0
            target_num = torch.zeros((1, num_classes))
            predict_num = torch.zeros((1, num_classes))
            acc_num = torch.zeros((1, num_classes))

            # 记录时间戳，正向开始训练时间
            train_start = time.time()
            list_time.append(train_start)
            print("forward training")
            train(model=model_meta, train_loader=train_loader, epoch=epoch, optimizer=optimizer,
                  loss_function=loss_function,
                  flip_data=False, input_channels=input_channels, seq_length=seq_length)

            # 记录时间戳，正向结束训练时间
            train_end = time.time()
            list_time.append(train_end)

            test_loss, out_list, label_list, TPR, FPR = test(model=model_meta, validate_loader=validate_loader,
                                                             input_channels=input_channels, seq_length=seq_length,
                                                             dataset_name=dataset_name)

            # 记录时间戳，反向开始训练时间
            train_flipped_start = time.time()
            list_time.append(train_flipped_start)
            print("backward training")
            train(model=model_meta, train_loader=train_loader, epoch=epoch, optimizer=optimizer,
                  loss_function=loss_function,
                  input_channels=input_channels, seq_length=seq_length,
                  flip_data=True)

            # 记录时间戳，反向结束训练时间
            train_flipped_end = time.time()
            list_time.append(train_flipped_end)

            flx_test_loss, fout_list, all_true_labels, TPR, FPR = test(model=model_meta,
                                                                       validate_loader=validate_loader,
                                                                       input_channels=input_channels,
                                                                       seq_length=seq_length,
                                                                       dataset_name=dataset_name,
                                                                       flip_data=True)

            print("双向训练结果如下：")
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
                save_dir = './models/USTC'
                os.makedirs(save_dir, exist_ok=True)
                best_model_filename = os.path.join(save_dir, 'META_BiTCN_best_USTC_' + str(frequency) + '.pt')
                torch.save(model_meta, best_model_filename)
                print(f'Saved new best model with accuracy {best_accuracy:.3f}% as {best_model_filename}')
        # 指定保存模型的目录
        save_dir = './models/USTC'
        os.makedirs(save_dir, exist_ok=True)
        save_filename = os.path.join(save_dir, 'META_BiTCN_final_USTC_' + str(frequency) + '.pt')
        # 保存模型的
        torch.save(model_meta, save_filename)
        print('Saved as %s' % save_filename)


def meta_predict(stacking_train, labels, input_size, num_classes):
    # 将输入数据转换为Tensor
    stacking_train = torch.tensor(stacking_train, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # 创建DataLoader
    dataset = TensorDataset(stacking_train, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 初始化预测结果
    accuracies = []
    tprs = []
    fprs = []
    f1_scores = []
    EP = 10
    # 遍历所有模型文件
    for ep in range(EP):
        model_path = f'./models/USTC/META_BiTCN_best_USTC_{ep}.pth'

        # 加载模型
        model_meta = meta_BiTCN(input_size=input_size, num_classes=num_classes, num_channels=8 * [25], dropout=0.05,
                                kernel_size=3)
        model_meta.load_state_dict(torch.load(model_path))
        model_meta = model_meta.to(device)
        model_meta.eval()
        # 进行预测
        predictions = []
        true_labels = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model_meta(inputs)
                _, predicted = torch.max(outputs, 1)

                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(targets.cpu().numpy())
        report = classification_report(true_labels, predictions, output_dict=True, zero_division=1)
        accuracy = report['accuracy']
        TPR, FPR = calculate_tpr_fpr(json_filepath='class_indices.json', true_labels=true_labels,
                                     pred_labels=predictions)
        f1_score = report['weighted avg']['f1-score']

        accuracies.append(accuracy)
        f1_scores.append(f1_score)
        fprs.append(FPR)
        tprs.append(TPR)

    accuracies = np.array(accuracies)
    f1_scores = np.array(f1_scores).flatten()
    fprs = np.array(fprs).flatten()
    tprs = np.array(tprs).flatten()
    print('Mean accuracy  (%)  ', f'{np.mean(accuracies):.4f}±{np.std(accuracies):.4f}')
    print('Mean FPR            ', f'{np.mean(fprs):.4f}±{np.std(fprs):.4f}')
    print('Mean FPR            ', f'{np.mean(tprs):.4f}±{np.std(tprs):.4f}')
    print('Mean F1 score  (%)  ', f'{np.mean(f1_scores):.4f}±{np.std(f1_scores):.4f}')


def train_model(train_loader, epochs, device, class_nums, validate_loader, dataset_name):
    input_channels = 3
    channel_sizes = [32] * 8
    kernel_size = 7
    seq_length = int(768 / input_channels)  # 序列长度

    model = TCN(input_channels, class_nums, channel_sizes, kernel_size=kernel_size, dropout=0.05)
    lr = 0.001
    model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # loss_function = FocalLoss()
    loss_function = F.nll_loss
    for epoch in range(epochs):
        print(f"\nepoch:{epoch + 1} ")
        correct = 0
        total = 0
        target_num = torch.zeros((1, class_nums))
        predict_num = torch.zeros((1, class_nums))
        acc_num = torch.zeros((1, class_nums))

        # 记录时间戳，正向开始训练时间
        print("forward training")
        train(model=model, train_loader=train_loader, epoch=epoch, optimizer=optimizer, loss_function=loss_function,
              flip_data=False, input_channels=3, seq_length=int(768/input_channels))
        # 反向结束训练时间
        test_loss, out_list, label_list, TPR, FPR = test(model=model, validate_loader=validate_loader,
                                                         dataset_name=dataset_name, input_channels=input_channels,
                                                         seq_length=seq_length)

        # 记录时间戳，反向开始训练时间
        print("backward training")
        train(model=model, train_loader=train_loader, epoch=epoch, optimizer=optimizer, loss_function=loss_function,
              flip_data=True,input_channels=input_channels,
              seq_length=seq_length)
        # 记录时间戳，反向结束训练时间
        flx_test_loss, fout_list, all_true_labels, TPR, FPR = test(model=model, validate_loader=validate_loader,
                                                                   dataset_name=dataset_name,
                                                                   input_channels=input_channels,seq_length=seq_length,
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


def main():
    PNG_PATH = "4_Png_16_ISAC"
    directory_path = os.path.join(os.path.abspath(os.getcwd()), f"../pre-processing/{PNG_PATH}/Train")
    entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    n_classes = folder_count  # 分类数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, _, _, _, dataset_name = data_pre_process(os.path.join(os.getcwd(), "../"), PNG_PATH, None)

    # 初始化存储指标的列表
    accuracies = []
    f1_scores = []
    tprs = []
    fprs = []
    for i in range(10):
        model_path = f'../models/{dataset_name[:-1]}/BiTCN_final_{dataset_name}' + str(i) + '.pt'
        print(f"Loading model {model_path}")
        # Load the BiTCN
        model = torch.load(model_path)
        model.to(device)

        test_predictions, acc, tpr, fpr, f1 = predict(model, test_loader, device,
                                                      num_classes=n_classes)

        accuracies.append(acc)
        f1_scores.append(f1)
        tprs.append(tpr)
        fprs.append(fpr)
    f1_scores = np.array(f1_scores).flatten()
    tprs = np.array(tprs).flatten()
    fprs = np.array(fprs).flatten()
    print("accuracies:", accuracies)
    print("f1_scores:", f1_scores)
    print("tprs:", tprs)
    print("fprs:", fprs)
    # 计算均值和标准差
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    mean_f1_score = np.mean(f1_scores)
    std_f1_score = np.std(f1_scores)

    mean_tpr = np.mean(tprs)
    std_tpr = np.std(tprs)

    mean_fpr = np.mean(fprs)
    std_fpr = np.std(fprs)
    print(f"Accuracy: {mean_accuracy:.4f}±{std_accuracy:.4f}")
    print(f"F1 Score: {mean_f1_score:.4f}±{std_f1_score:.4f}")
    print(f"TPR: {mean_tpr:.4f}±{std_tpr:.4f}")
    print(f"FPR: {mean_fpr:.4f}±{std_fpr:.4f}")

    csv_file_path = f'{dataset_name}BiTCN.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration'] + [str(i + 1) for i in range(10)])
        writer.writerow(['Accuracy'] + [f"{acc:.4f}" for acc in accuracies])
        writer.writerow(['TPR'] + [f"{tpr:.4f}" for tpr in tprs])
        writer.writerow(['FPR'] + [f"{fpr:.4f}" for fpr in fprs])
        writer.writerow(['F1 Score'] + [f"{f1:.4f}" for f1 in f1_scores])

    print(f"Results saved to {csv_file_path}")
    # print("Test Predictions:", test_predictions)


if __name__ == "__main__":
    main()

