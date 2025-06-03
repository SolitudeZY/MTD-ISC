import csv
import json
import time
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch import nn, optim
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import torch
from torchmetrics import Accuracy, F1Score, Precision, Recall

import utils
from FocalLoss import FocalLoss
from tcn import TemporalConvNet
from utils import data_pre_process, calculate_tpr_fpr, calculate_tpr_fpr_multiclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class meta_BiTCN(nn.Module):
    def __init__(self, in_channels, num_classes, num_channels, kernel_size=3, dropout=0.05):
        super(meta_BiTCN, self).__init__()
        self.convs_forward = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels if i == 0 else num_channels[i - 1] * 2,
                      out_channels=num_channels[i],
                      kernel_size=kernel_size,
                      padding=(kernel_size // 2))
            for i in range(len(num_channels))
        ])
        self.convs_backward = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels if i == 0 else num_channels[i - 1] * 2,
                      out_channels=num_channels[i],
                      kernel_size=kernel_size,
                      padding=(kernel_size // 2))
            for i in range(len(num_channels))
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels[-1] * 2, num_classes)  # 考虑双向卷积后的特征维度

    def forward(self, x):
        # 双向卷积
        for conv_forward, conv_backward in zip(self.convs_forward, self.convs_backward):
            x_forward = F.elu(conv_forward(x))
            x_backward = F.elu(conv_backward(torch.flip(x, [2])))  # 反向卷积
            x = torch.cat((x_forward, x_backward), dim=1)  # 合并正向和反向卷积的结果
            x = self.dropout(x)
        x = x.mean(dim=2)  # 全局平均池化
        x = self.fc(x)
        return x


def train(model, dataloader, criterion, optimizer, device, num_class):
    model.to(device)
    model.train()

    total_loss = 0
    accuracy = Accuracy(num_classes=num_class, task='multiclass').to(device)
    f1 = F1Score(num_classes=num_class, task='multiclass').to(device)
    precision = Precision(num_classes=num_class, task='multiclass').to(device)
    recall = Recall(num_classes=num_class, task='multiclass').to(device)

    scaler = GradScaler()  # 创建 GradScaler 对象

    for inputs, targets in dataloader:
        inputs = inputs.view(inputs.size(0), inputs.size(1), inputs.size(2) * inputs.size(3))
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast("cuda"):  # 使用 autocast 上下文管理器
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()  # 缩放损失并反向传播
        scaler.step(optimizer)  # 更新权重
        scaler.update()  # 更新缩放因子

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        accuracy.update(preds, targets)
        f1.update(preds, targets)
        precision.update(preds, targets)
        recall.update(preds, targets)

    avg_loss = total_loss / len(dataloader)
    acc = accuracy.compute()
    f1_score = f1.compute()
    tpr = recall.compute()  # TPR is the same as recall
    fpr = 1 - precision.compute()  # FPR is 1 - precision

    print(f"Train Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, F1 Score: {f1_score:.4f}, TPR: {tpr:.4f}, FPR: {fpr:.4f}")

    return model


def validate(model, dataloader, criterion, device, num_class, dataset_name):
    model.eval()
    total_loss = 0
    accuracy = Accuracy(num_classes=num_class, task='multiclass').to(device)
    f1 = F1Score(num_classes=num_class, task='multiclass').to(device)
    precision = Precision(num_classes=num_class, task='multiclass').to(device)
    recall = Recall(num_classes=num_class, task='multiclass').to(device)

    print("Evaluating BiTCN on validation set")
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.view(inputs.size(0), inputs.size(1), inputs.size(2) * inputs.size(3))
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            accuracy.update(preds, targets)
            f1.update(preds, targets)
            precision.update(preds, targets)
            recall.update(preds, targets)

            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy.compute()
    f1_score = f1.compute()
    tpr = recall.compute()  # TPR is the same as recall
    fpr = 1 - precision.compute()  # FPR is 1 - precision
    TPR, FPR = calculate_tpr_fpr(dataset_name + 'class_indices.json', true_labels=all_targets, pred_labels=all_preds)

    print(
        f"Validation Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, F1 Score: {f1_score:.4f}, TPR: {tpr:.4f}, FPR: {fpr:.4f}")
    print(f"\nTPR {TPR}, FPR: {FPR}\n")
    return all_preds, accuracy


def predict(model, dataloader, device, dataset_name):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    print("Predicting BiTCN\n")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.view(inputs.size(0), inputs.size(1), inputs.size(2) * inputs.size(3))

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        TPR, FPR = calculate_tpr_fpr(dataset_name + 'class_indices.json', pred_labels=all_preds, true_labels=all_labels)
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        accuracy = report['accuracy']
        precision_weighted = report['weighted avg']['precision']
        recall_macro = report['macro avg']['recall']
        f1_score_weighted = report['weighted avg']['f1-score']
        print(f'Precision: {precision_weighted:.4f}, Recall: {recall_macro:.4f}, F1 Score: {f1_score_weighted:.4f}, ')
        print(f'TPR: {TPR:.4f}, FPR: {FPR:.4f}')
        print(f'Accuracy: {accuracy:.4f}')
    return all_preds


def fit_bitcn(stacking_train, labels, input_size, num_classes, dataset_name, epochs):
    # 将NumPy数组转换为Tensor并创建DataLoader
    stacking_train_tensor = torch.tensor(stacking_train, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    path = '4_Png_16_USTC'
    data_directory = os.path.join(os.getcwd(), '')
    test_loader = utils.data_pre_process(data_directory, png_path=path, balance=None)[1]
    # 打印形状以确认
    print(f"stacking_train_tensor shape: {stacking_train_tensor.shape}")
    print(f"labels_tensor shape: {labels_tensor.shape}")
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(stacking_train_tensor, labels_tensor, test_size=0.2,
                                                        random_state=42)
    # 转换为Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 创建DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    validate_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)
    EP = 10
    for ep in range(EP):  # train
        print("frequency: ", ep)
        lr = 0.001
        model_meta = meta_BiTCN(in_channels=input_size, num_classes=num_classes, num_channels=[10, 20, 30],
                                dropout=0.05,
                                kernel_size=3)
        model_meta.cuda()
        optimizer = optim.Adam(model_meta.parameters(), lr=lr)
        loss_function = FocalLoss()
        # loss_function = F.nll_loss
        # loss_function = nn.CrossEntropyLoss()

        best_accuracy = 0.0
        for epoch in range(epochs):  # training begins here
            print(f"\nepoch:{epoch + 1} ")
            print(" training")
            total_loss = 0
            accuracy = Accuracy(num_classes=num_classes, task='multiclass').to(device)
            f1 = F1Score(num_classes=num_classes, task='multiclass').to(device)
            precision = Precision(num_classes=num_classes, task='multiclass').to(device)
            recall = Recall(num_classes=num_classes, task='multiclass').to(device)

            scaler = GradScaler()  # 创建 GradScaler 对象
            for inputs, targets in train_loader:
                # inputs = inputs.view(inputs.size(0), inputs.size(1), inputs.size(2) * inputs.size(3))
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                with autocast("cuda"):  # 使用 autocast 上下文管理器
                    outputs = model_meta(inputs)
                    loss = loss_function(outputs, targets)

                scaler.scale(loss).backward()  # 缩放损失并反向传播
                scaler.step(optimizer)  # 更新权重
                scaler.update()  # 更新缩放因子

                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                accuracy.update(preds, targets)
                f1.update(preds, targets)
                precision.update(preds, targets)
                recall.update(preds, targets)

            avg_loss = total_loss / len(train_loader)
            acc = accuracy.compute()
            f1_score = f1.compute()
            tpr = recall.compute()  # TPR is the same as recall
            fpr = 1 - precision.compute()  # FPR is 1 - precision

            print(
                f"Train Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, F1 Score: {f1_score:.4f}, TPR: {tpr:.4f}, "
                f"FPR: {fpr:.4f}\n")

            # accuracy = validate(model=model_meta, criterion=loss_function, device=device, dataloader=validate_loader,
            #                     num_class=num_classes, dataset_name=dataset_name)[1]
            all_preds, all_targets = [], []
            print("validating BiTCN")
            with torch.no_grad():
                for inputs, targets in validate_loader:
                    # inputs = inputs.view(inputs.size(0), inputs.size(1), inputs.size(2) * inputs.size(3))
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = model_meta(inputs)
                    loss = loss_function(outputs, targets)
                    total_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1)
                    accuracy.update(preds, targets)
                    f1.update(preds, targets)
                    precision.update(preds, targets)
                    recall.update(preds, targets)

                    all_preds.extend(preds.cpu().numpy().flatten())
                    all_targets.extend(targets.cpu().numpy())

            avg_loss = total_loss / len(validate_loader)
            acc = accuracy.compute()
            f1_score = f1.compute()
            tpr = recall.compute()  # TPR is the same as recall
            fpr = 1 - precision.compute()  # FPR is 1 - precision
            # TPR, FPR = calculate_tpr_fpr(dataset_name + 'class_indices.json', true_labels=all_targets,
            #                              pred_labels=all_preds)
            TPR, FPR = calculate_tpr_fpr_multiclass(all_targets, all_preds, num_classes)
            print(
                f"Validation Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, F1 Score: {f1_score:.4f}, TPR: {tpr:.4f}, FPR: {fpr:.4f}")
            print(f"\nTPR {TPR}, FPR: {FPR}\n")

            model_path = f'./models/{dataset_name}/Meta_BiTCN_best_{dataset_name}_{ep}.pth'
            if best_accuracy < acc:
                best_accuracy = acc
                torch.save(model_meta.state_dict(), model_path)
                print(f"Best model saved at epoch {epoch + 1} with validation accuracy: {best_accuracy:.6f}"
                      f"\nThe model is saved in {os.path.abspath(model_path)}")
    # return model_meta
    # Test
    # predict(model_meta, test_loader, device, dataset_name+'_')


# def fit(stacking_train, labels, input_size, num_classes, dataset_name, epochs):
#     # 将NumPy数组转换为Tensor并创建DataLoader
#     stacking_train_tensor = torch.tensor(stacking_train, dtype=torch.float32).to(device)
#     labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
#
#     # 打印形状以确认
#     print(f"stacking_train_tensor shape: {stacking_train_tensor.shape}")
#     print(f"labels_tensor shape: {labels_tensor.shape}")
#     # 分割数据集
#     X_train, X_test, y_train, y_test = train_test_split(stacking_train_tensor, labels_tensor, test_size=0.2,
#                                                         random_state=args['seed'])
#
#     # 转换为Tensor
#     X_train = torch.tensor(X_train, dtype=torch.float32)
#     y_train = torch.tensor(y_train, dtype=torch.long)
#     X_test = torch.tensor(X_test, dtype=torch.float32)
#     y_test = torch.tensor(y_test, dtype=torch.long)
#
#     # 创建DataLoader
#     train_dataset = TensorDataset(X_train, y_train)
#     test_dataset = TensorDataset(X_test, y_test)
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
#     validate_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)
#
#     input_channels = 5
#     seq_length = 1
#     EP = 10
#     for frequency in range(EP):
#         best_accuracy = 0.0
#         print("frequency: ", frequency)
#
#         model_meta = meta_BiTCN(in_channels=input_size, num_classes=num_classes, num_channels=8 * [25], dropout=0.05,
#                                 kernel_size=3)
#         model_meta.cuda()
#         optimizer = optim.Adam(model_meta.parameters(), lr=0.002)
#         # loss_function = FocalLoss()
#         # loss_function = F.nll_loss
#         loss_function = nn.CrossEntropyLoss()
#         for epoch in range(epochs):
#             print(f"\nepoch:{epoch + 1} ", end="")
#             correct = 0
#             total = 0
#             target_num = torch.zeros((1, num_classes))
#             predict_num = torch.zeros((1, num_classes))
#             acc_num = torch.zeros((1, num_classes))
#
#             # 记录时间戳，正向开始训练时间
#             train_start = time.time()
#             print("forward training")
#             train(model=model_meta, train_loader=train_loader, epoch=epoch, optimizer=optimizer,
#                   loss_function=loss_function,
#                   flip_data=False, input_channels=input_channels, seq_length=seq_length)
#
#             # 记录时间戳，正向结束训练时间
#             train_end = time.time()
#
#             test_loss, out_list, label_list, TPR, FPR = test(model=model_meta, validate_loader=validate_loader,
#                                                              input_channels=input_channels, seq_length=seq_length,
#                                                              dataset_name=dataset_name)
#
#             # 记录时间戳，反向开始训练时间
#             train_flipped_start = time.time()
#             print("backward training")
#             train(model=model_meta, train_loader=train_loader, epoch=epoch, optimizer=optimizer,
#                   loss_function=loss_function,
#                   input_channels=input_channels, seq_length=seq_length,
#                   flip_data=True)
#
#             # 记录时间戳，反向结束训练时间
#             train_flipped_end = time.time()
#
#             flx_test_loss, fout_list, all_true_labels, TPR, FPR = test(model=model_meta,
#                                                                        validate_loader=validate_loader,
#                                                                        input_channels=input_channels,
#                                                                        seq_length=seq_length,
#                                                                        dataset_name=dataset_name,
#                                                                        flip_data=True)
#
#             print("双向训练结果如下：")
#             result = []
#             loss = 0
#             for i in range(len(out_list)):
#                 result.append(out_list[i] + fout_list[i])
#             for j in range(len(label_list)):
#                 loss += F.nll_loss(result[j], label_list[j].long(), reduction='sum').item()
#                 pred = result[j].argmax(dim=1, keepdim=True)
#                 correct += pred.eq(label_list[j].view_as(pred)).sum().item()
#
#                 total += label_list[j].size(0)
#                 pre_mask = torch.zeros(result[j].size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
#                 predict_num += pre_mask.sum(0)  # TP+FP
#                 tar_mask = torch.zeros(result[j].size()).scatter_(1, label_list[j].cpu().view(-1, 1).long(), 1.)
#                 target_num += tar_mask.sum(0)  # TP+FN
#                 acc_mask = pre_mask * tar_mask
#                 acc_num += acc_mask.sum(0)  # TP
#
#             test_loss /= len(validate_loader.dataset)
#             recall = acc_num / target_num
#             precision = acc_num / predict_num
#             F1 = 2 * recall * precision / (recall + precision)
#             accuracy = acc_num.sum(1) / target_num.sum(1)
#
#             # 精度调整
#             recall = (recall.numpy()[0] * 100).round(3)
#             precision = (precision.numpy()[0] * 100).round(3)
#             F1 = (F1.numpy()[0] * 100).round(3)
#             accuracy = (accuracy.numpy()[0] * 100).round(3)
#
#             # 打印格式方便复制
#             print('testSize ：{}'.format(len(validate_loader.dataset)))
#             print('Recall (%)  ', " ".join('%s' % id for id in recall))
#             print('Precision(%)', " ".join('%s' % id for id in precision))
#             print('F1 (%)      ', " ".join('%s' % id for id in F1))
#             print('TPR(Recall) ', TPR)
#             print('FPR         ', FPR)
#             print('accuracy(%) ', accuracy)
#
#             if accuracy > best_accuracy:
#                 best_accuracy = accuracy
#                 # 保存模型
#                 save_dir = './models/USTC'
#                 os.makedirs(save_dir, exist_ok=True)
#                 best_model_filename = os.path.join(save_dir, 'Meta_BiTCN_best_USTC_' + str(frequency) + '.pth')
#                 torch.save(model_meta.state_dict(), best_model_filename)
#                 print(f'Saved new best model with accuracy {best_accuracy:.3f}% as {best_model_filename}')
#         # 指定保存模型的目录
#         save_dir = './models/USTC'
#         os.makedirs(save_dir, exist_ok=True)
#         save_filename = os.path.join(save_dir, 'Meta_BiTCN_best_USTC_' + str(frequency) + '.pth')
#         # 保存模型的
#         torch.save(model_meta.state_dict(), save_filename)
#         print('Saved as %s' % save_filename)


def meta_predict(stacking_train, labels, input_size, num_classes, device, dataset_name):
    global all_preds
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
    for ep in range(Epoch):
        print(f'L Epoch: {ep}')
        model_meta = meta_BiTCN(in_channels=input_size, num_classes=num_classes, num_channels=8 * [32], dropout=0.05,
                                kernel_size=3)
        model_path = f'./models/{dataset_name}/Meta_BiTCN_best_{dataset_name}_{ep}.pth'
        model_state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model_meta.load_state_dict(model_state_dict)
        model_meta.to(device)
        model_meta.eval()
        all_preds = []
        all_labels = []
        print("Predicting BiTCN\n")
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # inputs = inputs.view(inputs.size(0), inputs.size(1), inputs.size(2) * inputs.size(3))

                outputs = model_meta(inputs)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            TPR, FPR = calculate_tpr_fpr_multiclass(all_labels, all_preds, num_classes)

            report = classification_report(all_labels, all_preds, output_dict=True, zero_division=1)
            accuracy = report['accuracy']
            precision_weighted = report['weighted avg']['precision']
            recall_macro = report['weighted avg']['recall']
            f1_score_weighted = report['weighted avg']['f1-score']
            print(
                f'Precision: {precision_weighted:.4f}, Recall: {recall_macro:.4f}, F1 Score: {f1_score_weighted:.4f}, ')
            print(f'TPR: {TPR:.4f}, FPR: {FPR:.4f}')
            print(f'Accuracy: {accuracy:.4f}')

            if ep == 0:
                # 计算混淆矩阵
                cm = confusion_matrix(all_labels, all_preds)
                # 打印混淆矩阵
                print("Confusion Matrix:")
                print(cm)
                # 保存混淆矩阵到文件
                cm_file_path = f'{dataset_name}_BiTCN_confusion_matrix.csv'
                np.savetxt(cm_file_path, cm, delimiter=',', fmt='%d')
                print(f"Confusion matrix saved to {cm_file_path}")

            accuracies.append(accuracy)
            f1_scores.append(f1_score_weighted)
            tprs.append(TPR)
            fprs.append(FPR)

    accuracies = np.array(accuracies)
    f1_scores = np.array(f1_scores).flatten()
    fprs = np.array(fprs).flatten()
    tprs = np.array(tprs).flatten()
    print(f"accuracies: {accuracies}")
    print(f"f1_scores: {f1_scores}")
    print(f"tprs: {tprs}")
    print(f"fprs: {fprs}")

    print('Mean accuracy  (%)  ', f'{np.mean(accuracies):.4f}±{np.std(accuracies):.4f}')
    print('Mean FPR            ', f'{np.mean(fprs):.4f}±{np.std(fprs):.4f}')
    print('Mean TPR            ', f'{np.mean(tprs):.4f}±{np.std(tprs):.4f}')
    print('Mean F1 score  (%)  ', f'{np.mean(f1_scores):.4f}±{np.std(f1_scores):.4f}')

    csv_file_path = f'./predict/{dataset_name}_meta_BiTCN.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration'] + [str(i + 1) for i in range(10)])
        writer.writerow(['Accuracy'] + [f"{acc:.4f}" for acc in accuracies])
        writer.writerow(['TPR'] + [f"{tpr:.4f}" for tpr in tprs])
        writer.writerow(['FPR'] + [f"{fpr:.4f}" for fpr in fprs])
        writer.writerow(['F1 Score'] + [f"{f1:.4f}" for f1 in f1_scores])

    print(f"Results saved to {csv_file_path}")
    return all_preds


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # stacking_train = np.load('../USTC_stacking_train.npy')
    # stacking_train_reshaped = stacking_train.reshape(stacking_train.shape[0], stacking_train.shape[1], 1)
    # labels = np.load('../USTC_train_labels.npy')
    #
    num_classes = 10  # USTC
    epochs = 20
    in_channels = 3
    # fit_bitcn(stacking_train_reshaped, labels, input_size, num_classes, dataset_name, epochs)
    train_loader, validate_loader, _, _, _, dataset_name = data_pre_process(os.path.join(os.getcwd(), "../"),
                                                                            '4_Png_16_ISAC')
    model = meta_BiTCN(in_channels=in_channels, num_classes=num_classes, num_channels=8 * [25], dropout=0.05,
                       kernel_size=7)
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_acc = 0
    for i in range(epochs):  # 10个epoch即可0.99
        print(f'Epoch : {i}')
        train(model, train_loader, criterion=criterion, optimizer=optimizer, device=device, num_class=num_classes)
        accuracy = validate(model, validate_loader, criterion=criterion, device=device, num_class=num_classes,
                            dataset_name=dataset_name)[1]
        model_path = f'../models/ISAC/BiTCN_for_meta.pt'
        if best_acc < accuracy:
            best_acc = accuracy
            torch.save(model, model_path)
            print(f'Model saved to {model_path}')


if __name__ == "__main__":
    main()
