import os

from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from FocalLoss import FocalLoss
from utils import data_pre_process, calculate_tpr_fpr


class BiTCN(nn.Module):
    def __init__(self, num_classes, input_size, num_channels, kernel_size=3, dropout=0.01):
        super(BiTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                 padding=(kernel_size - 1) * dilation_size // 2),
                       nn.ELU(),
                       nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], num_classes)  # Assuming binary classification

    def forward(self, x):
        x = self.network(x)
        x = x.mean(dim=-1)  # Global average pooling
        x = self.linear(x)
        return x.squeeze()


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    input_channels = 3  # 输入通道
    seq_length = int(768 / input_channels)  # 序列长度

    log_interval = 1000
    step = 0
    for inputs, labels in train_loader:
        inputs = inputs.view(-1, input_channels, seq_length)  # (64,3,16,16) batch_size, channels, seq_length, height
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # outputs -> 混淆矩阵
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        step += 1
        if step % log_interval == 0:
            print(f'Step: {step}, Loss: {loss.item():.4f}')
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Training Loss: {epoch_loss:.4f}')


def calculate_tpr_fpr_multiclass(y_true, y_pred, n_classes):
    """
    计算多分类问题的TPR和FPR

    :param y_true: 真实标签，numpy数组或列表
    :param y_pred: 预测标签，numpy数组或列表
    :param n_classes: 类别数量
    :return: 每个类别的TPR和FPR列表，以及每个类别的样本数
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    tpr_list = []
    fpr_list = []
    support_list = []

    for i in range(n_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = np.sum(cm) - (TP + FP + FN)

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

        tpr_list.append(TPR)
        fpr_list.append(FPR)
        support_list.append(TP + FN)  # 支持度，即每个类别的样本数

    return tpr_list, fpr_list, support_list


def evaluate(model, data_loader, criterion, device,n_classes, mode='Validation', ):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_outputs = []  # 新增：累积所有批次的 outputs
    input_channels = 3
    seq_length = int(768 / input_channels)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, input_channels, seq_length)

            outputs = model(inputs)
            if mode == 'Validation':
                loss = criterion(outputs, labels.long())
                running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()  # 使用 argmax 获取预测类别
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().flatten())
            all_outputs.append(outputs.clone().detach().cpu())  # 累积 outputs

    epoch_loss = running_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')  # 使用加权平均

    # 计算每个类别的TPR和FPR
    tpr_list, fpr_list, support_list = calculate_tpr_fpr_multiclass(all_labels, all_preds, n_classes)

    # 计算加权平均的TPR和FPR
    total_support = sum(support_list)
    weighted_tpr = sum(tpr * support for tpr, support in zip(tpr_list, support_list)) / total_support
    weighted_fpr = sum(fpr * support for fpr, support in zip(fpr_list, support_list)) / total_support

    print(f'{mode} Weighted TPR: {weighted_tpr}, Weighted FPR: {weighted_fpr}')
    print(f"Accuracy: {accuracy:.4f}")

    # 多分类 ROC AUC
    if n_classes > 2:
        all_outputs = torch.cat(all_outputs, dim=0)  # 将所有批次的 outputs 合并成一个张量
        one_hot_labels = torch.nn.functional.one_hot(torch.tensor(all_labels), num_classes=n_classes).numpy()
        softmax_outputs = torch.nn.functional.softmax(all_outputs, dim=1).numpy()
        roc_auc = roc_auc_score(one_hot_labels, softmax_outputs, multi_class='ovr')
    else:
        all_outputs = torch.cat(all_outputs, dim=0)  # 将所有批次的 outputs 合并成一个张量
        sigmoid_outputs = torch.sigmoid(all_outputs).numpy()
        roc_auc = roc_auc_score(all_labels, sigmoid_outputs)

    print(
        f'{mode} Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}')
    print(classification_report(all_labels, all_preds, zero_division=1))
    return all_preds, all_labels


def test(model, test_loader, criterion, device, n_classes):
    preds, labels = evaluate(model, test_loader, criterion, device, mode='Test',n_classes=n_classes)

    report = classification_report(labels, preds, output_dict=True, zero_division=1)  # {‘Accuracy:’：25%,'2':50…………}
    accuracy = report.get('accuracy')
    F1_score = report.get('weighted avg').get('f1-score')
    recall = report.get('weighted avg').get('recall')
    # TPR, FPR = calculate_tpr_fpr("class_indices.json", labels, preds)
    # 计算每个类别的TPR和FPR
    tpr_list, fpr_list, support_list = calculate_tpr_fpr_multiclass(labels, preds, n_classes=10)

    # 计算加权平均的TPR和FPR
    total_support = sum(support_list)
    weighted_tpr = sum(tpr * support for tpr, support in zip(tpr_list, support_list)) / total_support
    weighted_fpr = sum(fpr * support for fpr, support in zip(fpr_list, support_list)) / total_support
    print(f'TPR: {weighted_tpr:.6f}, FPR: {weighted_fpr:.6f}')
    return accuracy, F1_score, recall, weighted_tpr, weighted_fpr


def main():
    # Hyperparameters
    png_path = '4_Png_16_USTC'  # png train and test data path

    directory_path = os.path.join(os.path.abspath(os.getcwd()), f"../pre-processing/{png_path}/Train")
    entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    n_classes = folder_count  # 分类数
    input_size = 3  # Number of features per time step (匹配input_channels)
    num_channels = [64, 128, 256]  # Number of channels in each layer
    kernel_size = 3  # Kernel size for each convolutional layer
    dropout = 0.01
    learning_rate = 0.0015

    dir = os.path.join(os.getcwd(), '../')
    train_loader, test_loader, _, _, _, dataset_name = data_pre_process(dir, png_path, None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiTCN(n_classes, input_size, num_channels, kernel_size, dropout).to(device)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(gamma=4, alpha=0.75)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    EP = 10
    num_epochs = 3
    for ep in range(0, EP):
        print(f'\t EPOCH {ep + 1}/{EP}')
        best_acc = 0.0
        for epoch in range(num_epochs):
            print(f'epoch {epoch + 1}/{num_epochs}')
            train(model, train_loader, criterion, optimizer, device)
            evaluate(model, test_loader, criterion, device, n_classes=n_classes)

        accuracy, F1_score, recall, TPR, FPR = test(model, test_loader, criterion, device,n_classes=n_classes)
        save_dir = f'../models/{dataset_name[:-1]}/BiTCN_best_{dataset_name}' + str(ep) + '.pt'
        if best_acc < accuracy:
            best_acc = accuracy
            torch.save(model, save_dir)
            print(f'Best Model saved to {save_dir}, epoch{ep + 1},best accuracy {best_acc}')


def calculate_means_and_stds():
    png_path = '4_Png_16_USTC'
    directory_path = os.path.join(os.path.abspath(os.getcwd()), f"../pre-processing/{png_path}/Train")
    entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    n_classes = folder_count  # 分类数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dire = os.path.join(os.path.abspath(os.getcwd()), '../')
    train_loader, test_loader, _, _, _, dataset_name = data_pre_process(dire, png_path, None)

    # 初始化存储指标的列表
    accuracies = []
    f1_scores = []
    tprs = []
    fprs = []
    for i in range(10):
        # model = BiTCN(n_classes, input_size, num_channels, kernel_size, dropout).to(device)
        model_path = f'../models/{dataset_name[:-1]}/BiTCN_best_{dataset_name}' + str(i) + '.pt'
        print(f"Loading model {model_path}")
        # Load the BiTCN
        model = torch.load(model_path)
        model.to(device)

        acc, f1, recall, tpr, fpr = test(model, test_loader, device=device, criterion=optim.Adam,n_classes=n_classes)

        accuracies.append(acc)
        f1_scores.append(f1)
        tprs.append(tpr)
        fprs.append(fpr)

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


# Example usage
if __name__ == '__main__':
    main()
    # calculate_means_and_stds()
