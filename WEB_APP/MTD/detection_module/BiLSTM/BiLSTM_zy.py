import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from FocalLoss import FocalLoss
from utils import data_pre_process, calculate_tpr_fpr  # 假设这是你的数据预处理函数
from sklearn.metrics import confusion_matrix


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_rate=0.01):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.elu = nn.ELU()  # 添加 ELU 激活函数
        self.dropout = nn.Dropout(dropout_rate)  # 添加 Dropout 层
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 3 * 16 * 16)  # X变为(batch_size, seq_len, input_size)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])
        out = self.dropout(out)  # 应用 Dropout 层
        out = self.elu(out)  # 应用 ELU 激活函数
        return out


# 训练函数
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    log_interval = 1000
    step = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        step += 1
        if step % log_interval == 0:
            print(f'Step: {step}, Loss: {loss.item():.4f}')
        running_loss += loss.item()
    return running_loss / len(train_loader)


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


# 验证/测试函数
def evaluate_model(model, data_loader, device, dataset_name, mode='Validation'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    tpr = recall_score(all_labels, all_preds, average='weighted')
    fpr = 1 - precision_score(all_labels, all_preds, average='weighted')

    # TPR, FPR = calculate_tpr_fpr(dataset_name + "class_indices.json", all_labels, all_preds)
    # 计算每个类别的TPR和FPR
    tpr_list, fpr_list, support_list = calculate_tpr_fpr_multiclass(all_labels, all_preds, n_classes=10)

    # 计算加权平均的TPR和FPR
    total_support = sum(support_list)
    weighted_tpr = sum(tpr * support for tpr, support in zip(tpr_list, support_list)) / total_support
    weighted_fpr = sum(fpr * support for fpr, support in zip(fpr_list, support_list)) / total_support

    print(f'{mode} Weighted TPR: {weighted_tpr}, Weighted FPR: {weighted_fpr}')
    print(f"Accuracy: {acc:.4f}")
    print(f"weighted tpr : {tpr}, fpr : {fpr}")
    return acc, f1, weighted_tpr, weighted_fpr


# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 参数设置
    png_dir = "4_Png_16_ISAC"

    hidden_size = 64
    num_layers = 2
    directory_path = os.path.join(os.path.abspath(os.getcwd()), F"../pre-processing/{png_dir}/Train")
    entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    output_size = folder_count
    seq_len = 16  # 假设seq_len是16，根据实际情况调整
    input_size = 3 * 16 * 16
    learning_rate = 0.001

    data_dir = os.path.join(os.getcwd(), "../")
    train_loader, test_loader, labels, _, _, dataset_name = data_pre_process(data_dir, png_dir, None)

    EP = 10
    num_epochs = 5
    for ep in range(1):
        print(f'\t EPOCH {ep + 1}/{EP}')

        model = BiLSTM(output_dim=output_size, input_dim=input_size, hidden_dim=hidden_size, layer_dim=num_layers)
        model.to(device)
        # criterion = nn.CrossEntropyLoss()
        criterion = FocalLoss(alpha=0.75, gamma=2)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        best_acc = 0.0
        for epoch in range(num_epochs):
            print(f'epoch {epoch + 1}/{num_epochs}')
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            print(f'Train Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}')

            # 验证
            val_acc, val_f1, val_tpr, val_fpr = evaluate_model(model, test_loader, device, dataset_name)
            print(f'Validation - Acc: {val_acc:.4f}, F1: {val_f1:.4f}, TPR: {val_tpr:.4f}, FPR: {val_fpr:.4f}')

        # 测试
        test_acc, test_f1, test_tpr, test_fpr = evaluate_model(model, test_loader, device, dataset_name, mode='Test')
        print(f'Test - Acc: {test_acc:.4f}, F1: {test_f1:.4f}, TPR: {test_tpr:.4f}, FPR: {test_fpr:.4f}')

        # save_dir = f'../models/{dataset_name[:-1]}/BiLSTM_best_{dataset_name[:-1]}_{ep}.pth'
        # if best_acc < test_acc:
        #     best_acc = test_acc
        #     torch.save(model.state_dict(), save_dir)
        #     print(f'Best Model saved to {save_dir}, accuracy {best_acc}, epoch{ep + 1}')


def calculate_means_and_stds():
    png_dir = '4_Png_16_ISAC'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directory_path = os.path.join(os.path.abspath(os.getcwd()), F"../pre-processing/{png_dir}/Train")
    entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    output_size = folder_count
    hidden_size = 64
    num_layers = 2
    learning_rate = 0.001
    seq_len = 16  # 假设seq_len是16，根据实际情况调整
    input_size = 16 * 16 * 3
    # _, test_loader = data_loader(data_path)
    _, test_loader, _, _, _, dataset_name = data_pre_process(data_directory=os.path.join(os.getcwd(), '../'),
                                                             png_path=png_dir)

    f1_scores = []
    fprs = []
    tprs = []
    accuracies = []
    for i in range(10):
        model_path = f'../models/{dataset_name[:-1]}/BiLSTM_best_{dataset_name}{i}.pth'
        model = BiLSTM(output_dim=output_size, input_dim=input_size, hidden_dim=hidden_size, layer_dim=num_layers).to(
            device)
        model.load_state_dict(torch.load(model_path))
        accuracy, f1, tpr, fpr = evaluate_model(model, test_loader, device, dataset_name)

        accuracies.append(accuracy)
        f1_scores.append(f1)
        fprs.append(fpr)
        tprs.append(tpr)

    # 转换为 NumPy 数组
    f1_scores = np.array(f1_scores)
    fprs = np.array(fprs)
    accuracies = np.array(accuracies)
    tprs = np.array(tprs)

    # 汇总所有类别的指标值
    all_f1_scores = f1_scores.flatten()
    all_fprs = fprs.flatten()
    all_tprs = tprs.flatten()

    # 计算均值和标准差
    mean_f1_score = np.mean(all_f1_scores)
    std_f1_score = np.std(all_f1_scores)

    mean_tpr = np.mean(all_tprs)
    std_tpr = np.std(all_tprs)

    mean_fpr = np.mean(all_fprs)
    std_fpr = np.std(all_fprs)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    # 打印结果
    print('Mean accuracy  (%)  ', f'{mean_accuracy:.4f}±{std_accuracy:.4f}')
    print('Mean FPR            ', f'{mean_fpr:.4f}±{std_fpr:.4f}')
    print('Mean TPR            ', f'{mean_tpr:.4f}±{std_tpr:.4f}')
    print('Mean F1 score  (%)  ', f'{mean_f1_score:.4f}±{std_f1_score:.4f}')


if __name__ == '__main__':
    main()
    # calculate_means_and_stds()
