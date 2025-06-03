import csv
import json
import os
import time
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

from FocalLoss import FocalLoss
from utils import data_pre_process, calculate_tpr_fpr, calculate_tpr_fpr_multiclass

# hyper parameters
num_epoches = 5
BATCH_SIZE = 128  # 批训练的数量
TIME_STEP = 1024  # 相当于序列长度（seq），等于单个图像的像素数，对于32x32的图像，就是1024（32 * 32）
INPUT_SIZE = 32  # 特征向量长度
LR = 0.001  # learning rate
class_num = 12


# 定义网络模型
class LSTM(nn.Module):
    def __init__(self, class_nums: int):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(Input_size=16,  # if use nn.LSTM_model(), it hardly learns
                           hidden_size=64,  # BiLSTM 隐藏单元
                           num_layers=1,  # BiLSTM 层数
                           batch_first=True,
                           # input & output will have batch size as 1s dimension. e.g. (batch, seq, Input_size)
                           )
        self.out = nn.Linear(64, out_features=class_nums)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        return x


# LSTM模型
class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))

        # 解码最后一个时间步的隐藏状态
        out = self.fc(out[:, -1, :])
        return out


# 结合CNN和LSTM的整体模型
class CNNtoLSTM(nn.Module):
    def __init__(self):
        super(CNNtoLSTM, self).__init__()
        self.cnn = CNN()
        self.lstm = LSTM_model(input_size=32 * 32, hidden_size=128, num_layers=1, num_classes=class_num)

    def forward(self, x):
        # 通过CNN
        x = self.cnn(x)

        # 将输出展平为 (batch_size, seq_length, Input_size)
        x = x.view(x.size(0), -1, 32 * 32)  # 假定CNN输出为(batch_size, channels, height, width)

        # 通过LSTM
        x = self.lstm(x)
        return x


# 元学习器
class LSTM_meta(nn.Module):
    def __init__(self, class_nums: int, input_size: int):
        super(LSTM_meta, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size,  # if use nn.LSTM(), it hardly learns
                           hidden_size=64,  # BiLSTM 隐藏单元
                           num_layers=1,  # BiLSTM 层数
                           batch_first=True,
                           # input & output will have batch size as 1s dimension. e.g. (batch, seq, Input_size)
                           )
        self.out = nn.Linear(64, out_features=class_nums)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


def load_model(model_path: str, Class_nums: int, device='cuda'):
    # 确保device是正确的类型
    if device == 'cuda':
        map_location = None  # 如果是cuda，则不需要指定map_location
    elif device == 'cpu':
        map_location = 'cpu'  # 如果是cpu，则指定map_location为'cpu'
    else:
        raise ValueError("Invalid device type. Expected 'cuda' or 'cpu'.")
    model = CNNtoLSTM()  # 实例化模型
    model.load_state_dict(torch.load(model_path, map_location=map_location))  # 加载模型的状态字典
    model.to(device)
    model.eval()  # 设置模型为评估模式
    return model


def predict(model_path: str, data_loader, Class_nums: int, device='cuda'):
    print("\nNow is LSTM_model predictions")
    # model = load_model(model_path, Class_nums, device)  # Load the model
    model = torch.load(model_path, map_location=device)
    predict_labels = []
    true_labels = []  # To store the true labels

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(imgs)
            _, pred = torch.max(outputs, 1)

            predict_labels.extend(pred.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate Precision, Recall, and F1 Score
    report = classification_report(true_labels, predict_labels, output_dict=True, zero_division=0)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    TPR, FPR = calculate_tpr_fpr(json_filepath='class_indices.json', true_labels=true_labels,
                                 pred_labels=predict_labels)
    print('Classification Report:')
    print(classification_report(true_labels, predict_labels, zero_division=0))
    print(f"Precision: {precision:.6f}, ",
          f"Recall: {recall:.6f}, ",
          f"F1 Score: {f1_score:.6f}")
    print(f"TPR: {TPR:.6f}, FPR: {FPR:.6f}")

    return predict_labels


def fit(num_classes, train_loader: Union[DataLoader, np.ndarray], input_size: int, device: str = 'cuda',
        epochs: int = 10, Png_flag: bool = True, dataset_name="",
        stacking_train: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None):
    """
    训练模型。

    参数:
    - model (nn.Module): 模型实例。
    - train_loader (DataLoader or np.ndarray): 训练数据加载器或 NumPy 数组。
    - Input_size (int ) : 第二层 LSTM_meta中的输入参数。
    - device (str): 设备类型 ('cuda' 或 'cpu')，默认为 'cuda'。
    - epochs (int): 训练轮数，默认为 5。
    - Png_flag (bool): 输入数据类型，True 表示图像，False 表示 NumPy 数组，默认为 True。
    - stacking_train (np.ndarray, optional): 用于堆叠训练的 NumPy 数组，默认为 None。
    - labels (np.ndarray, optional): 对应的标签，默认为 None。
    """

    print('Using NumPy array as input (Now lstm is meta learner)')
    if stacking_train is None or labels is None:
        raise ValueError("Both stacking_train and labels must be provided when Png_flag is False.")

    if isinstance(labels, dict):  # 如果 labels 是字典，提取其值
        labels = np.array(list(labels.values()))

    # 将NumPy数组转换为Tensor并创建DataLoader
    stacking_train_tensor = torch.tensor(stacking_train, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    train_dataset = TensorDataset(stacking_train_tensor, labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for ep in range(10):
        best_acc = 0.0
        print(f'L Epoch {ep + 1}')

        model_meta = LSTM_meta(class_nums=num_classes, input_size=input_size)
        model_meta = model_meta.to(device)
        meta_optimizer = torch.optim.Adam(model_meta.parameters(), lr=0.003)
        meta_criterion = nn.CrossEntropyLoss()
        # meta_criterion = FocalLoss()
        model_meta.train()

        for epoch in range(epochs):
            model_meta.train()
            print(f'Epoch {epoch + 1}')
            print('*' * 10)
            running_loss = 0.0
            running_acc = 0.0
            all_preds = []
            all_labels = []
            start_train_time = time.perf_counter()

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                outputs = model_meta(inputs)
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

            train_time = (time.perf_counter() - start_train_time)
            print(f"Train Time: {train_time:.2f}s")
            print(
                f'Finish {epoch + 1} epoch, Loss: {running_loss / len(train_loader.dataset):.6f}, Acc: {running_acc / len(train_loader.dataset):.6f}')

            validate(model_meta, device, train_loader, meta_criterion, num_classes)

            # 计算指标
            report = classification_report(all_labels, all_preds, output_dict=True, zero_division=1)
            accuracy = report.get('accuracy')
            TPR, FPR = calculate_tpr_fpr_multiclass(all_labels, all_preds, n_classes=num_classes)
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1_score = report['weighted avg']['f1-score']

            # 打印格式方便复制
            print(f'accuracy: {accuracy:.6f},Precision: {precision:.6f}, Recall: {recall:.6f}, F1 Score: {f1_score:.6f}')
            print('TPR: {:.6f}, FPR: {:.6f}'.format(TPR, FPR))
            print('Classification Report:')
            # print(classification_report(all_labels, all_preds, zero_division=1))

            if best_acc < accuracy:
                model_name = 'meta_LSTM_best_' + str(ep) + '.pt'
                print('New best model with accuracy {:.3f}% saved as {}, epoch{}'.format(accuracy * 100, model_name,epoch+1))
                save_model(model_meta, model_name=model_name, save_dir=f'models/{dataset_name}')


def validate(model, device, val_loader, criterion, num_classes):
    model.eval()
    running_val_loss = 0.0
    running_val_acc = 0.0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * labels.size(0)

            # 计算准确率
            _, preds = torch.max(outputs, 1)
            running_val_acc += (preds == labels).sum().item()

            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    val_accuracy = running_val_acc / len(val_loader.dataset)
    val_f1_score = f1_score(all_val_labels, all_val_preds, average='weighted')
    TPR, FPR = calculate_tpr_fpr_multiclass(all_val_labels, all_val_preds, num_classes)
    print(f'Validation Accuracy: {val_accuracy:.6f}')
    print(f'Validation F1 Score: {val_f1_score:.6f}')
    print(f"Validation TPR: {TPR}, Validation FPR: {FPR}")


def meta_predict(stacking_train, labels, num_classes, device, input_size, dataset_name):
    f1_scores = []
    fprs = []
    tprs = []
    accuracies = []

    stacking_train_tensor = torch.tensor(stacking_train, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    dataloader = DataLoader(TensorDataset(stacking_train_tensor, labels_tensor), batch_size=32, shuffle=False)
    for ep in range(10):
        print("test :{}".format(ep))
        model_path = f"./models/{dataset_name}/meta_LSTM_best_{ep}.pt"

        model_meta = LSTM_meta(class_nums=num_classes, input_size=input_size)
        model_meta = model_meta.to(device)
        model_meta.load_state_dict(torch.load(model_path))
        model_meta.eval()

        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                outputs = model_meta(inputs)
                # 计算准确率
                _, preds = torch.max(outputs, 1)

                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_f1_score = f1_score(all_val_labels, all_val_preds, average='weighted')
        TPR, FPR = calculate_tpr_fpr_multiclass(all_val_labels, all_val_preds, num_classes)
        print(f"Accuracy: {accuracy:.6f}, F1 Score: {val_f1_score:.6f}, TPR: {TPR:.6f}, FPR score: {FPR:.6f}")

        if ep == 0:
            # 计算混淆矩阵
            cm = confusion_matrix(all_val_labels, all_val_preds)
            # 打印混淆矩阵
            print("Confusion Matrix:")
            print(cm)
            # 保存混淆矩阵到文件
            cm_file_path = f'{dataset_name}_LSTM_confusion_matrix.csv'
            np.savetxt(cm_file_path, cm, delimiter=',', fmt='%d')
            print(f"Confusion matrix saved to {cm_file_path}")

        accuracies.append(accuracy)
        fprs.append(FPR)
        tprs.append(TPR)
        f1_scores.append(val_f1_score)

    accuracies = np.array(accuracies)
    f1_scores = np.array(f1_scores).flatten()
    fprs = np.array(fprs).flatten()
    tprs = np.array(tprs).flatten()
    print("accuracy: ", accuracies)
    print("fprs: ", fprs)
    print("tprs: ", tprs)
    print("f1_scores:", f1_scores)

    print('Mean accuracy  (%)  ', f'{np.mean(accuracies):.4f}±{np.std(accuracies):.4f}')
    print('Mean FPR            ', f'{np.mean(fprs):.4f}±{np.std(fprs):.4f}')
    print('Mean TPR            ', f'{np.mean(tprs):.4f}±{np.std(tprs):.4f}')
    print('Mean F1 score  (%)  ', f'{np.mean(f1_scores):.4f}±{np.std(f1_scores):.4f}')

    csv_file_path = f'./predict/{dataset_name}_Meta_LSTM.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration'] + [str(i + 1) for i in range(10)])
        writer.writerow(['Accuracy'] + [f"{acc:.4f}" for acc in accuracies])
        writer.writerow(['TPR'] + [f"{tpr:.4f}" for tpr in tprs])
        writer.writerow(['FPR'] + [f"{fpr:.4f}" for fpr in fprs])
        writer.writerow(['F1 Score'] + [f"{f1:.4f}" for f1 in f1_scores])

    print(f"Results saved to {csv_file_path}")
    return preds


def save_model(model, model_name: str = 'lstm-model.pt', save_dir='models'):
    # 指定保存模型的目录   save_dir
    # 创建目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    # 构建完整的文件路径
    save_file = os.path.join(save_dir, model_name)
    # 保存模型的状态字典
    torch.save(model.state_dict(), save_file)
    print('Saved as %s' % save_file)


# 主程序入口
def main():  # 数据集位置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    directory_path = os.path.join(os.path.abspath(os.getcwd()), "../pre-processing/4_Png_16_ISAC/Train")
    entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    class_nums = folder_count
    train_loader, validate_loader, label, _, _, dataset_name = data_pre_process(os.getcwd(), "4_Png_16_ISAC")

    EP = 10
    for i in range(EP):
        model__path = f'../models/LSTM_model_best_{dataset_name}_{i}.pt'
        predictions = predict(model__path, validate_loader, class_nums, device)

        print("shape of predictions:", len(predictions))


if __name__ == '__main__':
    main()
