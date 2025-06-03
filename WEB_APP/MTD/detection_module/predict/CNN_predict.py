import csv
import json
import os
import numpy as np
from torchvision import transforms, datasets
import torch.nn.functional as F  # 导入 F 模块以使用 softmax 函数
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from FocalLoss import FocalLoss
from utils import data_pre_process, calculate_tpr_fpr, calculate_tpr_fpr_multiclass


# 定义卷积神经网络
class CNN(nn.Module):
    def __init__(self, class_nums):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True)  # 替换为 ELU
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),  # 替换为 ELU
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),  # 替换为 ELU
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 假设经过两次MaxPooling后，每个特征图的大小变为4x4
        self.out = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1024),  # 调整这里以匹配新的输入尺寸
            nn.ELU(inplace=True),  # 替换为 ELU
            nn.Linear(1024, 128),
            nn.ELU(inplace=True),  # 替换为 ELU
            nn.Linear(128, out_features=class_nums)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.out(x)
        return x


class CNN_USTC(nn.Module):
    def __init__(self, class_nums):
        super(CNN_USTC, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True)  # 替换为 ELU
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),  # 替换为 ELU
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ELU(inplace=True),  # 替换为 ELU
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # 假设经过两次MaxPooling后，每个特征图的大小变为4x4
        self.out = nn.Sequential(
            nn.Linear(32 * 8 * 8, 1024),  # 调整这里以匹配新的输入尺寸
            nn.ELU(inplace=True),  # 替换为 ELU
            nn.Dropout(0.02),  # 在第一个全连接层后添加Dropout层
            nn.Linear(1024, 128),
            nn.ELU(inplace=True),  # 替换为 ELU
            nn.Dropout(0.02),
            nn.Linear(128, out_features=class_nums)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.out(x)
        return x


# 定义第二层学习器的神经网络
class StackingNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(StackingNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.elu = nn.ELU()  # 实例化 ELU 激活函数

    def forward(self, x):
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.fc3(x)
        return x


def fit(stacking_train: np.ndarray, labels: np.ndarray, dataset_name, num_classes: int, input_size: int, device='cuda',
        epochs=5):
    """
    训练第二层学习器。

    参数:
        stacking_train (np.ndarray): 第一层模型的预测标签矩阵。
        labels (np.ndarray): 真实标签。
        num_classes (int): 类别数量。
        input_size (int): 输入特征的维度。
        device (torch.device): 设备（CPU 或 GPU）。
        epochs (int): 训练轮数，默认为 5。
    """

    # 训练模型
    EP = 10
    for ep in range(EP):  # train
        print(f"L Epoch: {ep}")
        # 将数据转换为 PyTorch 张量
        stacking_train_tensor = torch.tensor(stacking_train, dtype=torch.float32).to(device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

        # 创建数据集和数据加载器
        dataset = TensorDataset(stacking_train_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        # 初始化模型
        model = StackingNet(input_size, num_classes).to(device)
        # 定义损失函数和优化器
        criterion = FocalLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0003)

        best_acc = 0.0
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}')

            # 评估模型
            model.eval()
            all_outputs = []
            all_labels = []
            all_predictions = []

            with torch.no_grad():
                for inputs, targets in dataloader:
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
            report = classification_report(all_labels, all_predictions, output_dict=True, zero_division=0)
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1_score = report['weighted avg']['f1-score']
            TPR, FPR = calculate_tpr_fpr_multiclass(all_labels, all_predictions, n_classes=num_classes)
            print(
                f'Precision: {precision:.6f}, Recall: {recall:.6f}, F1 Score: {f1_score:.6f}, Accuracy: {accuracy:.6f}')
            print(f"FPR: {FPR}, TPR: {TPR}")
            save_file = f'./models/{dataset_name}/meta_CNN_best_{dataset_name}_' + str(ep) + '.pth'
            if best_acc < accuracy:
                best_acc = accuracy
                print(f"Best model saved at epoch {epoch + 1} with validation accuracy: {best_acc:.6f}")
                torch.save(model.state_dict(), save_file)
        print("Finished Training!")
        print('Best accuracy:', best_acc)


def meta_predict(stacking_train: np.ndarray, labels: np.ndarray, num_classes: int, input_size: int, device='cuda',
                 dataset_name=''):
    EP = 10
    accuracies = []
    tprs = []
    fprs = []
    f1_scores = []
    for ep in range(EP):  # train
        print(f"L Epoch: {ep}")
        model_path = f'./models/{dataset_name}/meta_CNN_best_{dataset_name}_' + str(ep) + '.pth'
        model_state_dict = torch.load(model_path, map_location=device)
        # 将数据转换为 PyTorch 张量
        stacking_train_tensor = torch.tensor(stacking_train, dtype=torch.float32).to(device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

        # 创建数据集和数据加载器
        dataset = TensorDataset(stacking_train_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        # 初始化模型
        model = StackingNet(input_size, num_classes).to(device)
        model.load_state_dict(model_state_dict)

        model.eval()
        all_outputs = []
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, targets in dataloader:
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
        TPR, FPR = calculate_tpr_fpr_multiclass(all_labels, all_predictions, n_classes=num_classes)

        if ep == 0:
            # 计算混淆矩阵
            cm = confusion_matrix(all_labels, all_predictions)
            # 打印混淆矩阵
            print("Confusion Matrix:")
            print(cm)
            # 保存混淆矩阵到文件
            cm_file_path = f'{dataset_name}_CNN_confusion_matrix.csv'
            np.savetxt(cm_file_path, cm, delimiter=',', fmt='%d')
            print(f"Confusion matrix saved to {cm_file_path}")

        # 计算其他指标
        report = classification_report(all_labels, all_predictions, output_dict=True, zero_division=0)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']
        print(
            f'Precision: {precision:.6f}, Recall: {recall:.6f}, F1 Score: {f1_score:.6f}, Accuracy: {accuracy:.6f}')
        accuracies.append(accuracy)
        fprs.append(FPR)
        tprs.append(TPR)
        f1_scores.append(f1_score)

    accuracies = np.array(accuracies)
    f1_scores = np.array(f1_scores).flatten()
    fprs = np.array(fprs).flatten()
    tprs = np.array(tprs).flatten()

    print("Accuracies:", accuracies)
    print("F1 Scores:", f1_scores)
    print("FPRs:", fprs)
    print("TPRs:", tprs)

    print('Mean accuracy  (%)  ', f'{np.mean(accuracies):.4f}±{np.std(accuracies):.4f}')
    print('Mean FPR            ', f'{np.mean(fprs):.4f}±{np.std(fprs):.4f}')
    print('Mean TPR            ', f'{np.mean(tprs):.4f}±{np.std(tprs):.4f}')
    print('Mean F1 score  (%)  ', f'{np.mean(f1_scores):.4f}±{np.std(f1_scores):.4f}')

    csv_file_path = f'{dataset_name}_meta_CNN.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration'] + [str(i + 1) for i in range(10)])
        writer.writerow(['Accuracy'] + [f"{acc:.4f}" for acc in accuracies])
        writer.writerow(['TPR'] + [f"{tpr:.4f}" for tpr in tprs])
        writer.writerow(['FPR'] + [f"{fpr:.4f}" for fpr in fprs])
        writer.writerow(['F1 Score'] + [f"{f1:.4f}" for f1 in f1_scores])

    print(f"Results saved to {csv_file_path}")
    return all_predictions


def load_model(model_path, device='cuda', class_nums=10):
    # 加载模型
    model = CNN(class_nums)
    model_state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()
    return model


def train_model(train_loader, epochs, class_nums, device):
    model = CNN(class_nums)
    model.to(device)
    loss_function = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    all_labels, all_predictions = [], []
    for epoch in tqdm(range(epochs)):
        log_interval = 1000  # 设置日志输出的间隔
        cnt = 0
        model.train()
        for img, label in train_loader:
            img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)
            out = model(img)

            loss = loss_function(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(out.argmax(dim=1).cpu().numpy())
            cnt += 1
            if cnt % log_interval == 0:
                print('*' * 10)
                print(f'epoch:{epoch + 1},cnt{cnt} loss is {loss.item():.4f}')

    report = classification_report(all_labels, all_predictions, output_dict=True, zero_division=0)
    print("accuracy:", report['accuracy'])
    print("precision:", report['weighted avg']['precision'])
    print("recall:", report['weighted avg']['recall'])
    print("f1_score:", report['weighted avg']['f1-score'])
    return model


def predict(model, test_loader, class_num, device, dataset_name):
    print(f'Testing CNN on {dataset_name}')
    # 加载模型
    model = model.to(device)
    model.eval()

    # loss_function = nn.CrossEntropyLoss()
    loss_function = FocalLoss()
    # 初始化变量
    eval_loss = 0
    eval_acc = 0
    all_outputs = []  # 用于保存预测输出
    all_labels = []  # 保存真实标签
    all_predictions = []  # 保存预测标签

    thresholds = [0] * class_num  # 假设每个类别的阈值相同，可以根据实际情况调整

    with torch.no_grad():
        for data in test_loader:
            img, label = data
            img, label = img.to(device), label.to(device)
            out = model(img)
            # 在测试阶段应用 softmax 函数
            probs = F.softmax(out, dim=1)
            all_outputs.append(probs.cpu().detach().numpy())
            all_labels.append(label.cpu().numpy())

            # 获取每个样本最大概率对应的类别索引
            max_probs, predictions = torch.max(probs, dim=1)
            # 应用阈值
            for i in range(len(predictions)):
                if max_probs[i] < thresholds[predictions[i]]:
                    predictions[i] = -1  # 将不确定的预测标记为-1或其他未定义类别
            all_predictions.append(predictions.cpu().numpy())

            loss = loss_function(out, label)
            eval_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.item()

    # 将所有批次的预测标签拼接成一个 NumPy 数组
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    dataset_len = len(all_labels)

    # 计算相关指标
    report = classification_report(all_labels, all_predictions, output_dict=True, zero_division=0)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    accuracy = eval_acc / dataset_len

    # TPR, FPR = calculate_tpr_fpr(dataset_name + "class_indices.json", all_labels, all_predictions)
    TPR, FPR = calculate_tpr_fpr_multiclass(all_labels, all_predictions, class_num)
    # 输出结果
    print(f"\nTest Loss: {eval_loss / dataset_len:.6f}, Acc: {eval_acc / dataset_len:.6f}")
    print(f"Test Tpr: {TPR:.6f}, FPR: {FPR:.6f}")
    print(f"F1 Score: {f1_score:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"Accuracy: {accuracy:.6f}\n")

    return {
        'accuracy': accuracy,
        'f1_score': f1_score,
        'tpr': TPR,
        'fpr': FPR,
        'predictions': all_predictions
    }
    # return


def main():
    # 获得数据集的类别数量  4_Png_16_USTC   4_Png_16_CTU  4_Png_16_ISAC
    PNG_PATH = '4_Png_16_ISAC'
    directory_path = os.path.join(os.path.abspath(os.getcwd()), f"../pre-processing/{PNG_PATH}/Train")
    entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    class_nums = folder_count  # 此参数为分类的数量，需要根据实际情况（数据集）修改

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path

    # 初始化存储指标的列表
    accuracies = []
    f1_scores = []
    tprs = []
    fprs = []
    _, test_loader, _, _, _, dataset_name = data_pre_process(data_root, PNG_PATH)  # 4_Png_16_ISAC

    for ep in range(10):
        # model_path = f"../models/ISAC/CNN_model_USTC__best_{ep}.pth"  # 模型文件的路径
        # 预处理数据
        print("Epoch {}/{}".format(ep + 1, folder_count))

        model = CNN(class_nums)
        model_path = f"../models/{dataset_name[:-1]}/CNN_model_{dataset_name}_best_{ep}.pth"  # ISAC CTU
        # model_path = f"../models/{dataset_name[:-1]}/CNN_model_best_{dataset_name[:-1]}_{ep}.pth"  # USTC
        model.load_state_dict(torch.load(model_path))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 使用给定的模型路径和测试数据加载器进行预测
        metrics = predict(model, test_loader, class_nums, device, dataset_name)
        # 收集指标
        accuracies.append(metrics['accuracy'])
        f1_scores.append(metrics['f1_score'])
        tprs.append(metrics['tpr'])
        fprs.append(metrics['fpr'])

    accuracies = np.array(accuracies)
    f1_scores = np.array(accuracies).flatten()
    fprs = np.array(fprs).flatten()
    tprs = np.array(tprs).flatten()

    # 输出结果
    print("Accuracies:", accuracies)
    print("Tprs:", tprs)
    print("Fprs:", fprs)
    print("F1 scores:", f1_scores)

    print('Mean accuracy  (%)  ', f'{np.mean(accuracies):.4f}±{np.std(accuracies):.4f}')
    print('Mean FPR            ', f'{np.mean(fprs):.4f}±{np.std(fprs):.4f}')
    print('Mean TPR            ', f'{np.mean(tprs):.4f}±{np.std(tprs):.4f}')
    print('Mean F1 score  (%)  ', f'{np.mean(f1_scores):.4f}±{np.std(f1_scores):.4f}')

    csv_file_path = f'{dataset_name}CNN.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration'] + [str(i + 1) for i in range(10)])
        writer.writerow(['Accuracy'] + [f"{acc:.4f}" for acc in accuracies])
        writer.writerow(['TPR'] + [f"{tpr:.4f}" for tpr in tprs])
        writer.writerow(['FPR'] + [f"{fpr:.4f}" for fpr in fprs])
        writer.writerow(['F1 Score'] + [f"{f1:.4f}" for f1 in f1_scores])

    print(f"Results saved to {csv_file_path}")


def main_train():
    print('training')
    DATA_TRAIN_PATH = '4_Png_16_USTC'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    directory_path = os.path.join(os.path.abspath(os.getcwd()), f"../pre-processing/{DATA_TRAIN_PATH}/Train")
    entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    class_nums = folder_count  # 此参数为分类的数量，需要根据实际情况（数据集）修改
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path

    train_loader, test_loader, labels, train_num, test_num, dataset_name = data_pre_process(data_root, DATA_TRAIN_PATH)

    best_accuracy = 0.0
    save_path = f"../models/{dataset_name}/CNN_for_meta.pt"
    model = train_model(train_loader=train_loader, epochs=10, device=device, class_nums=class_nums)
    accuracy = predict(model=model, test_loader=test_loader, class_num=class_nums, device=device,
                       dataset_name=dataset_name)['accuracy']
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model, save_path)


if __name__ == "__main__":
    main()
