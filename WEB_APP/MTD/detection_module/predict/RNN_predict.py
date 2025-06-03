import csv
import math
import os
from os.path import exists

import numpy as np
from torch import nn
from torch.amp import autocast
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import utils
from FocalLoss import FocalLoss
from utils import data_pre_process, calculate_tpr_fpr, calculate_tpr_fpr_multiclass
from sklearn.metrics import classification_report
import torch
import time
from sklearn.metrics import confusion_matrix
from Attention_rnn import Attention
import torch.nn.functional as F

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


# RNN with Attention
# class RNN(nn.Module):
#     def __init__(self, num_classes):
#         super(RNN, self).__init__()
#         self.rnn = nn.RNN(
#             input_size=3 * 16,  # 输入的图像尺寸为16*16的RGB图像
#             hidden_size=64,
#             num_layers=1,
#             batch_first=True,
#         )
#         self.attention = Attention(hidden_size=64)  # 添加注意力机制
#         self.out_layer = nn.Linear(64, num_classes)  # 输出层
#
#     def forward(self, x):
#         output, _ = self.rnn(x)
#         # 使用注意力机制
#         attn_output = self.attention(output)
#         prediction = self.out_layer(attn_output)
#         return prediction


class RNN(nn.Module):
    def __init__(self, num_classes, input_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,  # 输入的图像尺寸为32*32的RGB图像
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.elu = nn.ELU()  # 添加ELU层
        self.attention = Attention(hidden_size=64)  # 添加注意力机制
        self.out_layer = nn.Linear(64, num_classes)  # 输出层

    def forward(self, x):
        output, _ = self.rnn(x)
        # 关注最后一个时间步的输出
        last_output = output[:, -1, :]
        elu_output = self.elu(last_output)  # 应用ELU激活函数

        prediction = self.out_layer(elu_output)
        return prediction


class RNN_meta(nn.Module):
    def __init__(self, num_classes, input_size):
        super(RNN_meta, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,  # 输入的图像尺寸为32*32的RGB图像
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.attention = Attention(hidden_size=64)  # 添加注意力机制
        self.out_layer = nn.Linear(64, num_classes)  # 输出层

    def forward(self, x):
        output, _ = self.rnn(x)
        # 使用注意力机制
        attn_output = self.attention(output)
        prediction = self.out_layer(attn_output)
        return prediction

    # def forward(self, x):
    #     output, _ = self.rnn(x)
    #     # 关注最后一个时间步的输出
    #     last_output = output[:, -1, :]
    #     prediction = self.out_layer(last_output)
    #     return prediction


def save_model(model, model_name: str = 'model.pt', save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    # 构建完整的文件路径
    save_file = os.path.join(save_dir, model_name)
    # 保存模型的状态字典
    torch.save(model, save_file)
    print('Saved as %s' % save_file)


def fit(num_classes: int, input_size: int, dataset_name, stacking_train: np.ndarray = None, labels: np.ndarray = None,
        epochs: int = 3, Device='cuda'):
    print('Using NumPy array as input (Now RNN is meta learner)')
    if stacking_train is None or labels is None:
        raise ValueError("Both stacking_train and labels must be provided when Png_flag is False.")

    if isinstance(labels, dict):  # 如果 labels 是字典，提取其值
        labels = np.array(list(labels.values()))

    # 将NumPy数组转换为Tensor并创建DataLoader
    stacking_train_tensor = torch.tensor(stacking_train, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    train_dataset = TensorDataset(stacking_train_tensor, labels_tensor)
    Data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    EP = 10
    for ep in range(EP):
        print(f'L Epoch {ep}')
        best_acc = 0.0
        # 处理NumPy数组
        model_meta = RNN_meta(input_size=input_size, num_classes=num_classes)
        model_meta = model_meta.to(Device)
        meta_optimizer = torch.optim.Adam(model_meta.parameters(), lr=0.001)
        # meta_criterion = FocalLoss()
        meta_criterion = nn.CrossEntropyLoss()
        model_meta.train()

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}')
            print('*' * 10)
            running_loss = 0.0
            running_acc = 0.0
            all_preds = []
            all_labels = []
            start_train_time = time.perf_counter()

            for inputs, labels in Data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                outputs = model_meta(inputs)
                loss = meta_criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
                _, preds = torch.max(outputs, 1)
                running_acc += (preds == labels).sum().item()
                meta_optimizer.zero_grad()
                loss.backward()
                meta_optimizer.step()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            train_time = (time.perf_counter() - start_train_time)
            print(f"Train Time: {train_time:.2f}s")
            print(
                f'Finish {epoch + 1} epoch, Loss: {running_loss / len(Data_loader.dataset):.6f}, Acc: {running_acc / len(Data_loader.dataset):.6f}')

            # 计算指标
            report = classification_report(all_labels, all_preds, output_dict=True, zero_division=1)
            accuracy = report.get('accuracy')
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1_score = report['weighted avg']['f1-score']

            # 打印格式方便复制
            print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')

            json_filepath = "class_indices.json"
            # TPR, FPR = calculate_tpr_fpr(dataset_name + json_filepath, all_labels, all_preds)
            TPR, FPR = utils.calculate_tpr_fpr_multiclass(all_labels, all_preds, num_classes)
            print(f'TPR: {TPR:.4f}，FPR: {FPR:.4f}')

            # 保存模型
            file_name = f'D:/Python Project/Deep-Traffic/models/{dataset_name}/meta_RNN_{dataset_name}_best_' + str(
                ep) + '.pth'
            if best_acc <= accuracy:
                best_acc = accuracy
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                torch.save(model_meta.state_dict(), file_name)
                print(f'Saving model at {file_name}, best epoch {epoch}, acc: {accuracy:.4f}')
        #
        # file_dir = 'D:/Python Project/Deep-Traffic/models/USTC/meta_RNN_USTC_final_' + str(ep) + '.pth'
        # os.makedirs(os.path.dirname(file_dir), exist_ok=True)
        # torch.save(model_meta.state_dict(), file_dir)


def meta_predict(stacking_train, labels, input_size, num_classes, dataset_name):
    all_predictions = []
    stacking_train_tensor = torch.tensor(stacking_train, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    train_dataset = TensorDataset(stacking_train_tensor, labels_tensor)
    Data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    accuracies = []
    tprs = []
    fprs = []
    f1_scores = []
    EP = 10
    for ep in range(EP):
        print(f'L Epoch: {ep}')
        model_path = f'./models/{dataset_name}/meta_RNN_{dataset_name}_best_{ep}.pth'
        model_meta = RNN_meta(input_size=input_size, num_classes=num_classes)
        model_meta.load_state_dict(torch.load(model_path))
        model_meta = model_meta.to(device)
        print('validating RNN')
        all_outputs = []
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            # 使用 tqdm 包裹 dataloader
            for inputs, targets in tqdm(Data_loader, desc=f'validating Epoch {ep + 1}/{EP}', leave=False):
                inputs = inputs.to(device)
                targets = targets.to(device)
                with autocast("cuda"):  # 使用 autocast 上下文管理器
                    outputs = model_meta(inputs)
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
        TPR, FPR = utils.calculate_tpr_fpr_multiclass(all_labels, all_predictions, num_classes)
        print(f"TPR: {TPR}, FPR: {FPR}")
        report = classification_report(all_labels, all_predictions, output_dict=True, zero_division=1)
        precision = report['weighted avg']['precision']
        recall = report['macro avg']['recall']
        f1_score = report['weighted avg']['f1-score']

        print(
            f'Precision: {precision:.6f}, Recall: {recall:.6f}, F1 Score: {f1_score:.6f}, Accuracy: {accuracy:.6f}')

        if ep == 0:
            # 计算混淆矩阵
            cm = confusion_matrix(all_labels, all_predictions)
            # 打印混淆矩阵
            print("Confusion Matrix:")
            print(cm)
            # 保存混淆矩阵到文件
            cm_file_path = f'{dataset_name}_RNN_confusion_matrix.csv'
            np.savetxt(cm_file_path, cm, delimiter=',', fmt='%d')
            print(f"Confusion matrix saved to {cm_file_path}")

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

    csv_file_path = f'{dataset_name}_Meta_RNN.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration'] + [str(i + 1) for i in range(10)])
        writer.writerow(['Accuracy'] + [f"{acc:.4f}" for acc in accuracies])
        writer.writerow(['TPR'] + [f"{tpr:.4f}" for tpr in tprs])
        writer.writerow(['FPR'] + [f"{fpr:.4f}" for fpr in fprs])
        writer.writerow(['F1 Score'] + [f"{f1:.4f}" for f1 in f1_scores])

    print(f"Results saved to {csv_file_path}")
    return all_predictions


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


def load_model(model_path):
    # 加载模型
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()
    return model


def get_accuracy(model, data_loader, num_class, dataset_name):
    """
    For RNN prediction,
    return predictions, recall, precision, f1_score, FPR, TPR, accuracy

    """
    model.eval()
    predictions = []
    true_labels = []
    correct = 0
    total = 0
    target_counts = torch.zeros(num_class).to(next(model.parameters()).device)  # 存储每个类别的实际出现次数
    predicted_counts = torch.zeros(num_class).to(next(model.parameters()).device)  # 存储每个类别的预测出现次数。
    accurate_counts = torch.zeros(num_class).to(next(model.parameters()).device)  # 存储每个类别正确预测的次数。

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

    # 计算指标
    recall = accurate_counts / target_counts
    precision = accurate_counts / predicted_counts
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = accurate_counts.sum() / target_counts.sum()

    # 确保分母不为零
    eps = 1e-7  # 一个很小的正数，用来防止除以零
    # 计算每个类别的真负样本数
    true_negative_counts = total_samples - (predicted_counts + (target_counts - accurate_counts))
    true_negative_counts = torch.max(true_negative_counts, torch.tensor([eps]).to(next(model.parameters()).device))
    # FP 假阳性的次数，即实际为负但被错误识别为正的样本数。
    false_positive_counts = predicted_counts - accurate_counts
    # 计算每个类别的假阳性率（FPR）
    fpr = false_positive_counts / (false_positive_counts + true_negative_counts)

    # 精度调整
    recall = (recall.cpu().numpy() * 1).round(5)
    precision = (precision.cpu().numpy() * 1).round(5)
    f1_score = (f1_score.cpu().numpy() * 1).round(5)
    accuracy = (accuracy.cpu().numpy() * 1).round(5)
    fpr = (fpr.cpu().numpy() * 1).round(5)

    # TPR, FPR = calculate_tpr_fpr(dataset_name + "class_indices.json", true_labels, predictions)
    TPR, FPR = calculate_tpr_fpr_multiclass(true_labels, predictions, num_class)
    # 打印格式方便复制
    print('recall(TPR) ', " ".join(f'{value:.5f}' for value in recall))
    print('precision   ', " ".join(f'{value:.5f}' for value in precision))
    print('F1 score    ', " ".join(f'{value:.5f}' for value in f1_score))
    print('FPR         ', " ".join(f'{value:.5f}' for value in fpr))
    print('accuracy    ', accuracy)
    acc = correct / total_samples
    print('correct={}, Test ACC:{:.5f}'.format(correct, acc))
    report = classification_report(true_labels, predictions, zero_division=1, output_dict=True)
    F1 = report['weighted avg']['f1-score']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    print("weighted TPR", TPR)
    print("weighted FPR", FPR)
    return predictions, recall, precision, F1, FPR, TPR, accuracy


def main():
    # 数据集路径
    PNG_PATH = '4_Png_16_ISAC'
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
    image_path = os.path.join(data_root, "pre-processing", PNG_PATH)
    assert os.path.exists(image_path), f"{image_path} path does not exist."

    batch_size = 64
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(num_workers))

    # 预处理数据
    # test_loader = preprocess_data(image_path, batch_size, num_workers)
    _, test_loader, label, _, _, dataset_name = data_pre_process(data_root, PNG_PATH)

    directory_path = os.path.join(os.path.abspath(os.getcwd()), f"../pre-processing/{PNG_PATH}/Train")
    entries = os.listdir(directory_path)
    # 过滤出文件夹并计数
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    num_classes = folder_count

    recalls = []
    precisions = []
    f1_scores = []
    fprs = []
    tprs = []
    accuracies = []
    num_epochs = 10

    for i in range(num_epochs):
        print("epoch: ", i)
        model_path = f'../models/{dataset_name[:-1]}/RNN_final_{dataset_name}' + str(i) + '.pt'  # 已训练好的模型权重文件路径
        model = load_model(model_path)

        predictions, recall, precision, f1_score, fpr, tpr, accuracy = get_accuracy(model, test_loader, num_classes,
                                                                                    dataset_name)

        # recalls.append(recall)
        # precisions.append(precision)
        f1_scores.append(f1_score)
        fprs.append(fpr)
        tprs.append(tpr)
        accuracies.append(accuracy)

    # 转换为 NumPy 数组
    # recalls = np.array(recalls)
    # precisions = np.array(precisions)
    f1_scores = np.array(f1_scores)
    fprs = np.array(fprs)
    accuracies = np.array(accuracies)
    tprs = np.array(tprs)

    # # 汇总所有类别的指标值
    # all_recalls = recalls.flatten()
    # all_precisions = precisions.flatten()
    all_f1_scores = f1_scores.flatten()
    all_fprs = fprs.flatten()
    all_tprs = tprs.flatten()

    # 计算均值和标准差
    # mean_recall = np.mean(all_recalls)
    # std_recall = np.std(all_recalls)
    #
    # mean_precision = np.mean(all_precisions)
    # std_precision = np.std(all_precisions)

    mean_f1_score = np.mean(all_f1_scores)
    std_f1_score = np.std(all_f1_scores)

    mean_tpr = np.mean(all_tprs)
    std_tpr = np.std(all_tprs)

    mean_fpr = np.mean(all_fprs)
    std_fpr = np.std(all_fprs)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    # 打印结果
    # print('Mean recall(weighted)', f'{mean_recall:.5f}±{std_recall:.5f}')
    # print('Mean precision   ', f'{mean_precision:.5f}±{std_precision:.5f}')
    print("accuracies", accuracies)
    print("tprs", all_tprs)
    print("fprs", all_fprs)
    print("f1 scores", all_f1_scores)
    print('Mean accuracy    ', f'{mean_accuracy:.4f}±{std_accuracy:.4f}')
    print('Mean FPR         ', f'{mean_fpr:.4f}±{std_fpr:.4f}')
    print('Mean TPR         ', f'{mean_tpr:.4f}±{std_tpr:.4f}')
    print('Mean F1 score    ', f'{mean_f1_score:.4f}±{std_f1_score:.4f}')

    csv_file_path = f'{dataset_name}RNN.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration'] + [str(i + 1) for i in range(10)])
        writer.writerow(['Accuracy'] + [f"{acc:.4f}" for acc in accuracies])
        writer.writerow(['TPR'] + [f"{tpr:.4f}" for tpr in tprs])
        writer.writerow(['FPR'] + [f"{fpr:.4f}" for fpr in fprs])
        writer.writerow(['F1 Score'] + [f"{f1:.4f}" for f1 in f1_scores])

    print(f"Results saved to {csv_file_path}")


if __name__ == "__main__":
    main()
