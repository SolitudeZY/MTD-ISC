import math

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import optimizer
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from FocalLoss import FocalLoss
from utils import data_pre_process, calculate_tpr_fpr, calculate_tpr_fpr_multiclass
import lstm
import csv  # 导入csv模块

# 设置超参数
num_epochs = 3
batch_size = 128
learning_rate = 0.01


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


# CNN模型，用于特征提取
class CNN_for_LSTM(nn.Module):
    def __init__(self):
        super(CNN_for_LSTM, self).__init__()
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
    def __init__(self, class_nums):
        super(CNNtoLSTM, self).__init__()
        self.cnn = CNN_for_LSTM()
        self.lstm = LSTM_model(input_size=16 * 16, hidden_size=128, num_layers=1, num_classes=class_nums)

    def forward(self, x):
        # 通过CNN
        x = self.cnn(x)

        # 将输出展平为 (batch_size, seq_length, Input_size)
        x = x.view(x.size(0), -1, 16 * 16)  # 假定CNN输出为(batch_size, channels, height, width)

        # 通过LSTM
        x = self.lstm(x)
        return x


class LSTM_meta(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_meta, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义LSTM层
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.05)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.05)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        # 定义一个全连接层进行输出
        self.fc = nn.Linear(hidden_size, num_classes)
        # 定义ELU激活函数
        self.elu = nn.ELU()

    def forward(self, x):
        # 第一层LSTM
        out1, _ = self.lstm1(x)
        # 第二层LSTM
        out2, _ = self.lstm2(out1)
        # 残差连接
        # 需要调整维度以确保可以进行相加
        if out1.size(1) != out2.size(1):
            out1 = out1[:, :out2.size(1), :]  # 只取出前面部分
        residual = out1 + out2
        # 使用ELU激活函数
        activated_residual = self.elu(residual)
        # 第三层LSTM
        out3, _ = self.lstm3(activated_residual)
        final_out = out3[:, -1, :]  # 获取最后一个时间步的输出
        out = self.fc(final_out)  # 输出层
        return out


class LSTM(nn.Module):
    def __init__(self, Class_Num: int, Input_Size: int):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=Input_Size,  # if use nn.LSTM_model(), it hardly learns
                           hidden_size=64,  # BiLSTM 隐藏单元
                           num_layers=1,  # BiLSTM 层数
                           batch_first=True,
                           # input & output will have batch size as 1s dimension. e.g. (batch, seq, Input_size)
                           )
        self.out = nn.Linear(64, out_features=Class_Num)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


def save_model(model, output_dir='models', model_filename='cnn_lstm_model.pt'):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, model_filename)
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')


def load_model(model_path, device):
    # model = CNNtoLSTM().to(device)
    model = torch.load(model_path, map_location=device)
    return model


def train_model(num_epoches, train_loader, class_nums):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_SIZE = 16  # 特征向量长度
    LR = 0.002  # learning rate
    # directory_path = os.path.join(os.path.abspath(os.getcwd()), "../pre-processing/4_Png_16_CTU/Train")
    # entries = os.listdir(directory_path)
    # folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    # class_nums = folder_count

    rnn = LSTM_meta(input_size=16, hidden_size=64, num_layers=2, num_classes=class_nums).to(device)
    rnn = rnn.cuda()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # 选择优化器，optimize all cnn parameters
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epoches):  # train
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        running_loss = 0.0
        running_acc = 0.0
        count = 0
        true_labels, pred_labels = [], []
        # startTrain = time.perf_counter()
        rnn.train()
        # 创建 tqdm 对象并存储在 pbar 变量中
        pbar = tqdm(train_loader, desc="training", leave=True)
        for imgs, labels in pbar:
            # 假设原始的 imgs 形状是 (batch, channels, height, width)，例如 (batch, 1, 16, 16)
            imgs = imgs.squeeze(1)  # 如果通道数为1，可以去掉这一维度
            imgs = imgs.float().cuda()  # 直接转换为浮点型并移到GPU
            labels = labels.cuda()

            # 假设 imgs 的形状现在是 (batch, height, width)，例如 (batch, 16, 16)
            batch_size = imgs.shape[0]
            seq_len = imgs.shape[1] * imgs.shape[2]  # 序列长度等于高度乘以宽度
            input_size = 16  # 对于MNIST数据集，每个像素是一个特征

            # 将图像数据展平为 (batch, seq_len, Input_size) 形状
            imgs = imgs.view(batch_size, seq_len, input_size)  # 展平为一维序列

            # 前向传播
            out = rnn(imgs)
            loss = criterion(out, labels)
            running_loss += loss.item() * labels.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == labels).sum()
            running_acc += num_correct.item()

            # 向后传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新进度条上的损失值
            pbar.set_postfix(loss=f"{loss.item():.6f}")

    return rnn


def predict(model, test_loader: DataLoader, class_num: int, device=torch.device("cuda")):
    """
    使用给定的LSTM模型对测试数据进行预测，并收集预测标签和真实标签。

    参数:
    model: 已经训练好的LSTM模型实例。
    test_loader: 包含测试数据的PyTorch DataLoader对象。
    class_num: 类别总数。
    threshold: 分类阈值，默认为0.5。

    返回:
    predict_labels: 包含所有测试样本预测标签的列表。
    true_labels: 包含所有测试样本真实标签的列表。
    """
    model.eval()  # 设置模型为评估模式
    true_labels = []  # 初始化用于存放真实标签的列表
    predict_labels = []  # 初始化用于存放预测标签的列表
    print('LSTM predicting, test_size is {}\n'.format(len(test_loader.dataset)))

    # thresholds = [0] * class_num  # class_nums 是你的类别数量

    with torch.no_grad():  # 关闭梯度计算
        for imgs, labels in test_loader:
            imgs = imgs.squeeze(1).float().cuda()  # 处理输入数据，确保其与模型要求的输入格式一致
            labels = labels.cuda()
            batch_size = imgs.shape[0]
            seq_len = imgs.shape[1] * imgs.shape[2]
            input_size = 16  # 根据实际情况调整
            imgs = imgs.view(batch_size, seq_len, input_size)  # 将图像数据展平为 (batch, seq_len, Input_size) 形状

            out = model(imgs)  # 前向传播获取输出
            # 在测试阶段应用 softmax 函数
            probs = torch.nn.functional.softmax(out, dim=1)

            # 获取每个样本最大概率对应的类别索引
            max_probs, preds = torch.max(probs, dim=1)

            # # 应用阈值
            # for i in range(len(preds)):
            #     if max_probs[i] < thresholds[preds[i]]:
            #         preds[i] = -1  # 将不确定的预测标记为-1或其他未定义类别

            # 存储真实标签和预测标签
            true_labels.extend(labels.cpu().numpy())  # 使用 extend 而不是 append
            predict_labels.extend(preds.cpu().numpy())  # 使用 extend 而不是 append

    report = classification_report(true_labels, predict_labels, output_dict=True, zero_division=0)
    accuracy = report.get('accuracy')
    # TPR, FPR = calculate_tpr_fpr("class_indices.json", true_labels=true_labels, pred_labels=predict_labels)
    TPR, FPR = calculate_tpr_fpr_multiclass(true_labels, predict_labels, class_num)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    print(f"accuracy: {accuracy}")
    print(f'Precision: {precision:.6f}, Recall: {recall:.6f}, F1 Score: {f1_score:.6f}')
    print(f"TPR: {TPR}, FPR: {FPR}")

    print(f"predictions shape is: {len(predict_labels)}\n")

    return predict_labels, true_labels


def calculate_metrics(true_labels, predict_labels, folder_count):
    report = classification_report(true_labels, predict_labels, output_dict=True, zero_division=0)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    accuracy = report['accuracy']
    # TPR, FPR = calculate_tpr_fpr(json_filepath=dataset_name + 'class_indices.json', true_labels=true_labels,
    #                              pred_labels=predict_labels)
    TPR, FPR = calculate_tpr_fpr_multiclass(y_true=true_labels, y_pred=predict_labels, n_classes=folder_count)
    return precision, recall, f1_score, accuracy, TPR, FPR


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir = os.path.join(os.getcwd(), '')
    png_path = "4_Png_16_ISAC"
    try:
        directory_path = os.path.join(os.path.abspath(os.getcwd()), "./pre-processing/" + png_path + "/Train")
        entries = os.listdir(directory_path)
    except FileNotFoundError:
        directory_path = os.path.join(os.path.abspath(os.getcwd()), "../pre-processing/" + png_path + "/Train")
        entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)

    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_accuracies = []
    all_TPRs = []
    all_FPRs = []
    train_loader, test_loader, _, _, _, dataset_name = data_pre_process(dir, png_path, None)

    for i in range(10):
        print(f"\n Iteration {i + 1}:")

        model_path = F'../models/{dataset_name[:-1]}/LSTM_model_best_{dataset_name}' + str(i) + '.pt'
        print("loading model from", model_path)
        # 'LSTM_model_best_' + dataset_name + str(ep) + '.pt'
        model = load_model(model_path, device=device)

        predict_labels, true_labels = predict(model, test_loader, folder_count)

        precision, recall, f1_score, accuracy, TPR, FPR = calculate_metrics(true_labels, predict_labels, folder_count)

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1_score)
        all_accuracies.append(accuracy)
        all_TPRs.append(TPR)
        all_FPRs.append(FPR)

        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
        print(f"TPR: {TPR:.4f}, FPR: {FPR:.4f}, Accuracy: {accuracy:.4f}")
    # 计算平均值和标准差
    # mean_precision = np.mean(all_precisions)
    # std_precision = np.std(all_precisions)
    # mean_recall = np.mean(all_recalls)
    # std_recall = np.std(all_recalls)
    mean_f1_score = np.mean(all_f1_scores)
    std_f1_score = np.std(all_f1_scores)
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    mean_TPR = np.mean(all_TPRs)
    std_TPR = np.std(all_TPRs)
    mean_FPR = np.mean(all_FPRs)
    std_FPR = np.std(all_FPRs)
    print("accuracy : ", all_accuracies)
    print("tprs", all_TPRs)
    print("fprs", all_FPRs)
    print("f1_scores", all_f1_scores)
    print("\nOverall Statistics:")
    # print(f"Mean Precision: {mean_precision:.6f} ± {std_precision:.6f}")
    # print(f"Mean Recall: {mean_recall:.6f} ± {std_recall:.6f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}±{std_accuracy:.4f}")
    print(f"Mean F1 Score: {mean_f1_score:.4f}±{std_f1_score:.4f}")
    print(f"Mean TPR: {mean_TPR:.4f}±{std_TPR:.4f}")
    print(f"Mean FPR: {mean_FPR:.4f}±{std_FPR:.4f}")

    csv_file_path = f'{dataset_name}LSTM.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration'] + [str(i + 1) for i in range(10)])
        writer.writerow(['Accuracy'] + [f"{acc:.4f}" for acc in all_accuracies])
        writer.writerow(['TPR'] + [f"{tpr:.4f}" for tpr in all_TPRs])
        writer.writerow(['FPR'] + [f"{fpr:.4f}" for fpr in all_FPRs])
        writer.writerow(['F1 Score'] + [f"{f1:.4f}" for f1 in all_f1_scores])

    print(f"Results saved to {csv_file_path}")


def main_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir = os.path.join(os.getcwd(), '../')
    print("training lstm_for_meta")
    PNG_path = "4_Png_16_CTU"
    train_loader, test_loader, _, _, _, dataset_name = data_pre_process(dir, PNG_path, None)

    directory_path = os.path.join(os.path.abspath(os.getcwd()), F"../pre-processing/{PNG_path}/Train")
    entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    class_nums = folder_count

    rnn = LSTM_meta(input_size=16, hidden_size=64, num_layers=2, num_classes=folder_count).to(device)

    save_path = F'../models/{dataset_name[:-1]}'
    filename = 'LSTM_for_meta.pt'
    save_dir = os.path.join(save_path, filename)

    criterion = FocalLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.002)

    rnn.train()
    # 训练模型
    EP = 20
    for epoch in range(EP):
        print(f"Epoch {epoch + 1}/{EP}")
        # 创建 tqdm 对象并存储在 pbar 变量中
        running_loss = 0.0
        running_acc = 0.0
        count = 0
        true_labels, pred_labels = [], []
        pbar = tqdm(train_loader, desc="training", leave=True)

        rnn.train()
        for imgs, labels in pbar:
            # 假设原始的 imgs 形状是 (batch, channels, height, width)，例如 (batch, 1, 16, 16)
            imgs = imgs.squeeze(1)  # 如果通道数为1，可以去掉这一维度
            imgs = imgs.float().cuda()  # 直接转换为浮点型并移到GPU
            labels = labels.cuda()

            # 假设 imgs 的形状现在是 (batch, height, width)，例如 (batch, 16, 16)
            batch_size = imgs.shape[0]
            seq_len = imgs.shape[1] * imgs.shape[2]  # 序列长度等于高度乘以宽度
            input_size = 16  # 对于MNIST数据集，每个像素是一个特征

            # 将图像数据展平为 (batch, seq_len, Input_size) 形状
            imgs = imgs.view(batch_size, seq_len, input_size)  # 展平为一维序列

            # 前向传播
            out = rnn(imgs)
            loss = criterion(out, labels)
            running_loss += loss.item() * labels.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == labels).sum()
            running_acc += num_correct.item()

            # 向后传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新进度条上的损失值
            pbar.set_postfix(loss=f"{loss.item():.6f}")

            count += 1
        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f} \n'.format(
            epoch + 1, running_loss / (len(train_loader.dataset)), running_acc / (len(train_loader.dataset))))

        # 开始测试
        rnn.eval()
        eval_loss = 0.0
        eval_acc = 0.0
        correct = 0
        total = np.array([labels.size(0) for _, labels in test_loader]).sum()

        target_num = torch.zeros((1, class_nums))
        predict_num = torch.zeros((1, class_nums))
        acc_num = torch.zeros((1, class_nums))

        # 创建一个列表来存储预测标签
        true_labels = []
        predictions = []
        # 测试模型
        with torch.no_grad():  # 不需要计算梯度，节省内存
            print("Testing...")
            for imgs, labels in test_loader:
                imgs = imgs.squeeze(1)  # (N,28,28)
                imgs = imgs.float().cuda()
                labels = labels.cuda()
                # 假设 imgs 的形状现在是 (batch, height, width)，例如 (batch, 28, 28)
                batch_size = imgs.shape[0]
                seq_len = imgs.shape[1] * imgs.shape[2]  # 序列长度等于高度乘以宽度
                input_size = 16  # 对于MNIST数据集，每个像素是一个特征; INPUT_SIZE=16
                imgs = imgs.view(batch_size, seq_len,
                                 input_size)  # 展平为一维序列,将图像数据展平为 (batch, seq_len, Input_size) 形状

                out = rnn(imgs)
                loss = criterion(out, labels)
                eval_loss += loss.item() * labels.size(0)
                _, pred = torch.max(out, 1)
                num_correct = (pred == labels).sum()
                eval_acc += num_correct.item()

                total += labels.size(0)
                correct += pred.eq(labels.data).cpu().sum()
                pre_mask = torch.zeros(out.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
                predict_num += pre_mask.sum(0)
                tar_mask = torch.zeros(out.size()).scatter_(1, labels.data.cpu().view(-1, 1), 1.)
                target_num += tar_mask.sum(0)
                acc_mask = pre_mask * tar_mask
                acc_num += acc_mask.sum(0)

                # 收集预测标签和真实标签
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        TPR, FPR = calculate_tpr_fpr(json_filepath=dataset_name + 'class_indices.json', true_labels=true_labels,
                                     pred_labels=predictions)
        print(f"TPR: {TPR}， FPR: {FPR}")

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
        print('recall   ', " ".join('%s' % id for id in recall))
        print('precision   ', " ".join('%s' % id for id in precision))
        print('F1   ', " ".join('%s' % id for id in F1))
        print('accuracy ', accuracy)
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            test_loader.dataset)), eval_acc / (len(test_loader.dataset))))

        torch.save(rnn, save_dir)
        print("Model saved at ", save_dir)


if __name__ == '__main__':
    main()
