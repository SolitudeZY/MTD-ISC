# -*- coding: utf-8 -*-
import math
import os
import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader
from tqdm import tqdm

from FocalLoss import FocalLoss
from utils import data_pre_process, calculate_tpr_fpr, calculate_tpr_fpr_multiclass
import torch.nn.functional as F


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


# 定义网络模型
class LSTM(nn.Module):
    def __init__(self, Class_Num: int, Input_Size: int):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=Input_Size,  # if use nn.LSTM_model(), it hardly learns
                           hidden_size=64,  # BiLSTM 隐藏单元
                           num_layers=1,  # BiLSTM 层数
                           batch_first=True,
                           # input & output will have batch size as 1s dimension. e.g. (batch, seq, Input_size)
                           )
        self.dropout = nn.Dropout(0.01)  # 添加Dropout层
        self.out = nn.Linear(64, out_features=Class_Num)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state
        # choose r_out at the last time step
        last_output = r_out[:, -1, :]
        dropout_output = self.dropout(last_output)  # 应用Dropout层
        out = self.out(dropout_output)
        return out


def prepare_data():
    # 数据增强
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(16),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize(16),
            transforms.CenterCrop(16),
            transforms.ToTensor(),
            # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    batch_size = 64
    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
    data_root = os.path.abspath(os.getcwd())  # get data root path
    image_path = os.path.join(data_root, "pre-processing", "4_Png_16_CTU")  # data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform["test"])

    test_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=nw)
    val_num, train_num = len(validate_dataset), len(train_dataset)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    return train_loader, test_loader


if __name__ == '__main__':
    # 数据输入
    PNG_PATH = '4_Png_16_ISAC'
    # PNG_PATH = '4_Png_16_USTC'
    # PNG_PATH = '4_Png_16_CTU'
    directory_path = os.path.join(os.path.abspath(os.getcwd()), F"./pre-processing/{PNG_PATH}/Train")
    entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    class_nums = folder_count

    BATCH_SIZE = 64  # 批训练的数量
    INPUT_SIZE = 16  # 特征向量长度
    EP = 10
    num_epoches = 5
    for ep in range(1):
        print("\n EPOCH:", ep)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rnn = LSTM(class_nums, INPUT_SIZE)  # 实例化
        rnn = rnn.cuda()
        best_acc = 0.0
        optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)  # 选择优化器
        # criterion = nn.CrossEntropyLoss()  # 定义损失函数，the target label is not one-hotted
        criterion = FocalLoss(gamma=2, alpha=0.75)
        # 数据预处理
        train_loader, test_loader, label, _, _, dataset_name = data_pre_process(os.getcwd(), PNG_PATH)
        # thresholds = [0] * class_nums  # class_nums 是你的类别数量

        for epoch in range(num_epoches):  # train
            print('epoch {}'.format(epoch + 1))
            print('*' * 10)
            running_loss = 0.0
            running_acc = 0.0
            count = 0
            true_labels, pred_labels = [], []
            startTrain = time.perf_counter()
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

                count += 1
            trainTime = (time.perf_counter() - startTrain)
            print("trainTime:", trainTime)
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

            startTest = time.perf_counter()
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
                    input_size = INPUT_SIZE  # 对于MNIST数据集，每个像素是一个特征; INPUT_SIZE=16
                    imgs = imgs.view(batch_size, seq_len, input_size)  # 展平为一维序列

                    out = rnn(imgs)
                    loss = criterion(out, labels)
                    eval_loss += loss.item() * labels.size(0)

                    # 在测试阶段应用 softmax 函数
                    probs = F.softmax(out, dim=1)

                    # 获取每个样本最大概率对应的类别索引
                    max_probs, preds = torch.max(probs, dim=1)

                    # # 应用阈值
                    # for i in range(len(preds)):
                    #     if max_probs[i] < thresholds[preds[i]]:
                    #         preds[i] = -1  # 将不确定的预测标记为-1或其他未定义类别

                    num_correct = (preds == labels).sum()
                    eval_acc += num_correct.item()

                    # total += labels.size(0)
                    # correct += pred.eq(labels.data).cpu().sum()
                    # pre_mask = torch.zeros(out.size()).scatter_(1, preds.cpu().view(-1, 1), 1.)
                    # predict_num += pre_mask.sum(0)
                    # tar_mask = torch.zeros(out.size()).scatter_(1, labels.data.cpu().view(-1, 1), 1.)
                    # target_num += tar_mask.sum(0)
                    # acc_mask = pre_mask * tar_mask
                    # acc_num += acc_mask.sum(0)

                    # 存储真实标签和预测标签
                    true_labels.extend(labels.cpu().numpy())  # 使用 extend 而不是 append
                    predictions.extend(preds.cpu().numpy())  # 使用 extend 而不是 append

            testTime = (time.perf_counter() - startTest)
            print("testTime:", testTime)

            TPR, FPR = calculate_tpr_fpr_multiclass(true_labels, predictions, class_nums)
            print(f"TPR: {TPR}， FPR: {FPR}")

            # recall = acc_num / target_num
            # precision = acc_num / predict_num
            # F1 = 2 * recall * precision / (recall + precision)
            # accuracy = acc_num.sum(1) / target_num.sum(1)
            #
            # # 精度调整
            # recall = (recall.numpy()[0] * 100).round(3)
            # precision = (precision.numpy()[0] * 100).round(3)
            # F1 = (F1.numpy()[0] * 100).round(3)
            # accuracy = (accuracy.numpy()[0] * 100).round(3)
            # print('recall   ', " ".join('%s' % id for id in recall))
            # print('precision   ', " ".join('%s' % id for id in precision))
            # print('F1   ', " ".join('%s' % id for id in F1))

            report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
            accuracy = accuracy_score(true_labels, predictions)
            recall = report['weighted avg']['recall']
            precision = report['weighted avg']['precision']
            F1 = report['weighted avg']['f1-score']
            # 打印格式方便复制
            print(f'Precision: {precision:.6f}, Recall: {recall:.6f}, F1 Score: {F1:.6f}, Accuracy: {accuracy:.6f}')
            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
                test_loader.dataset)), eval_acc / (len(test_loader.dataset))))

            save_dir = f'./models/{dataset_name[:-1]}'
            # if best_acc < accuracy:
            #     print('Saving best model at epoch', epoch)
            #     best_acc = accuracy
            #     save_file = 'LSTM_model_best_' + dataset_name + str(ep) + '.pt'
            #     save_filename = os.path.join(save_dir, save_file)
            #     torch.save(rnn, save_filename)
            #     print('Saved as %s' % save_filename)
