import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F  # 导入 F 模块以使用 softmax 函数
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from utils import data_pre_process, calculate_tpr_fpr, calculate_tpr_fpr_multiclass
from FocalLoss import FocalLoss
import torch.amp as amp


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
            nn.Dropout(0.01),  # 添加 Dropout 层
            nn.Linear(1024, 128),
            nn.ELU(inplace=True),  # 替换为 ELU
            nn.Dropout(0.01),  # 添加 Dropout 层
            nn.Linear(128, out_features=class_nums)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.out(x)
        return x


# class CNN(nn.Module):
#     def __init__(self, class_nums):
#         super(CNN, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # 16x16 -> 16x16
#             nn.BatchNorm2d(32),
#             nn.ELU(inplace=True)
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 16x16 -> 16x16
#             nn.BatchNorm2d(64),
#             nn.ELU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
#         )
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 8x8 -> 8x8
#             nn.BatchNorm2d(128),
#             nn.ELU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 -> 4x4
#         )
#
#         # 经过两次MaxPooling后，特征图大小为4x4
#         self.out = nn.Sequential(
#             nn.Linear(128 * 4 * 4, 512),  # 输入特征图大小为4x4，通道数为128
#             nn.ELU(inplace=True),
#             nn.Dropout(0.1),  # 在第一个全连接层后添加Dropout层
#             nn.Linear(512, 128),
#             nn.ELU(inplace=True),
#             nn.Dropout(0.1),  # 在第二个全连接层后添加Dropout层
#             nn.Linear(128, out_features=class_nums)
#         )
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = x.view(x.size(0), -1)  # 展平
#         x = self.out(x)
#         return x


if __name__ == '__main__':
    # 定义超参数
    batch_size = 64
    PNG_PATH = '4_Png_16_USTC'
    directory_path = os.path.join(os.path.abspath(os.getcwd()), F"./pre-processing/{PNG_PATH}/Train")
    entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    class_num = folder_count
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, _, _, _, dataset_name = data_pre_process(data_directory=os.getcwd(),
                                                                        png_path=PNG_PATH)
    EP = 10
    epochs = 4
    scaler = amp.GradScaler('cuda')
    for ep in range(1):
        best_accuracy = 0.0  # 初始化最佳准确率为0

        model = CNN(class_num).to(device)
        # loss_function = nn.CrossEntropyLoss()
        loss_function = FocalLoss(gamma=2, alpha=0.75)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        print(f"\n第{ep + 1}轮训练开始")

        for epoch in range(epochs):
            cnt = 0
            print(f"epoch: {epoch + 1}/{epochs}")
            model.train()
            for img, label in tqdm(train_loader, desc='training'):
                img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)

                # 使用 autocast 上下文管理器进行混合精度训练
                with torch.amp.autocast('cuda'):
                    out = model(img)
                    loss = loss_function(out, label)

                # 反向传播和优化步骤
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # 测试网络
            print("validating")
            model.eval()
            eval_loss = 0
            eval_acc = 0
            all_outputs = []  # 用于保存预测输出，使用softmax函数来计算对每个类的概率
            all_labels = []  # 保存真实标签
            all_predictions = []  # 新增一个列表来保存预测标签

            thresholds = [0] * class_num  # 假设每个类别的阈值相同，可以根据实际情况调整

            with torch.no_grad():
                for data in test_loader:
                    img, label = data
                    img, label = img.to(device), label.to(device)
                    with torch.amp.autocast('cuda'):
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
            dataset_len = len(test_loader.dataset)
            # tpr, fpr = calculate_tpr_fpr(dataset_name + 'class_indices.json', all_labels, all_predictions)
            tpr, fpr = calculate_tpr_fpr_multiclass(all_labels, all_predictions, class_num)
            # 计算相关指标
            report = classification_report(all_labels, all_predictions, output_dict=True, zero_division=0)
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1_score = report['weighted avg']['f1-score']
            accuracy = eval_acc / dataset_len

            # 输出指标
            print(f"F1 Score: {f1_score:.6f}")
            print(f"Precision: {precision:.6f}")
            print(f"Recall: {recall:.6f}")
            print(f"Accuracy: {accuracy:.6f}")
            print(f"tpr: {tpr}, fpr: {fpr}")

            print(f'Test Loss: {eval_loss / dataset_len:.6f}, Acc: {eval_acc / dataset_len:.6f}')

            # 保存最佳模型  若要保存模型文件将下面这一段解开注释
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # 指定保存模型的目录
                save_dir = f'./models/{dataset_name[:-1]}'
                # 创建目录（如果不存在）
                os.makedirs(save_dir, exist_ok=True)
                save_name = f'CNN_model_best_{dataset_name[:-1]}_{ep}.pth'
                save_filename = os.path.join(save_dir, save_name)
                # 保存模型
                torch.save(model.state_dict(), save_filename)
                print(f'Saved best model at epoch {epoch + 1} with accuracy {best_accuracy:.6f}')

        save_dir = f'./models/{dataset_name[:-1]}'
        save_filename = os.path.join(save_dir, f'CNN_{dataset_name}_final_{ep}.pth')
        torch.save(model.state_dict(), save_filename)
        print(f'Saved final model  ')

    print('Training complete.')

    # # 计算混淆矩阵
    # cm = confusion_matrix(all_labels, all_predictions)
    # # print(f"Confusion matrix:\n{cm}")
    # # 读取类别到索引的映射，找到正常流量的类别标签
    # with open(dataset_name + 'class_indices.json', 'r') as json_file:
    #     cla_dict = json.load(json_file)
    # # 找到值为 "Normal" 的索引
    # normal_traffic_idx = None
    # for idx, class_name in cla_dict.items():
    #     if class_name == "Normal":
    #         normal_traffic_idx = int(idx)
    #         print(f"Found 'Normal' class at index: {idx}")
    #         break
    # if normal_traffic_idx is None:
    #     raise ValueError("Class 'Normal' not found in the JSON file.")
    #
    # # 恶意流量的标签
    # malicious_traffic_indices = list(range(class_num))
    # malicious_traffic_indices.remove(normal_traffic_idx)
    #
    # # 统计正常流量的数量及其正确分类的数量
    # tp_normal = cm[normal_traffic_idx, normal_traffic_idx]
    # fn_normal = np.sum(cm[normal_traffic_idx, :]) - tp_normal
    # fp_normal = np.sum(cm[:, normal_traffic_idx]) - tp_normal
    # tn_normal = np.sum(cm) - tp_normal - fp_normal - fn_normal
    # # # 输出正常流量的指标 print(f"Normal Traffic: TP: {tp_normal}, FN: {fn_normal}, FP: {fp_normal},
    # # TN: {tn_normal}") print(f"Normal Traffic: TPR:{tp_normal / (tp_normal + fn_normal)}, FPR:{fp_normal / (
    # # fp_normal + tn_normal)}")
    #
    # # 初始化总 TP, FP, FN, TN
    # total_tp = tp_normal
    # total_fp = fp_normal
    # total_fn = fn_normal
    # total_tn = tn_normal
    #
    # # 分别计算每个恶意流量标签的 TPR 和 FPR
    # for idx in malicious_traffic_indices:
    #     # 提取 TP, FP, FN, TN
    #     tp_malicious = cm[idx, idx]
    #     fp_malicious = np.sum(cm[:, idx]) - tp_malicious
    #     fn_malicious = np.sum(cm[idx, :]) - tp_malicious
    #     tn_malicious = np.sum(cm) - tp_malicious - fp_malicious - fn_malicious
    #
    #     # 计算 TPR 和 FPR
    #     tpr_malicious = tp_malicious / (tp_malicious + fn_malicious) if tp_malicious + fn_malicious > 0 else 0
    #     fpr_malicious = fp_malicious / (fp_malicious + tn_malicious) if fp_malicious + tn_malicious > 0 else 0
    #
    #     # 累加总 TP, FP, FN, TN
    #     total_tp += tp_malicious
    #     total_fp += fp_malicious
    #     total_fn += fn_malicious
    #     total_tn += tn_malicious
    #
    # # 计算总的 TPR 和 FPR
    # total_tpr = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    # total_fpr = total_fp / (total_fp + total_tn) if total_fp + total_tn > 0 else 0
    # # test_time = (time.perf_counter() - start_train)
    # print(f'total_tpr: {total_tpr:.6f}, total_fpr: {total_fpr:.6f}')
    # # print(f"testTime: {test_time}")
