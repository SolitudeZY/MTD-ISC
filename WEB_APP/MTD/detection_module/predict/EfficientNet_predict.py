import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix

import utils
from FocalLoss import FocalLoss
from utils import data_pre_process, get_class_nums, calculate_tpr_fpr

from EfficientNet import EfficientNet


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        se = F.adaptive_avg_pool2d(x, 1)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se


# MBConv模块
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, kernel_size):
        super(MBConv, self).__init__()
        self.stride = stride
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        expanded_channels = in_channels * expansion_factor
        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(expanded_channels)

        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size, stride=stride,
                                        padding=kernel_size // 2, groups=expanded_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_channels)

        self.se = SEBlock(expanded_channels)

        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        out = F.relu6(self.bn0(self.expand_conv(x)))
        out = F.relu6(self.bn1(self.depthwise_conv(out)))
        out = self.se(out)
        out = self.bn2(self.project_conv(out))

        if self.use_residual:
            out += identity

        return out


# EfficientNet基础结构
class meta_EfficientNet(nn.Module):
    def __init__(self, num_classes, channels=3, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.05):
        super(meta_EfficientNet, self).__init__()

        # 调整 Stem 层的尺寸
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=int(32 * width_coefficient), kernel_size=3, stride=1,
                      padding=1, bias=False),  # 将 stride 改为 1
            nn.BatchNorm2d(int(32 * width_coefficient)),
            nn.ReLU6(inplace=True)
        )

        # 调整 Blocks 中的步长和内核大小
        self.blocks = nn.Sequential(
            MBConv(in_channels=int(32 * width_coefficient), out_channels=int(16 * width_coefficient),
                   expansion_factor=1, stride=1, kernel_size=3),
            MBConv(in_channels=int(16 * width_coefficient), out_channels=int(24 * width_coefficient),
                   expansion_factor=6, stride=1, kernel_size=3),  # 将 stride 改为 1
            MBConv(in_channels=int(24 * width_coefficient), out_channels=int(40 * width_coefficient),
                   expansion_factor=6, stride=1, kernel_size=5),  # 将 stride 改为 1
            MBConv(in_channels=int(40 * width_coefficient), out_channels=int(80 * width_coefficient),
                   expansion_factor=6, stride=1, kernel_size=3),  # 将 stride 改为 1
            MBConv(in_channels=int(80 * width_coefficient), out_channels=int(112 * width_coefficient),
                   expansion_factor=6, stride=1, kernel_size=5),  # 将 stride 改为 1
            MBConv(in_channels=int(112 * width_coefficient), out_channels=int(192 * width_coefficient),
                   expansion_factor=6, stride=1, kernel_size=5),  # 将 stride 改为 1
            MBConv(in_channels=int(192 * width_coefficient), out_channels=int(320 * width_coefficient),
                   expansion_factor=6, stride=1, kernel_size=3),  # 将 stride 改为 1
        )

        self.top = nn.Sequential(
            nn.Conv2d(int(320 * width_coefficient), int(1280 * width_coefficient), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(1280 * width_coefficient)),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(1280 * width_coefficient), num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.top(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# hyperparameters
hyper_epoch = 0


def validate2(model, validate_loader, device, dataset_name, num_classes):
    model.eval()
    acc = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        val_bar = tqdm(validate_loader, desc="Validation")
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            all_predictions.extend(predict_y.cpu().numpy())
            all_labels.extend(val_labels.cpu().numpy())

    val_accurate = acc / len(validate_loader.dataset)

    # 计算精确率、召回率和F1分数
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    TPR, FPR = utils.calculate_tpr_fpr_multiclass(all_labels, all_predictions, num_classes)

    if hyper_epoch == 0:
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        # 打印混淆矩阵
        print("Confusion Matrix:")
        print(cm)
        # 保存混淆矩阵到文件
        cm_file_path = f'{dataset_name}_EfficientNet_confusion_matrix.csv'
        np.savetxt(cm_file_path, cm, delimiter=',', fmt='%d')
        print(f"Confusion matrix saved to {cm_file_path}")

    # 打印结果
    print('Validation accuracy: %.5f' % val_accurate)
    print('Precision: %.5f' % precision)
    print('Recall: %.5f' % recall)
    print('F1 Score: %.5f' % f1)
    print('TPR: %.6f' % TPR)
    print('FPR: %.6f' % FPR)
    # 打印分类报告
    # report = classification_report(all_labels, all_predictions, zero_division=1, output_dict=True)
    # print(report)

    # 返回结果
    # return {
    #     'accuracy': val_accurate,
    #     'precision': precision,
    #     'recall': recall,
    #     'f1': f1,
    #     'tpr': TPR,
    #     'fpr': FPR,
    # }
    return val_accurate, precision, recall, TPR, FPR, f1, all_predictions


def fit(stacking_train, labels, num_classes, dataset_name, epochs=10, device="cuda", ):
    # batch_size, channels, height, weight = 64，3，1，1
    # num_classes = 1000
    # 转换为 PyTorch 张量
    stacking_train = torch.tensor(stacking_train, dtype=torch.float32).clone().detach()
    labels = torch.tensor(labels, dtype=torch.long).clone().detach()
    channels = stacking_train.shape[1]
    height = 1
    width = 1
    # 重新调整输入数据的形状
    stacking_train = stacking_train.reshape(-1, channels, height, width)
    print('reshaped stacking_train.shape:', stacking_train.shape)

    print("Before train_test_split:")
    print("labels shape:", labels.shape)
    # 将数据集分为训练集和测试集
    X_train, X_val, y_train, y_val = train_test_split(stacking_train.numpy(), labels.numpy(), test_size=0.2,
                                                      random_state=42)
    # 将分割后的数据集转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    batch_size = 64  # 设置批量大小
    # Train Batch Size: 1,5,1,1 最后一个epoch只有一个数据，使用drop_last丢掉，防止batch_size在训练过程中不大于1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validate_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    EP = 10
    for ep in range(EP):
        model = meta_EfficientNet(channels=channels, num_classes=num_classes).to(device)
        model.to(device)  # 将模型移动到 GPU（如果可用）
        # 定义损失函数和优化器
        # loss_function = nn.CrossEntropyLoss()
        loss_function = FocalLoss(gamma=4, alpha=0.75)
        optimizer = optim.Adamax(model.parameters(), lr=0.001)
        print('Epoch %d' % ep)
        # 确保输入数据的形状为 [batch_size, channels, height, width]
        best_acc = 0.0
        for epoch in range(epochs):
            if train_loader:
                model.train()
                running_loss = 0.0
                train_bar = tqdm(train_loader)
                for step, data in enumerate(train_bar):
                    images, labels = data
                    optimizer.zero_grad()
                    outputs = model(images.to(device))
                    loss = loss_function(outputs, labels.to(device))
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

            model.eval()
            acc = 0.0
            all_predictions = []
            with torch.no_grad():
                val_bar = tqdm(validate_loader)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = model(val_images.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                    all_predictions.extend(predict_y.cpu().numpy())

                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

            val_accurate = acc / len(validate_loader.dataset)
            print('[epoch %d] train_loss: %.5f  val_accuracy: %.6f' % (
                epoch + 1, running_loss / len(train_loader) if train_loader else 0, val_accurate))

            # 测试模型
            validate2(model=model, validate_loader=validate_loader, device=device, dataset_name=dataset_name,
                      num_classes=num_classes)

            save_path = f'./models/{dataset_name}/meta_Efficientnet_model_best_{dataset_name}_{ep}.pth'
            if val_accurate > best_acc:
                best_acc = val_accurate
                print("saving at best epoch{}".format(epoch))
                torch.save(model.state_dict(), save_path)
        save_path = f'./models/{dataset_name}/Efficientnet_model_final{ep}.pt'
        torch.save(model, save_path)
        print('Finished Training! best accuracy:', best_acc)


def meta_predict(stacking_train, labels, device, num_classes, dataset_name):
    # 转换为 PyTorch 张量
    stacking_train = torch.tensor(stacking_train, dtype=torch.float32).clone().detach()
    labels = torch.tensor(labels, dtype=torch.long).clone().detach()
    channels = stacking_train.shape[1]
    height = 1
    width = 1
    # 重新调整输入数据的形状
    stacking_train = stacking_train.reshape(-1, channels, height, width)
    print('reshaped stacking_train.shape:', stacking_train.shape)

    stacking_train_tensor = torch.tensor(stacking_train, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

    # 创建数据集和数据加载器
    dataset = TensorDataset(stacking_train_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    preds = []
    accuracies = []
    tprs = []
    fprs = []
    f1_scores = []
    EP = 10
    global hyper_epoch, all_predictions
    for ep in range(EP):
        hyper_epoch = ep
        model_path = f'./models/{dataset_name}/meta_Efficientnet_model_best_{dataset_name}_{ep}.pth'
        model_state_dict = torch.load(model_path, device)
        model = meta_EfficientNet(channels=channels, num_classes=num_classes).to(device)
        model.load_state_dict(model_state_dict)
        model.to(device)  # 将模型移动到 GPU（如果可用）

        print('Epoch %d' % ep)
        # 测试模型
        accuracy, precision, recall, TPR, FPR, f1, all_predictions = validate2(model=model, validate_loader=dataloader,
                                                                               device=device,
                                                                               dataset_name=dataset_name,
                                                                               num_classes=num_classes)
        accuracies.append(accuracy)
        fprs.append(FPR)
        tprs.append(TPR)
        f1_scores.append(f1)
    accuracies = np.array(accuracies)
    f1_scores = np.array(f1_scores).flatten()
    fprs = np.array(fprs).flatten()
    tprs = np.array(tprs).flatten()
    print('Mean accuracy  (%)  ', f'{np.mean(accuracies):.4f}±{np.std(accuracies):.4f}')
    print('Mean FPR            ', f'{np.mean(fprs):.4f}±{np.std(fprs):.4f}')
    print('Mean TPR            ', f'{np.mean(tprs):.4f}±{np.std(tprs):.4f}')
    print('Mean F1 score  (%)  ', f'{np.mean(f1_scores):.4f}±{np.std(f1_scores):.4f}')

    csv_file_path = f'./predict/{dataset_name}_meta_EfficientNet.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration'] + [str(i + 1) for i in range(10)])
        writer.writerow(['Accuracy'] + [f"{acc:.4f}" for acc in accuracies])
        writer.writerow(['TPR'] + [f"{tpr:.4f}" for tpr in tprs])
        writer.writerow(['FPR'] + [f"{fpr:.4f}" for fpr in fprs])
        writer.writerow(['F1 Score'] + [f"{f1:.4f}" for f1 in f1_scores])

    print(f"Results saved to {csv_file_path}")
    return all_predictions


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    png_path = '4_Png_16_ISAC'
    train_loader, test_loader, labels, _, _, dataset_name = data_pre_process(os.getcwd(), png_path)
    num_classes = get_class_nums(png_path)

    accuracies = []
    f1_scores = []
    tprs = []
    fprs = []
    recalls = []
    precisions = []
    for i in range(10):
        print("la:", i)
        model = EfficientNet(num_classes=num_classes).to(device)
        model_path = f"../models/{dataset_name}/EfficientNet_model_best_" + dataset_name + str(i) + ".pth"
        model_state_dict = torch.load(model_path)
        model.load_state_dict(model_state_dict)  # 直接加载状态字典到模型中
        model.to(device)  # 将模型移动到指定设备
        metrics = validate2(model, test_loader, device, dataset_name, num_classes)

        accuracies.append(metrics['accuracy'])
        f1_scores.append(metrics['f1'])
        tprs.append(metrics['tpr'])
        fprs.append(metrics['fpr'])
        recalls.append(metrics['recall'])
        precisions.append(metrics['precision'])

    # 改为numpy格式
    accuracies = np.array(accuracies)
    f1_scores = np.array(f1_scores).flatten()
    tprs = np.array(tprs)
    fprs = np.array(fprs)
    recalls = np.array(recalls).flatten()
    precisions = np.array(precisions).flatten()

    print(f"mean_accuracy: {np.mean(accuracies):.4f}±{np.std(accuracies):.4f} ")
    print(f"mean_f1_score: {np.mean(f1_scores):.4f}±{np.std(f1_scores):.4f} ")
    print(f"mean_tpr: {np.mean(tprs):.4f}±{np.std(tprs):.4f}")
    print(f"mean_fpr: {np.mean(fprs):.4f}±{np.std(fprs):.4f}")
    print(f"mean_recall: {np.mean(recalls):.4f}±{np.std(recalls):.4f}")
    print(f"mean_precision: {np.mean(precisions):.4f}±{np.std(precisions):.4f}")


if __name__ == '__main__':
    main()
