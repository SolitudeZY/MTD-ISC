import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from FocalLoss import FocalLoss
from utils import data_pre_process, get_class_nums, calculate_tpr_fpr, calculate_tpr_fpr_multiclass


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
class EfficientNet(nn.Module):
    def __init__(self, num_classes, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.05):
        super(EfficientNet, self).__init__()

        # 调整 Stem 层的尺寸
        self.stem = nn.Sequential(
            nn.Conv2d(3, int(32 * width_coefficient), kernel_size=3, stride=1, padding=1, bias=False),  # 将 stride 改为 1
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


def load_model(model, load_path, device):
    model.load_state_dict(torch.load(load_path, map_location=device))
    return model


def train_and_validate(model, train_loader, validate_loader, device, epochs=10, ep=0, num_classes=10):
    # loss_function = nn.CrossEntropyLoss()
    loss_function = FocalLoss(2, 0.75)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_acc = 0.0
    for epoch in range(epochs):
        if train_loader:
            model.train()
            running_loss = 0.0
            # train_bar = tqdm(train_loader)
            # train_bar = tqdm(train_loader, leave=True, bar_format='{l_bar}{bar:30}{r_bar}', colour='green')
            train_bar = tqdm(train_loader, leave=True, bar_format='\033[32m{l_bar}{bar:30}{r_bar}\033[0m')
            for step, data in enumerate(train_bar):
                images, labels = data
                optimizer.zero_grad()
                logits = model(images.to(device))
                loss = loss_function(logits, labels.to(device))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                train_bar.desc = "train epoch[{}/{}] loss:{:.6f}".format(epoch + 1, epochs, loss)

        model.eval()
        acc = 0.0
        all_predictions = []
        true_labels = []
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                all_predictions.extend(predict_y.cpu().numpy())
                true_labels.extend(val_labels.cpu().numpy())
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accurate = acc / len(validate_loader.dataset)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (
            epoch + 1, running_loss / len(train_loader) if train_loader else 0, val_accurate))

        re = classification_report(true_labels, all_predictions,zero_division=1, output_dict=True)
        print('acc: ', re['accuracy'])
        print('precision: ', re['weighted avg']['precision'])
        print('recall: ', re['weighted avg']['recall'])
        print('f1_score: ', re['weighted avg']['f1-score'])
        tpr, fpr = calculate_tpr_fpr_multiclass(true_labels, all_predictions, num_classes)
        # save_path = './models/USTC/EfficientNet_model_best_' + dataset_name + str(ep) + '.pth'  # 改路径
        # if val_accurate > best_acc:
        #     best_acc = val_accurate
        #     print("saving at best epoch{}".format(epoch))
        #     torch.save(model.state_dict(), save_path)
        # save_path = './models/USTC/Efficientnet_model_final_' + dataset_name + str(ep) + '.pth'
        # torch.save(model.state_dict(), save_path)
    print('Finished Training! best accuracy:', best_acc)


def validate(model, validate_loader, device):
    model.eval()
    acc = 0.0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        # val_bar = tqdm(validate_loader)
        val_bar = tqdm(validate_loader, leave=True, bar_format='\033[32m{l_bar}{bar:30}{r_bar}\033[0m')
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            all_predictions.extend(predict_y.cpu().numpy())
            all_labels.extend(val_labels.cpu().numpy())

            val_bar.desc = "Validation"

    val_accurate = acc / len(validate_loader.dataset)
    print('Validation accuracy: %.3f' % val_accurate)


def validate2(model, validate_loader, device, dataset_name, num_classes: int):
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

    tpr, fpr = calculate_tpr_fpr(json_filepath=dataset_name + 'class_indices.json', true_labels=all_labels,
                                 pred_labels=all_predictions)
    TPR, FPR = calculate_tpr_fpr_multiclass(y_true=all_labels, y_pred=all_predictions, n_classes=num_classes)
    # 打印结果
    print('Validation accuracy: %.5f' % val_accurate)
    print('Precision: %.5f' % precision)
    print('Recall: %.5f' % recall)
    print('F1 Score: %.5f' % f1)
    print(f'Total TPR: {tpr:.6f}, FPR: {fpr:.6f}')
    print(f"weighted Tpr: {TPR}, Fpr: {FPR}")

    # 打印分类报告
    # report = classification_report(all_labels, all_predictions, zero_division=1, output_dict=True)
    # print(report)

    # 返回结果
    return {
        'accuracy': val_accurate,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def main():
    PNG_PATH = '4_Png_16_CTU'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = get_class_nums(PNG_PATH)  # 4_Png_16_CTU
    epoch = 3
    # Train the model
    for ep in range(1):
        print("L epoch:", ep)
        model = EfficientNet(num_classes=num_classes).to(device)
        # model = EfficientNet(num_classes=num_classes, width_coefficient=0.8, depth_coefficient=0.8).to(device)

        train_loader, test_loader, _, _, _, dataset_name = data_pre_process(os.getcwd(), PNG_PATH)

        train_and_validate(model, train_loader, test_loader, device, epochs=epoch,
                           ep=ep, num_classes=num_classes,
                           )

        # Validate the model
        validate2(model, test_loader, device, dataset_name=dataset_name, num_classes=num_classes)


if __name__ == "__main__":
    main()

# def validate2(model, validate_loader, device):
#     model.eval()
#     acc = 0.0
#     all_predictions = []
#     all_labels = []
#     with torch.no_grad():
#         val_bar = tqdm(validate_loader)
#         for val_data in val_bar:
#             val_images, val_labels = val_data
#             outputs = model(val_images.to(device))
#             predict_y = torch.max(outputs, dim=1)[1]
#             acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
#
#             all_predictions.extend(predict_y.cpu().numpy())
#             all_labels.extend(val_labels.cpu().numpy())
#
#             val_bar.desc = "Validation"
#
#     val_accurate = acc / len(validate_loader.dataset)
#
#     # 计算TPR和FPR
#     TP = sum((np.array(all_predictions) == 1) & (np.array(all_labels) == 1))  # True Positives
#     TN = sum((np.array(all_predictions) == 0) & (np.array(all_labels) == 0))  # True Negatives
#     FP = sum((np.array(all_predictions) == 1) & (np.array(all_labels) == 0))  # False Positives
#     FN = sum((np.array(all_predictions) == 0) & (np.array(all_labels) == 1))  # False Negatives
#
#     # 计算TPR和FPR
#     TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
#     FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
#
#     precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)
#     recall = recall_score(all_labels, all_predictions, average='weighted')
#     f1 = f1_score(all_labels, all_predictions, average='weighted')
#
#     print('Validation accuracy: %.5f' % val_accurate)
#     print('Precision: %.5f' % precision)
#     print('Recall: %.5f' % recall)
#     print('F1 Score: %.5f' % f1)
#     print('TPR (True Positive Rate): %.5f' % TPR)
#     print('FPR (False Positive Rate): %.5f' % FPR)
