import csv
import os
import json
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
# from sklearn.metrics import confusion_matrix
from FocalLoss import FocalLoss
from Resnet_new.resnet_model import resnet34, resnet50, resnet101, resnext101_32x8d, resnext50_32x4d
from Resnet_new.meta_ResNet import meta_resnet
# from renet_model_improved import resnet34, resnet50, resnet101, resnext101_32x8d, resnext50_32x4d
from tqdm import tqdm
from prettytable import PrettyTable
from utils import data_pre_process, get_class_nums, calculate_tpr_fpr_multiclass


# from sklearn.metrics import roc_auc_score
class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    # def summary(self):
    #     # calculate accuracy
    #     sum_TP = 0
    #     for i in range(self.num_classes):
    #         sum_TP += self.matrix[i, i]
    #     acc = sum_TP / np.sum(self.matrix)
    #     print("the ResNet accuracy is ", acc)
    #     F1_list = []
    #     FPR_list = []
    #     TPR_list = []  # 新增TPR列表
    #     # precision, recall, specificity
    #     table = PrettyTable()
    #     table.field_names = ["", "Precision", "Recall", "F1", "FPR", "TPR"]  # 新增TPR列
    #     for i in range(self.num_classes):
    #         TP = self.matrix[i, i]
    #         FP = np.sum(self.matrix[i, :]) - TP
    #         FN = np.sum(self.matrix[:, i]) - TP
    #         TN = np.sum(self.matrix) - TP - FP - FN
    #         # Accuracy = round((TP + TN) / (TP + FP + FN + TN), 3) if TP + FP + FN + TN != 0 else 0.
    #         Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
    #         Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
    #         F1 = round(2 * (TP / (TP + FP) * (TP / (TP + FN))) / ((TP / (TP + FP)) + (TP / (TP + FN))), 3) if ((TP / (
    #                     TP + FP)) + (TP / (TP + FN))) and TP + FP and TP + FN != 0 else 0
    #         FPR = round(FP / (TN + FP), 3) if TN + FP != 0 else 0.
    #         TPR = round(Recall, 3)  # TPR = Recall
    #
    #         F1_list.append(F1)
    #         FPR_list.append(FPR)
    #         TPR_list.append(TPR)
    #
    #         table.add_row([self.labels[i], Precision, Recall, F1, FPR, TPR])  # 新增TPR数据
    #
    #     F1 = 0
    #     fpr = 0
    #     tpr = 0
    #     for k in range(len(F1_list)):
    #         F1 = F1 + F1_list[k]
    #         fpr = fpr + FPR_list[k]
    #         tpr = tpr + TPR_list[k]  # 新增TPR累加
    #
    #     print("F1-measure:", F1 / self.num_classes)
    #     print("FPR:", fpr / self.num_classes)
    #     print("TPR:", tpr / self.num_classes)  # 新增TPR平均值
    #     print(table)
    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the ResNet accuracy is ", acc)

        # 读取类别到索引的映射，找到正常流量的类别标签
        with open('class_indices.json', 'r') as json_file:
            cla_dict = json.load(json_file)

        # 找到值为 "Normal" 的索引
        normal_traffic_idx = None
        for idx, class_name in cla_dict.items():
            if class_name == "Normal":
                normal_traffic_idx = int(idx)
                break
        if normal_traffic_idx is None:
            raise ValueError("Class 'Normal' not found in the JSON file.")

        # 恶意流量的标签
        malicious_traffic_indices = list(range(self.num_classes))
        malicious_traffic_indices.remove(normal_traffic_idx)

        # 统计正常流量的数量及其正确分类的数量
        tp_normal = self.matrix[normal_traffic_idx, normal_traffic_idx]
        fn_normal = np.sum(self.matrix[normal_traffic_idx, :]) - tp_normal
        fp_normal = np.sum(self.matrix[:, normal_traffic_idx]) - tp_normal
        tn_normal = np.sum(self.matrix) - tp_normal - fp_normal - fn_normal

        # 统计恶意流量的数量及其正确分类的数量
        tp_malicious = 0
        fn_malicious = 0
        fp_malicious = 0
        tn_malicious = 0

        for idx in malicious_traffic_indices:
            # 提取 TP, FP, FN, TN
            tp = self.matrix[idx, idx]
            fp = np.sum(self.matrix[:, idx]) - tp
            fn = np.sum(self.matrix[idx, :]) - tp
            tn = np.sum(self.matrix) - tp - fp - fn

            tp_malicious += tp
            fn_malicious += fn
            fp_malicious += fp
            tn_malicious += tn

        # 计算总的 TPR 和 FPR
        total_tp = tp_normal + tp_malicious
        total_fp = fp_normal + fp_malicious
        total_fn = fn_normal + fn_malicious
        total_tn = tn_normal + tn_malicious

        total_tpr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        total_fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0

        # 输出结果
        print(f"Total TP: {int(total_tp)}, TN: {int(total_tn)}, FP: {int(total_fp)}, FN: {int(total_fn)}")
        print(f"Total: TPR: {total_tpr:.6f}, FPR: {total_fpr:.6f}")

        # precision, recall, specificity
        table = PrettyTable()
        F1_list = []
        table.field_names = ["Accuracy", "Precision", "Recall", "F1", "FPR", "TPR"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            F1 = round(2 * (TP / (TP + FP) * (TP / (TP + FN))) / ((TP / (TP + FP)) + (TP / (TP + FN))), 3) if ((TP / (
                    TP + FP)) + (TP / (TP + FN))) and TP + FP and TP + FN != 0 else 0.
            FPR = round(FP / (TN + FP), 3) if TN + FP != 0 else 0.
            TPR = round(Recall, 3)  # TPR = Recall
            F1_list.append(F1)
            table.add_row([self.labels[i], Precision, Recall, F1, FPR, TPR])
        F1 = 0
        for k in range(len(F1_list)):
            F1 = F1 + F1_list[k]
        print("F1-measure:", F1 / self.num_classes)
        print(table)
        return total_tpr, total_fpr

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


def fit(stacking_train, labels, device, num_classes, dataset_name, epochs=5, ):
    # batch_size, channels, height, weight = 64，3，1，1
    # num_classes = 1000
    # 训练模型
    EP = 10
    channels = stacking_train.shape[1]
    height = 1
    width = 1
    # 重新调整输入数据的形状
    stacking_train = stacking_train.reshape(-1, channels, height, width)
    for ep in range(EP):
        print("L epoch", ep)

        # 确保输入数据的形状为 [batch_size, channels, height, width]
        print('reshaped stacking_train.shape:', stacking_train.shape)
        model = meta_resnet(channels=channels)  # meta_resnet
        # num_ftrs = model.fc.in_features  # 获取分类器层的输入特征数量
        # model.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)  # 替换为新的分类器层
        model.to(device)  # 将模型移动到 GPU（如果可用）
        batch_size = 64  # 设置批量大小

        # 转换为 PyTorch 张量
        stacking_train = torch.tensor(stacking_train, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

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
        # Train Batch Size: 1,5,1,1 最后一个epoch只有一个数据，使用drop_last丢掉，防止batch_size在训练过程中不大于1
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        # 定义损失函数和优化器
        # criterion = nn.CrossEntropyLoss()
        criterion = FocalLoss(gamma=4, alpha=0.75)
        optimizer = optim.Adagrad(model.parameters(), lr=0.0005)

        best_val_accuracy = 0.0  # 初始化最佳验证准确率
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                # print(f"Train Batch Size: {inputs.size(0)},{inputs.size(1)},{inputs.size(2)},{inputs.size(3)}")
                # 打印每个批次的大小
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            train_loss = running_loss / len(train_loader.dataset)

            # 验证模型
            model.eval()
            running_val_loss = 0.0
            correct = 0
            true_labels = []
            pred_labels = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    running_val_loss += loss.item() * inputs.size(0)

                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()

                    # 收集真实标签和预测标签
                    true_labels.extend(targets.cpu().numpy())
                    pred_labels.extend(predicted.cpu().numpy())

            val_loss = running_val_loss / len(val_loader.dataset)
            val_accuracy = correct / len(val_loader.dataset)

            report = classification_report(y_true=true_labels, y_pred=pred_labels, output_dict=True, zero_division=1)

            f1_score = report['weighted avg']['f1-score']
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            accuracy = report['accuracy']
            TPR, FPR = calculate_tpr_fpr_multiclass(y_true=true_labels, y_pred=pred_labels, n_classes=num_classes)

            print(f"\nAccuracy on validation set: {accuracy}")
            print(f"Precision: {precision:.6f}, \n",
                  f"Recall: {recall:.6f},\n "
                  f"F1 Score: {f1_score:.6f}")
            print(f"TPR: {TPR}, FPR: {FPR}\n")
            print(
                f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.8f}')
            save_path = f'./models/{dataset_name}/meta_ResNet_model_best_' + str(ep) + '.pth'
            if val_accuracy > best_val_accuracy:
                print("saving at ", save_path)
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved at epoch {epoch + 1} with validation accuracy: {best_val_accuracy:.6f}")

            print(f"finish train,best accuracy {best_val_accuracy}")


def meta_predict(stacking_train, labels, dataset_name, num_classes):
    global pred_labels
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EP = 10
    accuracies = []
    tprs = []
    fprs = []
    f1_scores = []
    for ep in range(EP):
        print("Epoch : " + str(ep))
        # 确保输入数据的形状为 [batch_size, channels, height, width]
        channels = stacking_train.shape[1]
        height = 1
        width = 1
        # 重新调整输入数据的形状
        model = meta_resnet(channels=channels)  # meta_resnet
        stacking_train = stacking_train.reshape(-1, channels, height, width)
        print('reshaped stacking_train.shape:', stacking_train.shape)
        model_path = f'./models/{dataset_name}/meta_ResNet_model_best_' + str(ep) + '.pth'
        model_state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(model_state_dict)
        model.to(device)  # 将模型移动到 GPU（如果可用）
        batch_size = 64  # 设置批量大小

        # # 转换为 PyTorch 张量
        # stacking_train = torch.tensor(stacking_train, dtype=torch.float32)
        # labels = torch.tensor(labels, dtype=torch.long)
        stacking_train_tensor = torch.tensor(stacking_train, dtype=torch.float32).to(device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

        # 创建数据集和数据加载器
        dataset = TensorDataset(stacking_train_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        # 验证模型
        model.eval()
        running_val_loss = 0.0
        correct = 0
        true_labels = []
        pred_labels = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()

                # 收集真实标签和预测标签
                true_labels.extend(targets.cpu().numpy())
                pred_labels.extend(predicted.cpu().numpy())

        val_loss = running_val_loss / len(dataloader.dataset)
        val_accuracy = correct / len(dataloader.dataset)

        report = classification_report(y_true=true_labels, y_pred=pred_labels, output_dict=True, zero_division=1)

        f1_score = report['weighted avg']['f1-score']
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        accuracy = report['accuracy']
        TPR, FPR = calculate_tpr_fpr_multiclass(y_true=true_labels, y_pred=pred_labels, n_classes=num_classes)

        if ep == 0:
            # 计算混淆矩阵
            cm = confusion_matrix(true_labels, pred_labels)
            # 打印混淆矩阵
            print("Confusion Matrix:")
            print(cm)
            # 保存混淆矩阵到文件
            cm_file_path = f'{dataset_name}_ResNet_confusion_matrix.csv'
            np.savetxt(cm_file_path, cm, delimiter=',', fmt='%d')
            print(f"Confusion matrix saved to {cm_file_path}")

        accuracies.append(val_accuracy)
        f1_scores.append(f1_score)
        fprs.append(FPR)
        tprs.append(TPR)
        print(f"\nAccuracy on validation set: {accuracy}")
        print(f"Precision: {precision:.6f}, \n",
              f"Recall: {recall:.6f},\n "
              f"F1 Score: {f1_score:.6f} \n")
        print(f"TPR: {TPR}, FPR: {FPR}")

    accuracies = np.array(accuracies).flatten()
    f1_scores = np.array(accuracies).flatten()
    fprs = np.array(fprs).flatten()
    tprs = np.array(tprs).flatten()

    print('Mean accuracy  (%)  ', f'{np.mean(accuracies):.4f}±{np.std(accuracies):.4f}')
    print('Mean FPR            ', f'{np.mean(fprs):.4f}±{np.std(fprs):.4f}')
    print('Mean FPR            ', f'{np.mean(tprs):.4f}±{np.std(tprs):.4f}')
    print('Mean F1 score  (%)  ', f'{np.mean(f1_scores):.4f}±{np.std(f1_scores):.4f}')

    csv_file_path = f'./predict/{dataset_name}_meta_ResNet.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration'] + [str(i + 1) for i in range(10)])
        writer.writerow(['Accuracy'] + [f"{acc:.4f}" for acc in accuracies])
        writer.writerow(['TPR'] + [f"{tpr:.4f}" for tpr in tprs])
        writer.writerow(['FPR'] + [f"{fpr:.4f}" for fpr in fprs])
        writer.writerow(['F1 Score'] + [f"{f1:.4f}" for f1 in f1_scores])

    print(f"Results saved to {csv_file_path}")
    return pred_labels


def predict(model, dataset_name, num_classes, test_loader, ):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    json_path = dataset_name + 'class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=num_classes, labels=labels)
    model.eval()
    Y_label = []
    Y_prob = []
    with torch.no_grad():
        for val_data in tqdm(test_loader):
            val_images, val_labels = val_data
            label = val_labels.numpy()
            Y_label.append(label[0])
            outputs = model(val_images.to(device))

            pro = torch.softmax(outputs, dim=1)
            pro = pro.cpu()
            prob = pro.numpy()
            Y_prob.append(prob[0])

            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    # print("Y_label:", Y_label)
    # print("Y_prob", Y_prob)
    y_true = np.array(Y_label)
    y_pred = np.argmax(np.array(Y_prob), axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    # TPR, FPR = calculate_tpr_fpr(json_filepath=json_path, true_labels=y_true, pred_labels=y_pred)
    TPR, FPR = calculate_tpr_fpr_multiclass(y_true=y_true, y_pred=y_pred, n_classes=num_classes)

    # auc_score = roc_auc_score(Y_label, Y_prob, multi_class="ovr")
    # confusion.plot()
    confusion.summary()
    # print("auc_score:", auc_score)
    return accuracy, TPR, FPR, f1


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    png_path = '4_Png_16_ISAC'
    directory_path = os.path.join(os.path.abspath(os.getcwd()), f"../pre-processing/{png_path}/Train")
    entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    num_classes = folder_count
    accuracies = []
    f1_scores = []
    tprs = []
    fprs = []
    train_loader, test_loader, labels, _, _, dataset_name = data_pre_process(os.getcwd(), png_path)

    for i in range(10):
        model = resnet34(num_classes=num_classes).to(device)  # 实例化模型时，将数据集分类个数赋给num_classes并传入模型
        weights_path = f"../models/{dataset_name[:-1]}/Resnet34_{dataset_name}" + str(i) + ".pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))  # 载入刚刚训练好的模型参数

        accuracy, tpr_value, fpr_value, f1 = predict(model=model, dataset_name=dataset_name, num_classes=num_classes,
                                                     test_loader=test_loader)

        accuracies.append(accuracy)
        f1_scores.append(f1)
        tprs.append(tpr_value)
        fprs.append(fpr_value)

    accuracies = np.array(accuracies).flatten()
    f1_scores = np.array(f1_scores).flatten()
    fprs = np.array(fprs).flatten()
    tprs = np.array(tprs).flatten()
    # 计算均值和方差
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    mean_tpr = np.mean(tprs)
    std_tpr = np.std(tprs)
    mean_fpr = np.mean(fprs)
    std_fpr = np.std(fprs)

    print(f'Mean Accuracy: {mean_accuracy:.4f}±{std_accuracy:.4f}')
    print(f'Mean F1 Score: {mean_f1:.4f}±{std_f1:.4f}')
    print(f'Mean TPR: {mean_tpr:.4f}±{std_tpr:.4f}')
    print(f'Mean FPR: {mean_fpr:.4f}±{std_fpr:.4f}')
    # print(f'predictions : {predictions}')

    csv_file_path = f'{dataset_name}ResNet.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration'] + [str(i + 1) for i in range(10)])
        writer.writerow(['Accuracy'] + [f"{acc:.4f}" for acc in accuracies])
        writer.writerow(['TPR'] + [f"{tpr:.4f}" for tpr in tprs])
        writer.writerow(['FPR'] + [f"{fpr:.4f}" for fpr in fprs])
        writer.writerow(['F1 Score'] + [f"{f1:.4f}" for f1 in f1_scores])

    print(f"Results saved to {csv_file_path}")


if __name__ == '__main__':
    main()
