import csv
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn, optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import utils
from Attention import Attention
from FocalLoss import FocalLoss
from utils import data_pre_process, calculate_tpr_fpr
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report
import torch
from torch.utils.data import TensorDataset, DataLoader

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class BIRNNWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(BIRNNWithAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.elu = nn.ELU()  # 添加 ELU 激活函数
        self.attention = Attention(hidden_dim)  # 添加注意力机制

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)

        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 3 * 16 * 16)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 使用最后一个隐藏状态作为解码器的隐藏状态
        # 注意：这里我们使用最后一个隐藏状态 hn[-1] 作为解码器的隐藏状态
        # 这是因为在双向 LSTM 中，hn[-1] 包含了整个序列的信息
        # 如果需要使用更复杂的策略来获取解码器的隐藏状态，可以根据具体任务进行调整
        context = self.attention(hn[-1].unsqueeze(1), out)  # 应用注意力机制

        # 使用上下文向量作为特征表示
        out = self.fc(context)
        out = self.elu(out)  # 应用 ELU 激活函数
        return out


class BIRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(BIRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.elu = nn.ELU()  # 添加 ELU 激活函数

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)

        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 3 * 16 * 16)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])
        out = self.elu(out)  # 应用 ELU 激活函数
        return out


class meta_BIRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(meta_BIRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.elu = nn.ELU()  # 添加 ELU 激活函数

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)

        # 调整输入数据形状
        x = x.permute(0, 2, 1)  # 调整为 (batch_size, feature_dim, sequence_length)

        # 调整为 LSTM 输入形状 (batch_size, sequence_length, feature_dim)
        x = x.contiguous().view(x.size(0), x.size(2), -1)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])
        out = self.elu(out)  # 应用 ELU 激活函数
        return out


def fit(BiLSTM_num_classes, stacking_train, labels, dataset_name: str, epochs=10, input_size=3 * 16 * 16, hidden_size=128,
        num_layers=2,
        Device='cuda'):
    stacking_train_tensor = torch.tensor(stacking_train, dtype=torch.float32).to(Device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(Device)
    EP = 10
    # dataset_name = dataset_name[:-1]
    for ep in range(EP):  # train
        print(f"L Epoch: {ep}")
        # 创建数据集和数据加载器
        dataset = TensorDataset(stacking_train_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        # 初始化模型
        model = meta_BIRNN(input_size, hidden_size, num_layers, BiLSTM_num_classes).to(Device)
        # 定义损失函数和优化器
        criterion = FocalLoss()
        learning_rate = 0.00063  # 示例学习率
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

        best_accuracy = 0.0
        for epoch in range(epochs):  # 每个EP训练epochs次
            model.train()
            running_loss = 0.0

            for images, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}')

            # 评估模型
            model.eval()
            all_predictions = []
            all_labels = []
            all_probabilities = []

            # 训练集评估
            with torch.no_grad():
                for images, labels in dataloader:
                    outputs = model(images)
                    predicted = torch.max(outputs.data, 1)[1]
                    all_predictions.extend(predicted.cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())
                    all_probabilities.extend(outputs.softmax(dim=1).cpu().numpy().tolist())

            # 计算训练集评估指标
            train_accuracy = accuracy_score(all_labels, all_predictions)
            train_precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)
            train_recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=1)
            train_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=1)

            print(f'\nTraining Set:')
            print(f'Accuracy: {train_accuracy:.6f}')
            print(f'Precision: {train_precision:.6f}')
            print(f'Recall: {train_recall:.6f}')
            print(f'F1 Score: {train_f1:.6f} \n')
            TPR, FPR = utils.calculate_tpr_fpr_multiclass(y_pred=all_predictions, n_classes=BiLSTM_num_classes,
                                                          y_true=all_labels)
            print(f"FPR: {FPR}, TPR: {TPR}")

            # 验证集评估
            all_predictions = []
            all_labels = []
            all_probabilities = []

            with torch.no_grad():
                for images, labels in dataloader:
                    outputs = model(images)
                    predicted = torch.max(outputs.data, 1)[1]
                    all_predictions.extend(predicted.cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())
                    all_probabilities.extend(outputs.softmax(dim=1).cpu().numpy().tolist())

            # 计算验证集评估指标
            val_accuracy = accuracy_score(all_labels, all_predictions)
            val_precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)
            val_recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=1)
            val_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=1)

            print(f'\nValidation Set:')
            print(f'Accuracy: {val_accuracy:.6f}')
            print(f'Precision: {val_precision:.6f}')
            print(f'Recall: {val_recall:.6f}')
            print(f'F1 Score: {val_f1:.6f} \n')
            TPR, FPR = utils.calculate_tpr_fpr_multiclass(y_pred=all_predictions, n_classes=BiLSTM_num_classes,
                                                          y_true=all_labels)
            print(f"FPR: {FPR}, TPR: {TPR}")
            print('---' * 10)

            save_path = f'./models/{dataset_name}/meta_BiLSTM_best_' + str(ep) + '.pth'
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                print(f"Best model saved at epoch {epoch + 1} with validation accuracy: {best_accuracy:.6f}")
                torch.save(model.state_dict(), save_path)


def predict_meta(device, num_classes, dataset_name, stacking_train, labels, input_size=3 * 16 * 16,
                 hidden_size=128, num_layers=2):
    global all_predictions
    stacking_train_tensor = torch.tensor(stacking_train, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    accuracies = []
    tprs = []
    fprs = []
    f1_scores = []
    for ep in range(10):
        print("test :{}".format(ep))

        print(f"stacking_train shape: {stacking_train.shape}")
        print(f"labels shape: {labels_tensor.shape}")
        # 创建数据集和数据加载器
        dataset = TensorDataset(stacking_train_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        # 加载保存的模型状态字典
        model_path = f'./models/{dataset_name}/meta_BiLSTM_best_' + str(ep) + '.pth'
        model_meta = meta_BIRNN(input_size, hidden_size, num_layers, num_classes).to(device)
        model_state_dict = torch.load(model_path)
        model_meta.load_state_dict(model_state_dict)
        model_meta.eval()

        # 验证集评估
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for images, labels in dataloader:
                outputs = model_meta(images)
                predicted = torch.max(outputs.data, 1)[1]
                all_predictions.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_probabilities.extend(outputs.softmax(dim=1).cpu().numpy().tolist())

        # 计算验证集评估指标
        val_accuracy = accuracy_score(all_labels, all_predictions)
        val_precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)
        val_recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=1)
        val_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=1)
        # tpr, fpr = calculate_tpr_fpr(dataset_name + "class_indices.json", all_labels, all_predictions)
        tpr, fpr = utils.calculate_tpr_fpr_multiclass(y_pred=all_predictions, n_classes=num_classes,
                                                      y_true=all_labels)

        if ep == 0:
            # 计算混淆矩阵
            cm = confusion_matrix(all_labels, all_predictions)
            # 打印混淆矩阵
            print("Confusion Matrix:")
            print(cm)
            # 保存混淆矩阵到文件
            cm_file_path = f'{dataset_name}_BiLSTM_confusion_matrix.csv'
            np.savetxt(cm_file_path, cm, delimiter=',', fmt='%d')
            print(f"Confusion matrix saved to {cm_file_path}")

        accuracies.append(val_accuracy)
        fprs.append(fpr)
        tprs.append(tpr)
        f1_scores.append(val_f1)

        print(f'\nValidation Set:')
        print(f'Accuracy: {val_accuracy:.6f}')
        print(f'Precision: {val_precision:.6f}')
        print(f'Recall: {val_recall:.6f}')
        print(f'F1 Score: {val_f1:.6f} \n')
        FPR, TPR = utils.calculate_tpr_fpr_multiclass(all_labels, all_predictions, num_classes)
        print(f"FPR: {FPR}, TPR: {TPR}")
        print('---' * 10)

    accuracies = np.array(accuracies)
    f1_scores = np.array(f1_scores).flatten()
    fprs = np.array(fprs).flatten()
    tprs = np.array(tprs).flatten()

    print(f"accuracies: {accuracies}")
    print(f"f1_scores: {f1_scores}")
    print(f"fprs: {fprs}")
    print(f"tprs: {tprs}")

    print('Mean accuracy  (%)  ', f'{np.mean(accuracies):.4f}±{np.std(accuracies):.4f}')
    print('Mean FPR            ', f'{np.mean(fprs):.4f}±{np.std(fprs):.4f}')
    print('Mean TPR            ', f'{np.mean(tprs):.4f}±{np.std(tprs):.4f}')
    print('Mean F1 score  (%)  ', f'{np.mean(f1_scores):.4f}±{np.std(f1_scores):.4f}')

    csv_file_path = f'./predict/{dataset_name}_meta_BiLSTM.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration'] + [str(i + 1) for i in range(10)])
        writer.writerow(['Accuracy'] + [f"{acc:.4f}" for acc in accuracies])
        writer.writerow(['TPR'] + [f"{tpr:.4f}" for tpr in tprs])
        writer.writerow(['FPR'] + [f"{fpr:.4f}" for fpr in fprs])
        writer.writerow(['F1 Score'] + [f"{f1:.4f}" for f1 in f1_scores])

    print(f"Results saved to {csv_file_path}")
    return all_predictions


def load_model(model_path, device):
    """
    Loads a trained model from a file. If the file contains only the model's state dictionary,
    it will create a new model instance and load the state dictionary into it.

    Args:
        model_path (str): Path to the saved model or state dictionary file.
        device (torch.device): Device to load the model onto.

    Returns:
        nn.Module: Loaded model.
    """
    # Check if the file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Try loading the entire model directly
    try:
        model = torch.load(model_path, map_location=device)
        if isinstance(model, nn.Module):
            print("Loaded entire model directly.")
            return model.to(device)
    except Exception as e:
        print(f"Failed to load entire model directly: {e}. Trying to load state_dict.")

    # If direct loading fails, assume it's a state_dict and load it into a new model instance
    try:
        # Determine the model architecture based on the file name or other logic
        # Here we assume the model is BIRNN for simplicity; you can add more complex logic if needed
        checkpoint = torch.load(model_path, map_location=device)
        input_size = 3 * 16 * 16
        hidden_size = 128
        num_layers = 2
        num_classes = checkpoint['num_classes'] if 'num_classes' in checkpoint else 10  # Adjust based on your needs

        model = BIRNN(input_size, hidden_size, num_layers, num_classes).to(device)
        model.load_state_dict(checkpoint)
        print("Loaded model from state_dict.")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model or state_dict from {model_path}: {e}")


def predict(model, data_loader, DEVICE, num_classes, dataset_name):
    """
    使用提供的模型预测给定数据加载器中的数据，并计算 Precision、Recall、F1 Score 和 Accuracy。

    参数:
    - model (nn.Module): 训练好的模型。
    - data_loader (DataLoader): 包含待预测数据的数据加载器。
    - DEVICE (str): 设备类型（例如 'cuda' 或 'cpu'）。
    - num_classes (int): 类别数量。

    返回:
    - List[int]: 预测的标签列表。
    - float: 预测的准确率。
    - List[float]: Precision 列表。
    - List[float]: Recall 列表。
    - List[float]: F1 Score 列表。
    """
    model.eval()  # 设置模型为评估模式
    all_predictions = []
    all_labels = []  # 存储所有真实标签
    correct = 0  # 初始化正确预测的数量
    total = 0  # 初始化总的预测数量

    with torch.no_grad():  # 不需要梯度计算
        for images, labels in data_loader:  # 遍历数据加载器中的每个批次
            images = images.to(DEVICE)  # 将图像转移到指定设备
            labels = labels.to(DEVICE)  # 将标签转移到指定设备
            outputs = model(images)  # 通过模型得到输出
            _, predicted = torch.max(outputs.data, 1)  # 获取预测标签
            all_predictions.extend(predicted.cpu().numpy().tolist())  # 将预测结果添加到列表中
            all_labels.extend(labels.cpu().numpy().tolist())  # 将真实标签添加到列表中
            total += labels.size(0)  # 更新总数
            correct += (predicted == labels).sum().item()  # 更新正确预测的数量

    accuracy = correct / total  # 计算准确率

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_predictions, labels=list(range(num_classes)))

    # TPR, FPR = calculate_tpr_fpr(dataset_name + "class_indices.json", true_labels=all_labels,
    #                              pred_labels=all_predictions)
    TPR, FPR = utils.calculate_tpr_fpr_multiclass(all_labels, all_predictions, num_classes)
    # 计算 Precision、Recall 和 F1 Score
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # 处理除零错误
    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0
    f1_score[np.isnan(f1_score)] = 0
    report = classification_report(all_labels, all_predictions,zero_division=1, output_dict=True)
    f1 = report['weighted avg']['f1-score']
    recall = report['weighted avg']['recall']

    print('precision:\n ', precision)
    print('recall: \n', recall)
    print('f1_score: \n', f1_score)
    print('accuracy: \n', accuracy)

    return all_predictions, accuracy, TPR, FPR, f1


def train_model(train_loader, class_nums, num_epochs=5, device=torch.device("cuda")):
    # Hyper Parameters
    sequence_length = 16
    input_size = 3 * 16 * 16
    hidden_size = 128
    num_layers = 2

    model = BIRNN(input_size, hidden_size, num_layers, class_nums).to(device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 初始化 GradScaler
    scaler = GradScaler()
    true_labels = []
    predictions = []
    for epoch in range(num_epochs):
        print("Epoch_" + str(epoch + 1))
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=True)

        for i, (images, labels) in enumerate(pbar):
            images = images.squeeze(1).float().to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            # 使用 autocast 上下文管理器
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # 使用 GradScaler 缩放损失
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            true_labels.extend(labels.cpu().numpy())

            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(probs, dim=1)
            predictions.extend(preds.cpu().numpy())

            pbar.set_description(f'Epoch {epoch + 1}/{num_epochs} Loss: {loss.item():.4f}')
    report = classification_report(true_labels, predictions, output_dict=True, zero_division=1)
    print("accuracy: ", accuracy_score(true_labels, predictions))
    print("precision: ", report['weighted avg']['precision'])
    print("recall: ", report['weighted avg']['recall'])
    print("f1_score: ", report['weighted avg']['f1-score'])
    return model


# 示例用法
# predictions, accuracy, precision, recall, f1_score = predict(model, data_loader, DEVICE, num_classes)

def main():
    PNG_PATH = "4_Png_16_ISAC"
    directory_path = os.path.join(os.path.abspath(os.getcwd()), f"../pre-processing/{PNG_PATH}/Train")
    entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    num_classes = folder_count

    # _, test_loader = data_loader(data_path)
    _, test_loader, _, _, _, dataset_name = utils.data_pre_process(data_directory=os.path.join(os.getcwd(), '../'),
                                                                   png_path=PNG_PATH)
    f1_scores = []
    fprs = []
    tprs = []
    accuracies = []
    for i in range(10):
        # model_path = '../models/USTC/meta_BiLSTM_best_6.pth'
        model_path = f'../models/{dataset_name[:-1]}/BiLSTM_final_{dataset_name}' + str(i) + '.pt'
        model = load_model(model_path, DEVICE)
        predictions, accuracy, tpr, fpr, f1 = predict(model, test_loader, DEVICE, num_classes, dataset_name)

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
    # print("Predictions:", predictions[:])
    # print("Accuracy:", accuracy)  # 打印准确率

    csv_file_path = f'{dataset_name}BiLSTM.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration'] + [str(i + 1) for i in range(10)])
        writer.writerow(['Accuracy'] + [f"{acc:.4f}" for acc in accuracies])
        writer.writerow(['TPR'] + [f"{tpr:.4f}" for tpr in all_tprs])
        writer.writerow(['FPR'] + [f"{fpr:.4f}" for fpr in all_fprs])
        writer.writerow(['F1 Score'] + [f"{f1:.4f}" for f1 in all_f1_scores])

    print(f"Results saved to {csv_file_path}")


if __name__ == "__main__":
    main()
