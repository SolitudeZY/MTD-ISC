import os
import json

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, classification_report
from torchvision import transforms
import matplotlib.pyplot as plt
from resnet_model import resnet34, resnet50, resnet101, resnext101_32x8d, resnext50_32x4d
# from renet_model_improved import resnet34, resnet50, resnet101, resnext101_32x8d, resnext50_32x4d
from torchvision import transforms, datasets
from tqdm import tqdm
from prettytable import PrettyTable
from utils import data_pre_process, get_class_nums, calculate_tpr_fpr
from sklearn.metrics import accuracy_score


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

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the ResNet accuracy is ", acc)
        F1_list = []
        FPR_list = []
        TPR_list = []  # 新增TPR列表
        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "F1", "FPR", "TPR"]  # 新增TPR列
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            # Accuracy = round((TP + TN) / (TP + FP + FN + TN), 3) if TP + FP + FN + TN != 0 else 0.
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            F1 = round(2 * (TP / (TP + FP) * (TP / (TP + FN))) / ((TP / (TP + FP)) + (TP / (TP + FN))), 3) if ((TP / (
                    TP + FP)) + (TP / (TP + FN))) and TP + FP and TP + FN != 0 else 0
            FPR = round(FP / (TN + FP), 3) if TN + FP != 0 else 0.
            TPR = round(Recall, 3)  # TPR = Recall

            F1_list.append(F1)
            FPR_list.append(FPR)
            TPR_list.append(TPR)

            table.add_row([self.labels[i], Precision, Recall, F1, FPR, TPR])  # 新增TPR数据

        F1 = 0
        fpr = 0
        tpr = 0
        for k in range(len(F1_list)):
            F1 = F1 + F1_list[k]
            fpr = fpr + FPR_list[k]
            tpr = tpr + TPR_list[k]  # 新增TPR累加

        print("F1-measure:", F1 / self.num_classes)
        print("macro FPR:", fpr / self.num_classes)
        print("macro TPR:", tpr / self.num_classes)  # 新增TPR平均值
        print(table)

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


def predict(resnet_model, reshaped_val, input_channel, seq_length, num_classes):
    return None


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [
            transforms.Resize(16),
            transforms.CenterCrop(16),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])  # 采用和训练方法一样的图像标准化处理，两者标准化参数相同。
    # transforms.Normalize([0.485], [0.229])])
    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
    image_path1 = os.path.join(data_root, "pre-processing", "4_Png_16_USTC")
    assert os.path.exists(image_path1), "{} path does not exist.".format(image_path1)
    test_dataset = datasets.ImageFolder(root=os.path.join(image_path1, "test"),
                                        transform=data_transform)
    test_num = len(test_dataset)

    test_loader = data_pre_process(data_directory=os.getcwd(), png_path="4_Png_16_USTC")[1]
    print(test_num)
    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    all_f1_scores = []
    all_accuracies = []
    all_TPRs = []
    all_FPRs = []
    epochs = 10
    for i in range(epochs):
        print("test :{}".format(i))
        directory_path = os.path.join(os.path.abspath(os.getcwd()), "../pre-processing/4_Png_16_USTC/Train")
        entries = os.listdir(directory_path)
        folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
        num_classes = folder_count

        model = resnet34(num_classes=num_classes).to(device)  # 实例化模型时，将数据集分类个数赋给num_classes并传入模型

        weights_path = "../models/USTC/Resnet34_" + str(i) + ".pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))  # 载入刚刚训练好的模型参数
        labels = [label for _, label in class_indict.items()]
        confusion = ConfusionMatrix(num_classes=num_classes, labels=labels)
        model.eval()
        Y_label = []
        Y_prob = []
        true_labels = []
        pred_labels = []
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

                true_labels.extend(val_labels.to("cpu").numpy())  # 将真实标签添加到列表中
                pred_labels.extend(outputs.to("cpu").numpy())  # 将预测标签添加到列表中

                confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
        # print("Y_label:", Y_label)
        # print("Y_prob", Y_prob)
        auc_score = roc_auc_score(Y_label, Y_prob, multi_class="ovr")
        # confusion.plot()
        confusion.summary()
        TPR, FPR = calculate_tpr_fpr("class_indices.json", true_labels=true_labels, pred_labels=pred_labels)
        print(f"TPR:, {TPR}, FPR:, {FPR}")
        print("auc_score:", auc_score)
        report = classification_report(Y_label, Y_prob, zero_division=1, output_dict=True)
        accuracy = report['accuracy']
        f1_score = report['weight avg']['f1-score']
        all_accuracies.append(accuracy)
        all_f1_scores.append(f1_score)
        all_TPRs.append(TPR)
        all_FPRs.append(FPR)


if __name__ == '__main__':
    main()


