import os
import json

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms
import matplotlib.pyplot as plt
from renet_model import resnet34, resnet50, resnet101, resnext101_32x8d, resnext50_32x4d
# from renet_model_improved import resnet34, resnet50, resnet101, resnext101_32x8d, resnext50_32x4d
from torchvision import transforms, datasets
from tqdm import tqdm
from prettytable import PrettyTable


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
        print("the model accuracy is ", acc)
        F1_list = []
        FPR_list = []

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "F1", "FPR"]
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
            F1_list.append(F1)
            FPR_list.append(FPR)
            table.add_row([self.labels[i], Precision, Recall, F1, FPR])
        su1 = 0
        su2 = 0
        for k in range(len(F1_list)):
            su1 = su1 + F1_list[k]
        for k in range(len(FPR_list)):
            su2 = su2 + FPR_list[k]
        print("F1-measure:", su1 / 10)
        print("FPR:", su2 / 10)
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


def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(device)
    data_transform = transforms.Compose(
        [
            # transforms.Resize(32),  # 验证过程图像预处理有变动，将原图片的长宽比固定不动，将其最小边长缩放到256
            # transforms.CenterCrop(32),  # 再使用中心裁剪裁剪一个224×224大小的图片
            # transforms.Resize(256),  # 验证过程图像预处理有变动，将原图片的长宽比固定不动，将其最小边长缩放到256
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # 采用和训练方法一样的图像标准化处理，两者标准化参数相同。
    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
    image_path1 = os.path.join(data_root, "datasets", "UJS-new-improved-ddim-splite")  # flower data set path
    assert os.path.exists(image_path1), "{} path does not exist.".format(image_path1)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path1, "test"),
                                         transform=data_transform)
    test_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=nw)
    test_num = len(train_dataset)
    print(test_num)
    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    for i in range(10):
        print("la:{}".format(i))
        # create model
        model = resnet34(num_classes=10).to(device)  # 实例化模型时，将数据集分类个数赋给num_classes并传入模型


        # load model weights
        weights_path = "./resnet-UJS-new-improved-difftpt-splite-" + str(i) + ".pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))  # 载入刚刚训练好的模型参数
        labels = [label for _, label in class_indict.items()]
        confusion = ConfusionMatrix(num_classes=10, labels=labels)
        model.eval()
        Y_label = []
        Y_prob = []
        with torch.no_grad():
            for val_data in tqdm(test_loader):
                val_images, val_labels = val_data
                ########################## zyw
                label = val_labels.numpy()
                Y_label.append(label[0])
                ########################## zyw
                outputs = model(val_images.to(device))

                ########################## zyw
                pro = torch.softmax(outputs, dim=1)
                pro = pro.cpu()
                prob = pro.numpy()
                Y_prob.append(prob[0])
                ########################## zyw

                outputs = torch.softmax(outputs, dim=1)
                outputs = torch.argmax(outputs, dim=1)
                confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
        # print("Y_label:", Y_label)
        # print("Y_prob", Y_prob)
        auc_score = roc_auc_score(Y_label, Y_prob, multi_class="ovr")
        confusion.plot()
        confusion.summary()
        print("auc_score:", auc_score)


if __name__ == '__main__':
    main()
