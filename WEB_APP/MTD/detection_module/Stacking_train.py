import os
from os.path import exists
from time import sleep
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn
import json
from torch.utils.data import WeightedRandomSampler, SubsetRandomSampler
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from tqdm import tqdm
# from utils import data_pre_process
from stacking_esemble_classifier import StackingClassifier, save_labels, get_path
from predict.LSTM_predict import LSTM_model as lstm
from predict.BiLSTM_predict import BIRNNWithAttention
from BiLSTM.BiLSTM import BIRNN
from CNN_to_LSTM import CNN_for_LSTM, LSTM_model, CNNtoLSTM
from LSTM_predicts import LSTM_meta
from size_16_temp.CNN import CNN
from rnn import RNN, Attention
from tcn_train_and_test import TCN_model
from utils import calculate_mean_and_std
from BiTCN_pred_temp import meta_BiTCN
import LSTM_predict
import LSTM_predicts

base_classifiers = [
    'lstm_model',
    'BiTCN_model',
    'cnn_model',
    'BiLSTM_model',
    'tcn_model',
    # 'rnn_model',
]
# 选择不同的组合器
# combiner = 'cnn_model'
# combiner = 'rnn_model'
# combiner = 'lstm_model'
# combiner = 'tcn_model'
combiner = 'BiTCN_model'
# combiner = 'BiLSTM_model'
# combiner = 'ResNet_model'
# combiner = 'EfficientNet_model'

Stacking_Classifier = StackingClassifier(base_classifiers, combiner)


def get_clf():
    return Stacking_Classifier


def main():
    # 数据预处理
    png_path = get_path()
    root = os.getcwd()
    # png_path = '4_Png_16_CTU'
    X_train, X_test, y_test, y_train, dataset_name = data_pre_process(root, png_path, None)

    train_save_path = './' + dataset_name + '_stacking_train.npy'
    test_save_path = './' + dataset_name + '_stacking_test.npy'
    print('Train saved to: ', train_save_path)
    print('Test saved to: ', test_save_path)
    # 加载训练集
    if not os.path.exists(train_save_path):
        print(f'generating stacking_train……')
        print('train_save_path is :', train_save_path)
        print('test_save_path is :', test_save_path)
        stacking_train, stacking_test = StackingClassifier.fit(self=Stacking_Classifier, X_train=X_train, X_test=X_test,
                                                               dataset_name=dataset_name, y_train=y_train,
                                                               y_test=y_test)
    else:
        stacking_train = np.load(train_save_path)
        print('\nstacking_train loaded')
        print('stacking_train shape:', stacking_train.shape)
        stacking_test = np.load(test_save_path)
        print('\nstacking_test loaded')
        print('stacking_test shape:', stacking_test.shape)

    # 加载真实标签
    if (not os.path.exists(dataset_name + '_stacking_train_labels.npy') or
            not os.path.exists(dataset_name + '_stacking_test_labels.npy')):
        stacking_train_labels, stacking_test_labels = save_labels(X_train, X_test, dataset_name)
    else:
        stacking_train_labels = np.load(dataset_name + '_stacking_train_labels.npy')
        stacking_test_labels = np.load(dataset_name + '_stacking_test_labels.npy')
    # 第一层基模型预测结果作为第二层基模型训练数据，训练元学习器
    # confirm = str(input('是否需要训练元学习器？(y/n)\n'))
    # if confirm == 'y' or confirm == 'yes' or confirm == 'Y' or confirm == 'YES':
    StackingClassifier.meta_train(self=Stacking_Classifier, X_test=X_test, X_train=X_train,
                                  stacking_train=stacking_test, labels=stacking_test_labels,
                                  dataset_name=dataset_name)
    # # 测试模型
    StackingClassifier.partial_predict(self=Stacking_Classifier, X_test=X_test, stacking_test=stacking_test,
                                       labels=stacking_test_labels,
                                       dataset_name=dataset_name)


def data_pre_process(data_directory, png_path='4_Png_16_ISAC', balance: str = None):
    """"
    data_directory: 项目所在的目录
    png_path : 图片存放目录
    balance :是否需要采样（数据平衡）
    返回训练和验证数据加载器（train_loader、test_loader）以及标签(labels_list)、训练集数量、测试集数量和数据集名称（字符串）
    """
    print("*" * 10)
    batch_size = 64
    print(f"Now is training at {png_path}")
    # 确保数据目录存在
    X_data_root = data_directory
    image_path = os.path.normpath(os.path.join(X_data_root, "./pre-processing", png_path))
    if not os.path.exists(image_path):
        try:
            image_path = os.path.normpath(os.path.join(X_data_root, "../pre-processing", png_path))
        finally:
            assert os.path.exists(image_path), f"{image_path} does not exist. Please check the provided path."

    # 目前先设置为占位符，稍后将使用计算出的真实值来替换
    cal_mean_std_flag = 0
    mean = [0.485, 0.456, 0.406]  # 占位符
    std = [0.229, 0.224, 0.225]
    if png_path == "4_Png_16_USTC":
        mean = [0.2221, 0.1979, 0.1993]  # USTC
        std = [0.2910, 0.2859, 0.2792]
        # mean = [0.485, 0.456, 0.406]  # 占位符  For Resnet34
        # std = [0.229, 0.224, 0.225]
    elif png_path == "4_Png_16_CTU":
        mean = [0.2839, 0.2667, 0.2763]  # CTU
        std = [0.2977, 0.2947, 0.2932]
    elif png_path == "4_Png_16_ISAC":
        mean = [0.1842, 0.1811, 0.1761]
        std = [0.2673, 0.2641, 0.2632]
    # 使用计算出的均值和标准差更新数据转换
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(16),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),  # Normalize(mean=tensor([0.1959, 0.2064, 0.1560]), std=tensor([0.2911, 0.3230, 0.2709]))

        "test": transforms.Compose([
            transforms.Resize(16),
            transforms.CenterCrop(16),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    }
    print("*" * 10)
    # 确保数据目录存在
    X_data_root = data_directory
    image_path = os.path.normpath(os.path.join(X_data_root, "./pre-processing", png_path))
    if not os.path.exists(image_path):
        try:
            image_path = os.path.normpath(os.path.join(X_data_root, "../pre-processing", png_path))
            if not os.path.exists(image_path):
                image_path = os.path.normpath(os.path.join(X_data_root, "../../pre-processing", png_path))
        finally:
            assert os.path.exists(image_path), f"{image_path} does not exist. Please check the provided path."
    # 训练数据集
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    # 设置批大小和工作进程数
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    data_name = png_path.split("_")[-1]
    # 写入类别到索引的映射
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open(data_name + 'class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    print("class_indices:", cla_dict)

    # 计算每个类别的样本数量
    class_sample_counts = np.zeros(len(train_dataset.classes), dtype=int)
    for _, label in train_dataset.samples:
        class_sample_counts[label] += 1

    # 计算每个样本的权重
    class_weights = 1.0 / torch.tensor(class_sample_counts, dtype=torch.float)
    sample_weights = class_weights[train_dataset.targets]

    # 创建WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # 创建训练数据加载器，使用sampler而不是shuffle
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=sampler,
                                               num_workers=nw, pin_memory=True, drop_last=True)

    # 测试数据集
    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                        transform=data_transform["test"])
    val_num = len(test_dataset)

    # 收集所有测试数据集的真实标签
    # all_test_labels = [label for _, label in test_dataset]
    # all_train_labels = [label for _, label in train_dataset]

    if exists(f'{data_name}_train_labels.npy'):
        print(f"{data_name}_train_labels.npy exists")
        all_train_labels = np.load(f'{data_name}_train_labels.npy')
    else:
        print(f"{data_name}_train_labels.npy does not exist, saving...")
        all_train_labels = [label for _, label in train_dataset]
        np.save(f'{data_name}_train_labels.npy', all_train_labels)

    if exists(f'{data_name}_test_labels.npy'):
        print(f"{data_name}_test_labels.npy exists")
        all_test_labels = np.load(f'{data_name}_test_labels.npy')
    else:
        print(f"{data_name}_test_labels.npy does not exist, saving...")
        all_test_labels = [label for _, label in test_dataset]
        np.save(f'{data_name}_test_labels.npy', all_test_labels)
    # 创建测试数据加载器
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=nw, pin_memory=True, drop_last=False)

    # 打印类别计数
    unique, counts = np.unique(all_test_labels, return_counts=True)
    print("Test dataset class counts:", dict(zip(unique, counts)))
    print(f"{train_num} for training, {val_num} for testing")
    print("*" * 10)
    # train_loader_balanced, test_loader_balanced = undersample_balance_datasets(train_loader, test_loader)
    # TODO 恶意流量检测时，正常流量样本（190,127）明显多于恶意流量样本（总共161,331个，分为10个类别）
    try:
        if not balance:
            return train_loader, test_loader, all_test_labels, all_train_labels, data_name
    except ValueError as e:
        print(f"{e}.Invalid balance type. Please choose 'oversample' or 'undersample'.")


def frontend_main(png_path):
    base_classifiers = [
        'lstm_model',
        'BiTCN_model',
        'cnn_model',
        'BiLSTM_model',
        'tcn_model',
        # 'rnn_model',
    ]
    # 选择不同的组合器，例如使用LSTM作为组合器
    combiner = 'BiTCN_model'

    # 数据预处理
    root = os.getcwd()
    X_train, X_test, y_test, y_train, dataset_name = data_pre_process(root, png_path, None)

    Stacking_Classifier = StackingClassifier(base_classifiers, combiner)

    train_save_path = './' + dataset_name + '_stacking_train.npy'
    test_save_path = './' + dataset_name + '_stacking_test.npy'
    print('Train saved to: ', train_save_path)
    print('Test saved to: ', test_save_path)
    # 加载训练集
    if not os.path.exists(train_save_path):
        print(f'generating stacking_train……')
        print('train_save_path is :', train_save_path)
        print('test_save_path is :', test_save_path)
        stacking_train, stacking_test = Stacking_Classifier.fit(X_train=X_train, X_test=X_test,
                                                                dataset_name=dataset_name, y_train=y_train,
                                                                y_test=y_test)
    else:
        stacking_train = np.load(train_save_path)
        print('\nstacking_train loaded')
        print('stacking_train shape:', stacking_train.shape)
        stacking_test = np.load(test_save_path)
        print('\nstacking_test loaded')
        print('stacking_test shape:', stacking_test.shape)

    # 加载真实标签
    if (not os.path.exists(dataset_name + '_stacking_train_labels.npy') or
            not os.path.exists(dataset_name + '_stacking_test_labels.npy')):
        stacking_train_labels, stacking_test_labels = save_labels(X_train, X_test, dataset_name)
    else:
        stacking_train_labels = np.load(dataset_name + '_stacking_train_labels.npy')
        stacking_test_labels = np.load(dataset_name + '_stacking_test_labels.npy')
    # 第一层基模型预测结果作为第二层基模型训练数据，训练元学习器
    # StackingClassifier.meta_train(self=Stacking_Classifier,X_test=X_test, X_train=X_train,
    #                               stacking_train=stacking_test, labels=stacking_test_labels,
    #                               dataset_name=dataset_name)

    # 测试模型
    predictions = StackingClassifier.partial_predict(self=Stacking_Classifier,
                                                     X_test=X_test, stacking_test=stacking_test,
                                                     labels=stacking_test_labels,
                                                     dataset_name=dataset_name)

    correct_predictions = sum(1 for true, pred in zip(y_test, predictions) if true == pred)
    # total_predictions = len(y_test)
    # accuracy = correct_predictions / total_predictions

    # cm = confusion_matrix(y_test, predictions)
    # report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    # f1_score = report['weighted avg']['f1-score']
    #
    # result_text = (
    #     f"Accuracy: {accuracy:.6f}\n"
    #     f"F1 Score: {f1_score:.6f}\n"
    #     f"Confusion Matrix:\n{cm}\n"
    #     f"Classification Report:\n{classification_report(y_test, predictions, zero_division=0)}"
    # )
    #
    # print(result_text)


if __name__ == '__main__':
    main()
