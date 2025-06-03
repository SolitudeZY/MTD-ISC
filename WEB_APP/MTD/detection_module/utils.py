import json
import os
from array import array
from multiprocessing import Pool, Manager
from typing import Union

from sklearn.metrics import confusion_matrix
from torch.utils.data import WeightedRandomSampler, SubsetRandomSampler
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from tqdm import tqdm

import pandas as pd


def data_pre_process(data_directory, png_path='4_Png_16_USTC', balance: str = None):
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

    else:
        cal_mean_std_flag = 1  # 计算均值和方差

    # 未经归一化的数据转换
    unnormalized_transform = transforms.Compose([
        transforms.Resize(16),
        transforms.CenterCrop(16),
        transforms.ToTensor(),
    ])
    # 创建一个预处理的数据加载器，用于计算均值和标准差
    pre_process_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                               transform=unnormalized_transform)
    pre_process_loader = DataLoader(pre_process_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 计算均值和标准差
    if cal_mean_std_flag:
        print("Calculating mean and std...")
        mean, std = calculate_mean_and_std(pre_process_loader)
    print("Computed Mean:", mean)
    print("Computed Std:", std)

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
    # print(data_transform)
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

    data_name = png_path.split("_")[-1] + '_'
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
    all_test_labels = [label for _, label in test_dataset]
    # all_train_labels = [label for _, label in train_dataset]

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
            return train_loader, test_loader, all_test_labels, train_num, val_num, data_name

        # 过采样数据集
        elif balance == 'oversample':
            train_loader_balanced, test_loader_balanced = oversample_balance_datasets(train_loader, test_loader)
            balanced_train_class_counts = count_classes_in_loader(train_loader_balanced)
            balanced_test_class_counts = count_classes_in_loader(test_loader_balanced)
            print(
                f"Balanced train class counts: {balanced_train_class_counts}, \n"
                f"Balanced test class counts: {balanced_test_class_counts}")
            train_num = sum(balanced_train_class_counts.values())
            val_num = sum(balanced_test_class_counts.values())
            return train_loader_balanced, test_loader_balanced, all_test_labels, train_num, val_num, data_name
        # 欠采样数据集
        elif balance == 'undersample':
            train_loader_balanced, test_loader_balanced = undersample_balance_datasets(train_loader, test_loader)
            balanced_train_class_counts = count_classes_in_loader(train_loader_balanced)
            balanced_test_class_counts = count_classes_in_loader(test_loader_balanced)
            print(
                f"Balanced train class counts: {balanced_train_class_counts}, \n"
                f"Balanced test class counts: {balanced_test_class_counts}")
            train_num = sum(balanced_train_class_counts.values())
            val_num = sum(balanced_test_class_counts.values())
            return train_loader_balanced, test_loader_balanced, all_test_labels, train_num, val_num, data_name

    except ValueError as e:
        print(f"{e}.Invalid balance type. Please choose 'oversample' or 'undersample'.")

    # 返回训练和验证数据加载器以及类别到索引的映射
    # return train_loader_balanced, test_loader_balanced, all_test_labels


# def data_pre_process(data_directory, png_path='4_Png_16_CIC'):
#     """
#         返回训练 dataloader和测试 dataloader 以及对应的标签labels
#         这里的参数是（来自ImageNet)
#          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         备选使用（此数据来着原来的文件）：
#          transforms.Normalize (mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
#         return : train_dataloader, test_dataloader, labels
#     """
#     # 数据增强
#     # 数据预处理配置
#     data_transform = {
#         "train": transforms.Compose([
#             transforms.RandomResizedCrop(16),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         "test": transforms.Compose([
#             transforms.Resize(16),
#             transforms.CenterCrop(16),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#     }
#     print(data_transform)
#
#     X_data_root = data_directory
#     image_path = os.path.normpath(os.path.join(X_data_root, "pre-processing", png_path))
#     assert os.path.exists(image_path), f"{image_path} does not exist. Please check the provided path."
#
#     # 训练数据集
#     train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
#                                          transform=data_transform["train"])
#     train_num = len(train_dataset)
#
#     # 设置批大小和工作进程数
#     batch_size = 64
#     nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
#     print(f'Using {nw} dataloader workers every process')
#
#     # 创建训练数据加载器
#     train_loader = torch.utils.data.DataLoader(train_dataset,
#                                                batch_size=batch_size, shuffle=True,
#                                                num_workers=nw, pin_memory=True, drop_last=True)
#
#     # 测试数据集
#     test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
#                                         transform=data_transform["test"])
#     val_num = len(test_dataset)
#     print(f"using {train_num} images for training, {val_num} images for validation.")
#
#     # 收集所有测试数据集的真实标签
#     all_test_labels = [label for _, label in test_dataset]
#
#     # 创建测试数据加载器
#     test_loader = torch.utils.data.DataLoader(test_dataset,
#                                               batch_size=batch_size, shuffle=False,
#                                               num_workers=nw, pin_memory=True, drop_last=False)
#
#     # 打印类别计数，验证是否平衡
#     unique, counts = np.unique(all_test_labels, return_counts=True)
#     print(dict(zip(unique, counts)))
#
#     # 返回训练和验证数据加载器以及类别到索引的映射
#     return train_loader, test_loader, all_test_labels

def count_classes_in_loader(data_loader, num_processes=4):
    """
    统计数据加载器中的每个类别的样本数量，并使用多进程提高效率
    """
    print("calculating class counts...")

    # 定义一个局部函数，用于单个进程处理数据
    def count_classes(sub_loader, result_dict):
        local_class_counts = {}
        for _, labels in sub_loader:
            for label in labels:
                label = label.item()
                if label in local_class_counts:
                    local_class_counts[label] += 1
                else:
                    local_class_counts[label] = 1
        result_dict.update(local_class_counts)

    # 分割数据加载器
    dataset = data_loader.dataset
    sampler = data_loader.sampler
    batch_size = data_loader.batch_size
    num_batches = len(data_loader)

    # 创建子数据加载器
    sub_loaders = []
    batch_indices = list(range(num_batches))
    chunk_size = num_batches // num_processes
    for i in range(num_processes):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_processes - 1 else num_batches
        sub_sampler = torch.utils.data.SubsetRandomSampler(batch_indices[start:end])
        sub_loader = DataLoader(dataset, batch_size=batch_size, sampler=sub_sampler, num_workers=0)
        sub_loaders.append(sub_loader)

    # 使用多进程处理
    manager = Manager()
    result_dict = manager.dict()

    with Pool(processes=num_processes) as pool:
        pool.starmap(count_classes, [(sub_loader, result_dict) for sub_loader in sub_loaders])

    # 合并结果
    class_counts = dict(result_dict)
    return class_counts


def undersample_balance_datasets(train_loader, test_loader):  # 使用下采样
    print("undersample Balancing")
    # 从数据加载器中获取数据集
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset

    # 获取标签列表
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    test_labels = [test_dataset[i][1] for i in range(len(test_dataset))]

    # 统计每个类别的样本数量
    label_counts_train = {label: train_labels.count(label) for label in set(train_labels)}
    label_counts_test = {label: test_labels.count(label) for label in set(test_labels)}

    # 找到最小的类别样本数
    min_samples_train = min(label_counts_train.values())
    min_samples_test = min(label_counts_test.values())

    # 下采样
    balanced_train_indices = []
    balanced_test_indices = []
    for label in tqdm(set(train_labels), desc="Balancing dataset"):
        indices_train = [i for i, x in enumerate(train_labels) if x == label]
        indices_test = [i for i, x in enumerate(test_labels) if x == label]

        # 随机选择与最小类别相同的数量的样本索引
        balanced_train_indices.extend(np.random.choice(indices_train, min_samples_train, replace=False))
        balanced_test_indices.extend(np.random.choice(indices_test, min_samples_test, replace=False))

    # 创建新的数据加载器
    balanced_train_sampler = SubsetRandomSampler(balanced_train_indices)
    balanced_test_sampler = SubsetRandomSampler(balanced_test_indices)

    balanced_train_loader = DataLoader(train_dataset, batch_size=train_loader.batch_size,
                                       sampler=balanced_train_sampler)
    balanced_test_loader = DataLoader(test_dataset, batch_size=test_loader.batch_size, sampler=balanced_test_sampler)

    return balanced_train_loader, balanced_test_loader


def oversample_balance_datasets(train_loader, test_loader):
    print("Oversample Balancing")
    # 从数据加载器中获取数据集
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset

    # 获取标签列表
    train_labels = [train_dataset[i][1] for i in tqdm(range(len(train_dataset)), desc="获取train_labels标签列表")]
    test_labels = [test_dataset[i][1] for i in tqdm(range(len(test_dataset)), desc='获取test_labels标签列表')]

    # 统计每个类别的样本数量
    label_counts_train = {label: train_labels.count(label) for label in
                          tqdm(set(train_labels), desc='统计train每个类别的样本数量')}
    label_counts_test = {label: test_labels.count(label) for label in
                         tqdm(set(test_labels), desc='统计test每个类别的样本数量')}

    # 找到最大的类别样本数
    max_samples_train = max(label_counts_train.values())
    max_samples_test = max(label_counts_test.values())

    # 过采样
    balanced_train_indices = []
    balanced_test_indices = []

    # 对每个类别进行过采样
    for label in tqdm(set(train_labels), desc="Balancing dataset"):
        indices_train = [i for i, x in enumerate(train_labels) if x == label]
        indices_test = [i for i, x in enumerate(test_labels) if x == label]

        # 随机选择与最大类别相同的数量的样本索引
        balanced_train_indices.extend(np.random.choice(indices_train, max_samples_train, replace=True))
        balanced_test_indices.extend(np.random.choice(indices_test, max_samples_test, replace=True))

    # 创建新的数据加载器
    balanced_train_sampler = SubsetRandomSampler(balanced_train_indices)
    balanced_test_sampler = SubsetRandomSampler(balanced_test_indices)

    balanced_train_loader = DataLoader(train_dataset, batch_size=train_loader.batch_size,
                                       sampler=balanced_train_sampler)
    balanced_test_loader = DataLoader(test_dataset, batch_size=test_loader.batch_size,
                                      sampler=balanced_test_sampler)

    return balanced_train_loader, balanced_test_loader


def calculate_mean_and_std(loader):
    # Var[x] = E[X^2] - E^2[X]
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader, desc="Calculating mean and std of images", total=len(loader)):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


def save_model(model, path='/models', name='model.pt', model_state_dict=False):
    """
    Save the model
    默认值为： path='/models', name='model.pt', model_state_dict=False
    分别对应保存路径、保存文件名和是否保存为字典
    """
    # 指定保存模型的目录
    save_dir = path
    # 创建目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    # 构建完整的文件路径
    # save_filename = os.path.join(save_dir, 'lstm-model.pt')
    save_filename = os.path.join(save_dir, name)
    # 保存模型
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)
    if model_state_dict:
        torch.save(model.state_dict(), save_filename)
        print('Save model state_dict as ', save_filename)


def calculate_tpr_fpr_multiclass(y_true, y_pred, n_classes):
    """
    计算多分类问题的TPR和FPR

    :param y_true: 真实标签，numpy数组或列表
    :param y_pred: 预测标签，numpy数组或列表
    :param n_classes: 类别数量
    :return: 加权平均的TPR和FPR
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    tpr_list = []
    fpr_list = []
    support_list = []
    weighted_tpr = 0.0
    weighted_fpr = 0.0
    for i in range(n_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = np.sum(cm) - (TP + FP + FN)

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

        tpr_list.append(TPR)
        fpr_list.append(FPR)
        support_list.append(TP + FN)  # 支持度，即每个类别的样本数

    # 计算加权平均的TPR和FPR
    total_support = sum(support_list)
    weighted_tpr = sum(tpr * support for tpr, support in zip(tpr_list, support_list)) / total_support
    weighted_fpr = sum(fpr * support for fpr, support in zip(fpr_list, support_list)) / total_support

    return weighted_tpr, weighted_fpr


def calculate_tpr_fpr(json_filepath: str, true_labels, pred_labels):
    """
    计算总体TPR和FPR，使用json文件中的'Normal'类别索引
    Returns TPR, FPR
    """
    print("Calculating TPR and FPR")
    # 从json文件中加载类别索引
    with open(json_filepath, 'r') as json_file:
        cla_dict = json.load(json_file)
        # 找到值为 "Normal" 的索引
        normal_index = None
        for idx, class_name in cla_dict.items():
            if class_name == "Normal":
                normal_index = int(idx)
                # print(f"Found 'Normal' class at index: {idx}")
                break
        if normal_index is None:
            raise ValueError("Class 'Normal' not found in the JSON file.")
    # 计算混淆矩阵

    # 获取类别数量
    num_classes = len(cla_dict)

    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes)))
    # cm = confusion_matrix(true_labels, pred_labels)

    # 真正例（TP）是恶意流量被正确预测的总和（除了'Normal'索引的行和列）
    TP = np.sum(cm) - np.sum(cm[normal_index, :]) - np.sum(cm[:, normal_index]) + cm[normal_index, normal_index]
    # 假负例（FN）是恶意流量错误预测为'Normal'的数量
    FN = cm[:, normal_index].sum() - cm[normal_index, normal_index]
    # 假正例（FP）是'Normal'错误预测为恶意流量的数量
    FP = cm[normal_index, :].sum() - cm[normal_index, normal_index]
    # 真负例（TN）是'Normal'被正确预测的数量
    TN = cm[normal_index, normal_index]

    # 计算总体TPR和FPR
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}")
    return TPR, FPR


def get_class_nums(Png_path: str = "4_Png_16_USTC") -> int:
    try:
        directory_path = os.path.join(os.path.abspath(os.getcwd()), "./pre-processing/" + Png_path + "/Train")
        entries = os.listdir(directory_path)
    except FileNotFoundError:
        directory_path = os.path.join(os.path.abspath(os.getcwd()), "../pre-processing/" + Png_path + "/Train")
        entries = os.listdir(directory_path)
    # 过滤出文件夹并计数
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    return folder_count


def get_class_distribution(data_directory, png_path='4_Png_16_USTC'):
    """
    获取数据集中每个类别及其对应的实例数量。

    参数:
    - data_directory: 数据集所在的根目录。
    - png_path: 图片存放目录。

    返回:
    - 训练集类别名称到实例数量的字典。
    - 测试集类别名称到实例数量的字典。
    """
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
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"))

    # 测试数据集
    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"))

    # 获取类别到索引的映射
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # 计算每个类别的样本数量
    train_class_sample_counts = np.zeros(len(train_dataset.classes), dtype=int)
    for _, label in train_dataset.samples:
        train_class_sample_counts[label] += 1

    test_class_sample_counts = np.zeros(len(test_dataset.classes), dtype=int)
    for _, label in test_dataset.samples:
        test_class_sample_counts[label] += 1

    # 创建类别名称到实例数量的字典
    train_class_distribution = {idx_to_class[idx]: count for idx, count in enumerate(train_class_sample_counts)}
    test_class_distribution = {idx_to_class[idx]: count for idx, count in enumerate(test_class_sample_counts)}

    return train_class_distribution, test_class_distribution


def print_class_distribution(train_distribution, test_distribution, output_csv='class_distribution.csv'):
    """
    打印训练集和测试集的类别分布，并将其保存为CSV文件。

    参数:
    - train_distribution: 训练集类别名称到实例数量的字典。
    - test_distribution: 测试集类别名称到实例数量的字典。
    - output_csv: 输出CSV文件的路径。
    """
    # 将字典转换为DataFrame
    df = pd.DataFrame({
        'Class Name': list(train_distribution.keys()),
        'Train Count': list(train_distribution.values()),
        'Test Count': [test_distribution.get(cls, 0) for cls in train_distribution.keys()]
    })

    # 打印DataFrame
    print(df)

    # 将DataFrame保存为CSV文件
    df.to_csv(output_csv, index=False)
    print(f"Class distribution saved to {output_csv}")


if __name__ == "__main__":
    # 示例调用
    data_directory = os.getcwd()
    png_path = '4_Png_16_ISAC'  # 4_Png_16_CTU
    train_distribution, test_distribution = get_class_distribution(data_directory, png_path)
    print_class_distribution(train_distribution, test_distribution)
