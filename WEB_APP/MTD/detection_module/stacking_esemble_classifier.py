import os
from os.path import exists
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset
import BiLSTM_predict
import BiTCN_pred_temp
import BiTCN_predict
import CNN_predict
import EfficientNet_predict
import LSTM_predict
import LSTM_predicts
import RNN_predict
import ResNet_predict
import TCN_predicts
import resnet_predict
import tcn_train_and_test
from FocalLoss import FocalLoss
from utils import save_model, get_class_nums
from Resnet_new.resnet_model import resnet34

png_path = '4_Png_16_USTC'
class_numbers = get_class_nums(Png_path=png_path)
batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "")


def get_path():
    return png_path


def dataloader_to_numpy(dataloader):
    """ 将DataLoader对象转换为numpy数组 """
    X, y = [], []
    for batch in dataloader:
        inputs, targets = batch
        X.append(inputs.numpy())
        y.append(targets.numpy())
    return np.vstack(X), np.hstack(y)


def save_labels(X_train, X_test, dataset_name):
    """
    由于K折交叉验证会导致训练集和测试集数据量和原来不一致，故加此函数保证标签的数据量问题
    """
    print("Saving labels...")
    _, y_train_np = dataloader_to_numpy(X_train)
    _, y_test_np = dataloader_to_numpy(X_test)

    print("y_train shape ", y_train_np.shape)
    print("y_test shape", y_test_np.shape)
    # 提取 stacking_train 对应的标签
    stacking_train_labels = y_train_np
    stacking_train_labels_save_path = f'./{dataset_name}_stacking_train_labels.npy'
    np.save(stacking_train_labels_save_path, stacking_train_labels)
    print(f"Stacking train labels saved to: {stacking_train_labels_save_path}")

    # 保存 stacking_test 对应的标签
    stacking_test_labels = y_test_np
    stacking_test_labels_save_path = f'./{dataset_name}_stacking_test_labels.npy'
    np.save(stacking_test_labels_save_path, stacking_test_labels)
    print(f"Stacking test labels saved to: {stacking_test_labels_save_path}")

    print(f"stacking_train_labels.shape: {stacking_train_labels.shape}")
    print(f"stacking_test_labels.shape: {stacking_test_labels.shape}")

    return stacking_train_labels, stacking_test_labels


class StackingClassifier:
    def __init__(self, base_classifiers, combiner):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_classifiers = base_classifiers
        self.combiner = combiner

    def fit(self, X_train, y_train, dataset_name, X_test, y_test):
        """
        训练堆叠模型。

        参数:
        - X_train: 训练数据集的特征部分，即训练数据，numpy数组。
        - y_train: 训练数据集的目标变量，即真实标签，numpy数组。
        - dataset_name: 数据集名称，用于保存中间结果。
        - X_test: 测试数据集的特征部分，numpy数组。
        - y_test: 测试数据集的目标变量，numpy数组。

        说明:
        此函数遍历所有基础分类器，并使用整个训练集对每个基础分类器进行训练，然后在验证集上进行验证。预测结果被存储在一个二维数组中，
        该数组的每一列对应一个基础分类器的预测结果。最后，将这些预测结果作为输入，使用组合器模型
        （combiner）进行进一步的训练。
        """

        global train_pred, test_pred

        if (not os.path.exists(dataset_name + '_stacking_train_labels.npy') or
                not os.path.exists(dataset_name + '_stacking_test_labels.npy')):
            print(f"{dataset_name}_stacking_train_labels.npy does not exist"
                  f"or {dataset_name}_stacking_test_labels.npy does not exist")
            save_labels(X_train, X_test, dataset_name)
        else:
            stacking_train_labels = np.load(dataset_name + '_stacking_train_labels.npy')
            stacking_test_labels = np.load(dataset_name + '_stacking_test_labels.npy')

        # 假设 X_train 和 y_train 是 DataLoader 对象
        if isinstance(X_train, DataLoader):
            X_train_np, y_train_np = dataloader_to_numpy(X_train)
            X_test_np, y_test_np = dataloader_to_numpy(X_test)
        else:
            X_train_np, y_train_np = X_train, y_train
            X_test_np, y_test_np = X_test, y_test

        stacking_train = np.zeros((len(y_train_np), len(self.base_classifiers)))
        stacking_test = np.zeros((len(y_test_np), len(self.base_classifiers)))

        print('stacking_train shape:', stacking_train.shape)
        print('stacking_test shape:', stacking_test.shape)

        # 初始化多维列表来存储预测结果
        train_preds = []
        test_preds = []

        for model_no, classifier in enumerate(self.base_classifiers):
            print(f"Now is using {classifier} as a base_model")

            # 将 X_train 和 y_train 转换为 DataLoader
            train_dataset = TensorDataset(torch.tensor(X_train_np, dtype=torch.float32),
                                          torch.tensor(y_train_np, dtype=torch.long))
            print("train_dataset shape:", train_dataset.tensors[0].shape)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            test_dataset = TensorDataset(torch.tensor(X_test_np, dtype=torch.float32),
                                         torch.tensor(y_test_np, dtype=torch.long))
            print("test_dataset shape:", test_dataset.tensors[0].shape)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            if classifier == 'lstm_model':
                print("\nNow is using LSTM_model as a base_model")
                if not exists(f'./models/{dataset_name}/LSTM_for_meta.pt'):
                    print(f"now is training LSTM for meta")
                    lstm_model = LSTM_predicts.train_model(num_epoches=10, train_loader=train_loader,
                                                           class_nums=class_numbers)
                    torch.save(lstm_model, f'./models/{dataset_name}/LSTM_for_meta.pt')
                else:
                    lstm_model = torch.load(f'./models/{dataset_name}/LSTM_for_meta.pt')
                train_pred = LSTM_predicts.predict(model=lstm_model, test_loader=train_loader, device=self.device,
                                                   class_num=class_numbers)[0]
                test_pred = LSTM_predicts.predict(model=lstm_model, test_loader=test_loader, device=self.device,
                                                  class_num=class_numbers)[0]

            elif classifier == 'cnn_model':
                print("\nNow is using CNN as a base_model")
                model_path = f'./models/{dataset_name}/CNN_for_meta.pt'
                if not exists(model_path):
                    print(f"now is training CNN for meta")
                    model = CNN_predict.train_model(train_loader, epochs=10, class_nums=class_numbers,
                                                    device=self.device)
                    torch.save(model, model_path)
                else:
                    model = torch.load(model_path)
                print("\ntrain predictions")
                train_pred = \
                    CNN_predict.predict(model=model, test_loader=train_loader, class_num=class_numbers,
                                        device=self.device,
                                        dataset_name=dataset_name)['predictions']
                print("\ntest predictions")
                test_pred = CNN_predict.predict(model=model, test_loader=test_loader, class_num=class_numbers,
                                                device=self.device,
                                                dataset_name=dataset_name)['predictions']

            elif classifier == 'tcn_model':
                print("\nNow is using TCN as a base_model")
                model_path = f'./models/{dataset_name}/TCN_for_meta.pt'
                if not exists(model_path):
                    tcn_model_temp = tcn_train_and_test.get_model(class_numbers)
                    tcn_model = TCN_predicts.train_model(class_nums=class_numbers, train_loader=train_loader,
                                                         model=tcn_model_temp,
                                                         device=self.device,
                                                         epochs=5)
                    torch.save(tcn_model, model_path)
                else:
                    tcn_model = torch.load(f'./models/{dataset_name}/TCN_for_meta.pt', map_location=self.device)
                print("\ntrain predictions")
                train_pred = TCN_predicts.test(tcn_model, train_loader, class_numbers)[0]
                print("\ntest predictions")
                test_pred = TCN_predicts.test(tcn_model, test_loader, class_numbers)[0]

            elif classifier == 'BiTCN_model':
                print("\nNow is using BiTCN as a base_model")
                model_path = f'./models/{dataset_name}/BiTCN_for_meta.pt'
                # mdoel_path = 'D:/Python Project/Deep-Traffic/models/USTC/BiTCN_for_meta.pt'
                if not exists(model_path):
                    BiTCN_model = BiTCN_pred_temp.meta_BiTCN(in_channels=3, num_classes=class_numbers,
                                                             num_channels=8 * [25], dropout=0.05,
                                                             kernel_size=3)
                    epochs = 8
                    optimizer = optim.Adam(BiTCN_model.parameters(), lr=0.0015)
                    criterion = nn.CrossEntropyLoss()
                    for i in range(epochs):
                        BiTCN_model = BiTCN_pred_temp.train(BiTCN_model, train_loader, criterion=criterion,
                                                            optimizer=optimizer,
                                                            device=device,
                                                            num_class=class_numbers)
                        BiTCN_pred_temp.validate(BiTCN_model, train_loader, criterion=criterion, device=device,
                                                 num_class=class_numbers, dataset_name=dataset_name)
                        torch.save(BiTCN_model, model_path)
                        print(f"Saved BiTCN model at {model_path}")
                else:
                    BiTCN_model = torch.load(model_path, map_location=self.device)
                print("\ntrain predictions")
                train_pred = BiTCN_pred_temp.predict(BiTCN_model, train_loader, device=device,
                                                     dataset_name=dataset_name)
                print("\ntest predictions")
                test_pred = BiTCN_pred_temp.predict(BiTCN_model, test_loader, device=self.device,
                                                    dataset_name=dataset_name)

            elif classifier == 'BiLSTM_model':
                print("\nNow is using BiLSTM as a base_model")
                model_path = f"./models/{dataset_name}/BiLSTM_for_meta.pt"
                print("\ntrain predictions")
                if exists(model_path):
                    BiLSTM_model = torch.load(model_path, map_location=self.device)
                else:  # TODO 记得在训练函数中保存模型文件
                    BiLSTM_model = BiLSTM_predict.train_model(train_loader, class_nums=class_numbers,
                                                              device=self.device,
                                                              num_epochs=8)
                    torch.save(BiLSTM_model, model_path)
                train_pred = \
                    BiLSTM_predict.predict(BiLSTM_model, data_loader=train_loader, DEVICE=BiLSTM_predict.DEVICE,
                                           num_classes=class_numbers,
                                           dataset_name=dataset_name)[0]
                print("\ntest predictions")
                test_pred = BiLSTM_predict.predict(BiLSTM_model, test_loader, BiLSTM_predict.DEVICE,
                                                   num_classes=class_numbers,
                                                   dataset_name=dataset_name)[0]

            # 将预测结果添加到多维列表中
            train_preds.append(train_pred)
            test_preds.append(test_pred)

            # 保存每个模型的预测结果
            predictions_dir = f'./{dataset_name}_predictions/{classifier}/'
            os.makedirs(predictions_dir, exist_ok=True)
            train_predictions_filename = f'{predictions_dir}train.npy'
            test_predictions_filename = f'{predictions_dir}test.npy'
            np.save(train_predictions_filename, train_pred)
            np.save(test_predictions_filename, test_pred)

            # 调试信息
            print(f"{classifier} train np.array predictions shape: {np.array(train_pred).shape}")
            print(f"{classifier} test predictions shape: {np.array(test_pred).shape}")

        # 将多维列表转换为 NumPy 数组
        stacking_train = np.column_stack(train_preds)
        stacking_test = np.column_stack(test_preds)

        # 定义保存路径
        train_save_path = f'./{dataset_name}_stacking_train.npy'
        test_save_path = f'./{dataset_name}_stacking_test.npy'
        np.save(train_save_path, stacking_train)
        np.save(test_save_path, stacking_test)

        print("Stacking train and test predictions saved.")

        return stacking_train, stacking_test

    def meta_train(self, X_train, X_test, stacking_train, labels, dataset_name=None):
        # TODO 获得训练样本数和分类器数目
        n_samples, n_classifiers = len(stacking_train), len(self.base_classifiers)
        print("n_samples:", n_samples, "n_classifiers:", n_classifiers)

        if self.combiner == 'lstm_model':  # LSTM的期望输入为
            print('*' * 10)
            print("Now is training LSTM_model model as meta learner")
            # 对stacking_train进行处理，使其成为32维数据以匹配第二层模型的输入
            # TODO 重塑 stacking_train 并 新建一个LSTM_meta模型以适配输入
            stacking_train_reshaped = stacking_train.reshape(n_samples, 1, n_classifiers)
            print("stacking_train_reshaped shape is :", stacking_train_reshaped.shape)
            lstm_num_classes = class_numbers
            LSTM_predict.fit(lstm_num_classes, train_loader=X_train, stacking_train=stacking_train_reshaped,
                             labels=labels, epochs=3, input_size=len(self.base_classifiers),  # USTC epochs=3
                             dataset_name=dataset_name,
                             Png_flag=False)

        elif self.combiner == 'rnn_model':
            print('*' * 10)
            print("Now is training RNN model as meta learner")
            # 重构 stacking_train
            stacking_train_reshaped = stacking_train.reshape(n_samples, 1, n_classifiers)
            print("stacking_train_reshaped shape is :", stacking_train_reshaped.shape)
            rnn_num_classes = class_numbers  # USTC epochs=4, CTU epochs=3
            RNN_predict.fit(rnn_num_classes, stacking_train=stacking_train_reshaped, labels=labels, epochs=4,
                            input_size=len(self.base_classifiers), Device=device, dataset_name=dataset_name)

        elif self.combiner == 'BiLSTM_model':
            print('*' * 10)
            print("Now is training BiLSTM_model as meta learner")
            stacking_train_reshaped = stacking_train.reshape(n_samples, 1, n_classifiers)
            Bilstm_num_classes = class_numbers
            BiLSTM_predict.fit(Bilstm_num_classes, stacking_train=stacking_train_reshaped, labels=labels,
                               epochs=3, input_size=len(self.base_classifiers),
                               Device=device, dataset_name=dataset_name)
            # save_model(model=BiLSTM_model, path='./models', name='meta_BiLSTM_model.pt', model_state_dict=False)

        elif self.combiner == 'cnn_model':
            print('*' * 10)
            print("Now is training CNN_model as meta learner")
            CNN_predict.fit(stacking_train=stacking_train, labels=labels, epochs=4,
                            input_size=len(self.base_classifiers), dataset_name=dataset_name,
                            num_classes=class_numbers)

        elif self.combiner == 'tcn_model':  # expected input[1, 128, 6] to have 6 channels
            # TCN 模型的输入形状应该是 (N, C_in, L_in)，即 (batch_size, input_size, sequence_length)。
            print('*' * 10)
            print("Now is training TCN_model as meta learner")
            stacking_train_reshaped = stacking_train.reshape(stacking_train.shape[0], stacking_train.shape[1], 1)
            print("stacking_train shape ", stacking_train_reshaped.shape)
            print("stacking_train labels shape ", labels.shape)
            # 调整形状
            TCN_predicts.fit(stacking_train=stacking_train_reshaped, labels=labels, epochs=2, dataset_name=dataset_name,
                             input_size=len(self.base_classifiers),
                             num_classes=class_numbers)

        elif self.combiner == 'BiTCN_model':  # 同BiTCN
            print('*' * 10)
            print("Now is training BiTCN_model as meta learner")
            num_samples = stacking_train.shape[0]  # 样本数
            num_features = stacking_train.shape[1]  # 特征数
            # # reshaped_data = stacking_train.reshape((num_samples, 16, 16, 1))  # 变为单通道图像
            # # 转换为 RGB 图像 (复制三次通道)
            # # rgb_data = np.repeat(reshaped_data, 3, axis=-1)  # 将单通道扩展为三通道
            # required_features = 16 * 16  # 256
            # # 如果特征数不足 256，我们选择填充（这里使用0填充）
            # if num_features < required_features:
            #     padding = np.zeros((num_samples, required_features - num_features))
            #     stacking_train_padded = np.hstack((stacking_train, padding))
            # elif num_features > required_features:
            #     # 截取前 256 个特征（对于超过256的情况，但这在当前既定数量下不会发生）
            #     stacking_train_padded = stacking_train[:, :required_features]
            # else:
            #     stacking_train_padded = stacking_train
            # # 重塑为 (39042, 16, 16, 1)
            # reshaped_data = stacking_train_padded.reshape((num_samples, 16, 16, 1))
            # # 将单通道扩展为 RGB 三通道
            # rgb_data = np.repeat(reshaped_data, 3, axis=-1)
            # rgb_data = rgb_data.reshape((39042, 16 * 16, 3))  # (39042, 256, 3)
            # rgb_data = np.transpose(rgb_data, (0, 2, 1))  # 交换维度 -> (39042, 3, 256)
            stacking_train_reshaped = stacking_train.reshape(stacking_train.shape[0], stacking_train.shape[1], 1)
            BiTCN_pred_temp.fit_bitcn(stacking_train=stacking_train_reshaped, labels=labels, epochs=30,
                                      input_size=len(self.base_classifiers),
                                      # input_size=3,
                                      num_classes=class_numbers, dataset_name=dataset_name)
            # save_model(bitcn_meta, path='./models', name='meta_BiTCN_model.pt', model_state_dict=False)

        elif self.combiner == 'ResNet_model':
            print('*' * 10)
            print("Now is training resnet_model as meta learner")
            # resnet_model_path = 'models/USTC/Resnet34_3.pth'
            # batch_size, channels, height, weight
            reshaped_train = stacking_train.reshape(n_samples, n_classifiers, 1, 1)
            ResNet_predict.fit(stacking_train=reshaped_train, labels=labels, epochs=1, dataset_name=dataset_name,
                               device=device, num_classes=class_numbers)
            # save_model(ResNet_meta, path='./models', name='meta_ResNet_model.pt', model_state_dict=False)

        elif self.combiner == 'EfficientNet_model':
            print('*' * 10)
            print("Now is training EfficientNet_model as meta learner")
            EfficientNet_model_path = './models/EfficientNet_model.pth'

            # batch_size, channels, height, weight，同ResNet
            reshaped_train = stacking_train.reshape(n_samples, n_classifiers, 1, 1)
            EfficientNet_predict.fit(stacking_train=reshaped_train, labels=labels,
                                     epochs=2, dataset_name=dataset_name,
                                     device=device, num_classes=class_numbers
                                     )

    def partial_predict(self, X_test, stacking_test, labels, dataset_name=None):
        n_samples, n_classifiers = len(X_test.dataset), len(self.base_classifiers)

        stacking_train_tensor = torch.tensor(stacking_test, dtype=torch.float32).to(device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
        predictions = []
        # base_model_path = './models/USTC/'
        # TODO 深度学习模型的训练可以参照上面fit函数中的LSTM_predict文件
        if self.combiner == 'lstm_model':  # LSTM的期望输入为（
            print('*' * 10)
            print("Now is predicting LSTM_model model as meta learner")
            # 对stacking_train进行处理，使其成为32维数据以匹配第二层模型的输入
            seq_len = len(self.base_classifiers)  # 存疑 len(X_test.dataset)   # 序列长度为数组的列数，即每一排的长度
            # TODO 重塑 stacking_train 并 新建一个LSTM_meta模型以适配输入
            stacking_train_reshaped = stacking_test.reshape(n_samples, 1, n_classifiers)
            print("stacking_train_reshaped shape is :", stacking_train_reshaped.shape)
            predictions = LSTM_predict.meta_predict(stacking_train=stacking_train_reshaped, labels=labels,
                                                    device=device,
                                                    num_classes=class_numbers, dataset_name=dataset_name,
                                                    input_size=len(self.base_classifiers))

        elif self.combiner == 'rnn_model':
            print('*' * 10)
            print("Now is predicting RNN model as meta learner")
            # 重构 stacking_train
            stacking_train_reshaped = stacking_test.reshape(n_samples, 1, n_classifiers)
            print("stacking_train_reshaped shape is :", stacking_train_reshaped.shape)
            predictions = RNN_predict.meta_predict(stacking_train=stacking_train_reshaped, labels=labels,
                                                   dataset_name=dataset_name,
                                                   input_size=len(self.base_classifiers), num_classes=class_numbers)

        elif self.combiner == 'BiLSTM_model':
            print('*' * 10)
            print("Now is predicting BiLSTM_model as meta learner")
            stacking_train_reshaped = stacking_test.reshape(n_samples, 1, n_classifiers)
            predictions = BiLSTM_predict.predict_meta(device=device, dataset_name=dataset_name,
                                                      num_classes=class_numbers, stacking_train=stacking_train_reshaped,
                                                      input_size=len(self.base_classifiers),
                                                      labels=labels)

        elif self.combiner == 'cnn_model':
            print('*' * 10)
            print("Now is predicting CNN_model as meta learner")
            predictions = CNN_predict.meta_predict(stacking_train=stacking_test, labels=labels,
                                                   input_size=len(self.base_classifiers),
                                                   num_classes=class_numbers,
                                                   dataset_name=dataset_name)

        elif self.combiner == 'tcn_model':
            # TCN 模型的输入形状应该是 (N, C_in, L_in)，即 (batch_size, input_size, sequence_length)。
            print('*' * 10)
            print("Now is predicting TCN_model as meta learner")
            stacking_train_reshaped = stacking_test.reshape(stacking_test.shape[0], stacking_test.shape[1], 1)
            print("stacking_train shape ", stacking_train_reshaped.shape)
            # 调整形状
            predictions = TCN_predicts.meta_predict(stacking_train=stacking_train_reshaped, labels=labels,
                                                    input_size=len(self.base_classifiers), dataset_name=dataset_name,
                                                    num_classes=class_numbers, device=device)
            # save_model(tcn_meta, path='./models', name="meta_TCN_model.pt", model_state_dict=False)

        elif self.combiner == 'BiTCN_model':  # 同BiTCN
            print('*' * 10)
            print("Now is predicting BiTCN_model as meta learner")
            stacking_train_reshaped = stacking_test.reshape(stacking_test.shape[0], stacking_test.shape[1], 1)
            predictions = BiTCN_pred_temp.meta_predict(stacking_train=stacking_train_reshaped, labels=labels,
                                                       input_size=len(self.base_classifiers),  # 输入通道数
                                                       num_classes=class_numbers, device=device,
                                                       dataset_name=dataset_name)

        elif self.combiner == 'ResNet_model':
            print('*' * 10)
            print("Now is predicting resnet_model as meta learner")
            # resnet_model_path = 'models/USTC/Resnet34_3.pth'
            # stacking_train_reshaped = stacking_train.reshape()
            # batch_size, channels, height, weight
            reshaped_train = stacking_test.reshape(n_samples, n_classifiers, 1, 1)

            predictions = ResNet_predict.meta_predict(stacking_train=reshaped_train, labels=labels,
                                                      dataset_name=dataset_name,
                                                      num_classes=class_numbers)

        elif self.combiner == 'EfficientNet_model':
            print('*' * 10)
            print("Now is predicting EfficientNet_model as meta learner")
            # EfficientNet_model_path = './models/EfficientNet_model.pth'

            # batch_size, channels, height, weight，同ResNet
            reshaped_train = stacking_test.reshape(n_samples, n_classifiers, 1, 1)
            predictions = EfficientNet_predict.meta_predict(stacking_train=reshaped_train, labels=labels,
                                                            device=device, num_classes=class_numbers,
                                                            dataset_name=dataset_name,
                                                            )

        return predictions
