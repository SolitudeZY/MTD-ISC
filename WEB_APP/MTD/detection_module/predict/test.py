import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split

from Stacking_train import data_pre_process, get_clf
from stacking_esemble_classifier import png_path, StackingClassifier, save_labels
from utils import get_class_nums


class BiTCN(nn.Module):
    def __init__(self, in_channels, num_classes, num_channels, kernel_size=3, dropout=0.05):
        super(BiTCN, self).__init__()
        self.convs_forward = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels if i == 0 else num_channels[i - 1] * 2,
                      out_channels=num_channels[i],
                      kernel_size=kernel_size,
                      padding=(kernel_size // 2))
            for i in range(len(num_channels))
        ])
        self.convs_backward = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels if i == 0 else num_channels[i - 1] * 2,
                      out_channels=num_channels[i],
                      kernel_size=kernel_size,
                      padding=(kernel_size // 2))
            for i in range(len(num_channels))
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels[-1] * 2, num_classes)

    def forward(self, x):
        for conv_forward, conv_backward in zip(self.convs_forward, self.convs_backward):
            x_forward = nn.ELU()(conv_forward(x))
            x_backward = nn.ELU()(conv_backward(torch.flip(x, [2])))
            x = torch.cat((x_forward, x_backward), dim=1)
            x = self.dropout(x)
        x = x.mean(dim=2)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    Stacking_Classifier = get_clf()
    root = os.path.join(os.getcwd(), '../')
    X_train, X_test, y_test, y_train, dataset_name = data_pre_process(root, png_path, None)
    train_save_path = './' + dataset_name + '_stacking_train.npy'
    test_save_path = './' + dataset_name + '_stacking_test.npy'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # 将中间数据和标签转换为 TensorDataset
    stacking_train_tensor = torch.tensor(stacking_test, dtype=torch.float32)
    labels_tensor = torch.tensor(stacking_test_labels, dtype=torch.long)

    # 拆分数据集
    X_train, X_val, y_train, y_val = train_test_split(stacking_train_tensor, labels_tensor, test_size=0.2,
                                                      random_state=42)

    # 创建 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 定义模型
    in_channels = stacking_train.shape[1]
    num_classes = class_numbers = get_class_nums(Png_path=png_path)

    model = BiTCN(in_channels=in_channels, num_classes=num_classes, num_channels=[64, 64], kernel_size=3, dropout=0.05)
    model = model.cuda()

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    # 训练模型
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

    # 验证模型
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)
            val_loss += loss_function(val_outputs, val_labels).item()
            val_accuracy += (val_outputs.argmax(1) == val_labels).sum().item()
        val_loss /= len(val_loader)
        val_accuracy /= len(val_dataset)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
