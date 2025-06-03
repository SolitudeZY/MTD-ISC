import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torchvision import transforms, datasets
from tqdm import tqdm
from FocalLoss import FocalLoss
from utils import data_pre_process, calculate_tpr_fpr, calculate_tpr_fpr_multiclass
# from resnet_model_improved import resnet34, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d
from resnet_model import resnet34, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d
import torch.amp as amp  # 导入混合精度训练模块

#  ####基本上与AlexNet,VGG,GoogLeNet相似，不同在于1.图像预处理line18-line26，2.采用预训练模型权重文件进行迁移学习line64-line73

# data_transform = {
#     "train": transforms.Compose([
#         transforms.RandomResizedCrop(16),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),  # 标准化参数来自官网
#     "test": transforms.Compose([
#         transforms.Resize(16),  # 验证过程图像预处理有变动，将原图片的长宽比固定不动，将其最小边长缩放到256
#         transforms.CenterCrop(16),  # 再使用中心裁剪裁剪一个16×16大小的图片
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#
#     ])}
# data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
# image_path = os.path.join(data_root, "pre-processing", "4_Png_16_CTU")
# assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
# train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
#                                      transform=data_transform["train"])
# test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
#                                     transform=data_transform["test"])
# flower_list = train_dataset.class_to_idx
# cla_dict = dict((val, key) for key, val in flower_list.items())
# # write dict into json file
# json_str = json.dumps(cla_dict, indent=4)
# with open('class_indices.json', 'w') as json_file:
#     json_file.write(json_str)
#
# batch_size = 64
# nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
# print('Using {} dataloader workers every process'.format(nw))
#
# val_num = len(test_dataset)
# train_num = len(train_dataset)
# print("using {} images for training, {} images for validation.".format(train_num,
#                                                                        val_num))
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # png_path = '4_Png_16_CTU'
    png_path = '4_Png_16_ISAC'

    directory_path = os.path.join(os.path.abspath(os.getcwd()), f"../pre-processing/{png_path}/Train")
    entries = os.listdir(directory_path)
    folder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
    class_numbers = folder_count

    train_loader, test_loader, _, _, _, dataset_name = data_pre_process(data_directory=os.getcwd(), png_path=png_path)
    epochs = 5
    EP = 10
    dataset_name = dataset_name[:-1]
    for i in range(1):
        print("la:", i)
        print("training epoch {}:".format(i))
        net = resnet34(num_classes=class_numbers)
        best_acc = 0.0
        # save_path = './resnet-USTC-new-new-improved-NT-DDPM-splite-' + str(i) + '.pth'
        save_path = f"../models/{dataset_name}/Resnet34_{dataset_name}_{i}.pth"
        print("save_path exists: {}\nsaving model to {}".format(os.path.exists(save_path), save_path))
        train_steps = len(train_loader)

        for epoch in range(epochs):  # train
            scaler = amp.GradScaler(device='cuda')  # 创建GradScaler对象
            # print("save_path exist :", os.path.exists(save_path))
            if os.path.exists(save_path):
                net.load_state_dict(torch.load(save_path, map_location=device))  # 通过net.load_state_dict方法载入模型权重
                for param in net.parameters():
                    param.requires_grad = True
            # change fc layer structure
            in_channel = net.fc.in_features  # net.fc即model.py中定义的网络的全连接层,in_features是输入特征矩阵的深度
            net.fc = nn.Linear(in_channel, out_features=class_numbers)  # 重新定义全连接层，输入深度即上面获得的输入特征矩阵深度，类别为当前预测的花分类数据集类别5
            net.to(device)

            # define loss function and construct an optimizer
            params = [p for p in net.parameters() if p.requires_grad]
            optimizer = optim.Adam(params, lr=0.0001)
            # loss_function = nn.CrossEntropyLoss()
            loss_function = FocalLoss(gamma=2, alpha=0.75)

            # train
            net.train()  # 在训练过程中,self.training=True,有BN层的存在，区别于net.eval()
            running_loss = 0.0
            train_bar = tqdm(train_loader)
            for step, data in enumerate(train_bar):
                images, labels = data
                optimizer.zero_grad()
                with amp.autocast("cuda"):  # 使用autocast上下文管理器
                    logits = net(images.to(device))
                    loss = loss_function(logits, labels.to(device))

                scaler.scale(loss).backward()  # 缩放损失并反向传播
                scaler.step(optimizer)  # 更新参数
                scaler.update()  # 更新scaler

                # print statistics
                running_loss += loss.item()

                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

            # validate
            net.eval()  # 在验证过程中，self.training=False,没有BN层
            acc = 0.0  # accumulate accurate number / epoch
            all_pred = []
            true_labels = []
            with torch.no_grad():  # 用以禁止pytorch对参数进行跟踪，即在验证过程中不去计算损失梯度
                val_bar = tqdm(test_loader)
                for val_images, val_labels in val_bar:
                    outputs = net(val_images.to(device))
                    # loss = loss_function(outputs, test_labels)
                    predict_y = torch.max(outputs, dim=1)[1]
                    all_pred.extend(predict_y.cpu().numpy())
                    true_labels.extend(val_labels.cpu().numpy())
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                               epochs)
            val_num = len(test_loader.dataset)
            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.5f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))

            TPR, FPR = calculate_tpr_fpr_multiclass(true_labels, all_pred, n_classes=class_numbers)
            print(f"FPR: {FPR}, TPR: {TPR}")
            print("accuracy:", val_accurate)
            print("f1 score", f1_score(true_labels, all_pred, average='weighted'))

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
                print(f"saving at {save_path},accuracy: {val_accurate}", )
    print('Finished Training')


if __name__ == '__main__':
    main()
