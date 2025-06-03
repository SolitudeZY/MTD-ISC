import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

# from renet_model_improved import resnet34, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d
from renet_model import resnet34, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d


#  ####基本上与AlexNet,VGG,GoogLeNet相似，不同在于1.图像预处理line18-line26，2.采用预训练模型权重文件进行迁移学习line64-line73
def main():
    print(torch.cuda.is_available())
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:2")
    print("device:", device)
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),  # 标准化参数来自官网
        "val": transforms.Compose([
            # transforms.Resize(256),  # 验证过程图像预处理有变动，将原图片的长宽比固定不动，将其最小边长缩放到256
            # transforms.CenterCrop(224),  # 再使用中心裁剪裁剪一个224×224大小的图片
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
    image_path = os.path.join(data_root, "datasets", "UJS-new-improved-ddim-splite")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    epochs = 10
    EP = 10
    for i in range(EP):
        print("la:", i)
        net = resnet34()
        print("training epoch {}:".format(i))
        best_acc = 0.0
        save_path = './resnet-UJS-new-improved-su-ddpm-splite-' + str(i) + '.pth' # resnet-ustc-new-improved-row-
        train_steps = len(train_loader)
        for epoch in range(epochs):
            # assert os.path.exists(save_path), "file {} does not exist.".format(save_path)
            print(os.path.exists(save_path))
            if os.path.exists(save_path):
                net.load_state_dict(torch.load(save_path, map_location=device))  # 通过net.load_state_dict方法载入模型权重
                for param in net.parameters():
                    param.requires_grad = False
            # change fc layer structure
            in_channel = net.fc.in_features  # net.fc即model.py中定义的网络的全连接层,in_features是输入特征矩阵的深度
            net.fc = nn.Linear(in_channel, 10)  # 重新定义全连接层，输入深度即上面获得的输入特征矩阵深度，类别为当前预测的花分类数据集类别5
            net.to(device)

            # define loss function
            loss_function = nn.CrossEntropyLoss()

            # for param in net.parameters():
            #     param.requires_grad = False

            # construct an optimizer
            params = [p for p in net.parameters() if p.requires_grad]
            optimizer = optim.Adam(params, lr=0.001) # 0.0001
            # optimizer = optim.SGD(params, lr=0.001)  # 0.0001
            # train
            net.train()  # 在训练过程中,self.training=True,有BN层的存在，区别于net.eval()
            running_loss = 0.0
            train_bar = tqdm(train_loader)
            for step, data in enumerate(train_bar):
                images, labels = data
                optimizer.zero_grad()
                logits = net(images.to(device))
                loss = loss_function(logits, labels.to(device))  # 计算损失
                loss.backward()  # 将损失反向传播
                optimizer.step()  # 更新每一个节点的参数

                # print statistics
                running_loss += loss.item()

                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)

            # validate
            net.eval()  # 在验证过程中，self.training=False,没有BN层
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():  # 用以禁止pytorch对参数进行跟踪，即在验证过程中不去计算损失梯度
                val_bar = tqdm(validate_loader)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = net(val_images.to(device))
                    # loss = loss_function(outputs, test_labels)
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                               epochs)

            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
