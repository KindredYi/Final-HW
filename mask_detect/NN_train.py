from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.Resize((56, 56)),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])
     ])


class my_dataset(Dataset):
    def __init__(self, store_path, split, data_transform=None):
        self.store_path = store_path
        self.split = split
        self.transforms = data_transform
        self.img_list = []
        self.label_list = []
        for file in glob.glob(self.store_path + '/' + split + '/mask1/*.jpg'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(0)
        for file in glob.glob(self.store_path + '/' + split + '/nomask1/*.jpg'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(1)

    def __getitem__(self, item):
        # print(self.img_list[item])
        img = Image.open(self.img_list[item]).convert('L')
        # img = img.resize((224, 224), Image.ANTIALIAS)
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.label_list[item]
        return img, label

    def __len__(self):
        return len(self.img_list)


def define_model():
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # 三个全连接层
            self.fc1 = nn.Linear(56*56, 1200)
            self.fc15 = nn.Linear(1200, 84)
            self.fc2 = nn.Linear(84, 2)  # 最后输出为1x2的向量

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc15(x))
            x = self.fc2(x)
            return x

    net = Net()
    return net


################损失函数定义（CrossEntropyLoss）#########
def define_loss():
    Loss = nn.CrossEntropyLoss()
    return Loss


##############优化器定义#############
def define_optimizer(learning_rate):
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=0,
                                 amsgrad=False)
    return optimizer


def acc(y_pred, y, num):
    t = 0
    for i in range(num):
        if (y_pred[i][0] > y_pred[i][1]):
            c = 0
        else:
            c = 1
        if (c == y[i]):
            t = t + 1
    return t


###################模型训练#################
def train(loader, net, Loss, optimizer, path):
    print('start training:')
    d = -4
    loss_1 = 9999999
    i = 0
    for t in range(3):
        for x, y in loader:
            x = x.cuda(0)
            y = y.cuda(0)
            y_pred = net(x)  # 前向传播：通过像模型输入x计算预测的y
            loss = Loss(y_pred, y)  # 计算loss
            print("第{}次,CrossEntropyLoss为 {}".format(i + 1, loss.item()))
            optimizer.zero_grad()  # 在反向传播之前，使用optimizer将它要更新的所有张量的梯度清零
            loss.backward()  # 反向传播：根据模型的参数计算loss的梯度
            optimizer.step()  # 调用Optimizer的step函数使它所有参数更新
            i = i + 1
    y_pred = net(x)  # 前向传播：通过向模型输入x计算预测的y
    loss = Loss(y_pred, y)  # 计算最终的训练误差
    print("训练完成,CrossEntropyLoss为 {}".format(loss.item()))
    return net


###################模型测试###################
def test(loader, net, Loss, num):
    # net = torch.load(net_path)
    sum_loss = 0
    i = 0
    t = 0
    for x, y in loader:
        y_pred = net(x)
        loss = (Loss(y_pred, y))  # 计算loss
        sum_loss = sum_loss + loss
        t = t + acc(y_pred, y, num)
        i = i + 1
    print("测试完成,CrossEntropyLoss为 {}".format(sum_loss / i))
    print("测试完成，ACC为 {}".format(t / (i * num)))
    print(t)
    return 0


if __name__ == '__main__':
    store_path = './'
    split = 'train_data'
    train_dataset = my_dataset(store_path, split, data_transform)
    dataset_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
    net = define_model()
    net = net.cuda(0)
    Loss = define_loss()
    optimizer = define_optimizer(1e-3)
    # imshow(torchvision.utils.make_grid(images))
    Net1 = train(dataset_loader, net, Loss, optimizer, store_path)
    Net1.eval()
    Net1 = Net1.cpu()
    # torch.save(Net1.state_dict(), store_path + '/RESNET_model7.pth')
    split = 'test_data'
    num = 1
    test_dataset = my_dataset(store_path, split, data_transform)
    test_dataset_loader = DataLoader(test_dataset, batch_size=num, shuffle=True, num_workers=1)
    test(test_dataset_loader, Net1, Loss, num)
    # imshow(torchvision.utils.make_grid(images))

