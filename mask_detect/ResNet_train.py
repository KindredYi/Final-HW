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
            #print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(0)
        for file in glob.glob(self.store_path + '/' + split + '/nomask1/*.jpg'):
            #print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(1)
 
    def __getitem__(self, item):
        #print(self.img_list[item])
        img = Image.open(self.img_list[item]).convert('L')
        #img = img.resize((224, 224), Image.ANTIALIAS)
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.label_list[item]
        return img, label
 
    def __len__(self):
        return len(self.img_list)

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

####################模型定义（ResNet）##########
#定义ResNet基本模块-残差模块
def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)

# Residual Block
class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)
        if self.downsample:
            residual = self.downsample(x)
        out = out + residual
        out = F.relu(out, True)
        return out

class ResNet(nn.Module):
    # 实现主module：ResNet34
    # ResNet34 包含多个layer，每个layer又包含多个residual block
    # 用子module来实现residual block，用_make_layer函数来实现layer
    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()
        # 前几层图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(16, 16, 3)
        self.layer2 = self._make_layer(16, 32, 4, stride=1)
        self.layer3 = self._make_layer(32, 64, 6, stride=1)
        self.layer4 = self._make_layer(64, 64, 3, stride=1)
        self.fc = nn.Linear(1024, num_classes)  # 分类用的全连接

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        # 构建layer,包含多个residual block
        shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, 1, stride, bias=False), nn.BatchNorm2d(outchannel))
        layers = []
        layers.append(residual_block(inchannel, outchannel, stride, shortcut))
        for i in range(1, block_num):
            layers.append(residual_block(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        #print(x.shape)
        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)
        x = F.avg_pool2d(x, 7)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        return self.fc(x)


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

def acc(y_pred , y ,num):
    t=f=0
    for i in range(num):
        if(y_pred[i][0]>y_pred[i][1]):
            c=0
        else:
            c=1
        if(c==y[i]):
            t=t+1
    return t

###################模型训练#################
def train(loader, net, Loss, optimizer, path ):
    print('start training:')
    d = -4
    loss_1 = 9999999
    i=0
    for t in range(3):
        for x,y in loader:
            x=x.cuda(0)
            y=y.cuda(0)
            y_pred = net(x)  # 前向传播：通过像模型输入x计算预测的y
            loss = Loss(y_pred, y)  # 计算loss
            print("第{}次,CrossEntropyLoss为 {}".format(i + 1, loss.item()))
            optimizer.zero_grad()  # 在反向传播之前，使用optimizer将它要更新的所有张量的梯度清零
            loss.backward()  # 反向传播：根据模型的参数计算loss的梯度
            optimizer.step()  # 调用Optimizer的step函数使它所有参数更新
            i = i + 1
    y_pred = net(x)   # 前向传播：通过向模型输入x计算预测的y
    loss = Loss(y_pred, y)  # 计算最终的训练误差
    print("训练完成,CrossEntropyLoss为 {}".format(loss.item()))
    return net


###################模型测试###################
def test(loader, net, Loss , num):
    #net = torch.load(net_path)
    sum_loss = 0
    i=0
    t = 0
    for x,y in loader:
        y_pred = net(x)
        loss = (Loss(y_pred, y))  # 计算loss
        sum_loss = sum_loss+loss
        t = t + acc(y_pred, y, num)
        i = i + 1
    print("测试完成,CrossEntropyLoss为 {}".format(sum_loss/i))
    print("测试完成，ACC为 {}".format(t/(num*i)))
    print(t)
    return 0

if __name__ == '__main__':
    store_path = './'
    split = 'train_data'
    train_dataset = my_dataset(store_path, split, data_transform)
    dataset_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
    net = ResNet()
    net = net.cuda(0)
    Loss = define_loss()
    optimizer = define_optimizer(1e-3)
    #imshow(torchvision.utils.make_grid(images))
    Net1 = train(dataset_loader, net, Loss, optimizer, store_path)
    Net1.eval()
    Net1 = Net1.cpu()
    #torch.save(Net1.state_dict(), store_path + '/RESNET_model7.pth')
    split = 'test_data'
    num = 1
    test_dataset = my_dataset(store_path, split, data_transform)
    test_dataset_loader = DataLoader(test_dataset, batch_size=num, shuffle=True, num_workers=1)
    test(test_dataset_loader, Net1, Loss, num)
    #imshow(torchvision.utils.make_grid(images))

