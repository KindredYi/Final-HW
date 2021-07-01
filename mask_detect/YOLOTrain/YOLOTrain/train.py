import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from Utils.dataloader import yolo_dataset_collate, YoloDataset
from Net.net_train import YOLOLoss,Generator
from Net.YOLO_Net import YoloBlock
from tqdm import tqdm
import time

def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1,3,2])

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_loss = 0
    val_loss = 0
    net.train()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            optimizer.zero_grad()
            outputs = net(images)
            losses = []
            num_pos_all = 0
            for i in range(2):
                loss_item, num_pos = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item)
                num_pos_all += num_pos
            loss = sum(losses) / num_pos_all
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(**{'当前损失': total_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    net.eval()
    print('开始验证')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []
                num_pos_all = 0
                for i in range(2):
                    loss_item, num_pos = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos
                loss = sum(losses) / num_pos_all
                val_loss += loss.item()
            pbar.set_postfix(**{'当前损失': val_loss / (iteration + 1)})
            pbar.update(1)
    print('完成验证')
    print('迭代:' + str(epoch + 1) + '/' + str(Epoch))
    print('当前损失: %.4f || 验证损失: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    t=time.ctime()
    print('保存权重:', str(epoch + 1))
    # torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    # (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
if __name__ == "__main__":
    Cuda = False
    Use_Data_Loader = True
    normalize = False
    input_shape = (416, 416)
    anchors_path = 'Model/anchors.txt'
    classes_path = 'Model/classes.txt'
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)
    mosaic =True
    Cosine_lr = True
    smoooth_label = 0
    model = YoloBlock(len(anchors[0]), num_classes)
    model_path = "Model/YOLO V4 tiny_weight.pth"
    print('已加载预训练权重')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('完成')
    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()
    yolo_losses = []
    for i in range(2):
        yolo_losses.append(YOLOLoss(np.reshape(anchors, [-1, 2]), num_classes, \
                                    (input_shape[1], input_shape[0]), smoooth_label, Cuda, normalize))

    annotation_path = './trainlist_20000.txt'
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    if True:
        lr = 1e-3
        Batch_size = 32
        Init_Epoch = 0
        Freeze_Epoch = 50
        optimizer = optim.Adam(net.parameters(), lr)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic,
                                        is_train=True)
            val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
            gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (input_shape[0], input_shape[1])).generate(train=True, mosaic=mosaic)
            gen_val = Generator(Batch_size, lines[num_train:],
                                (input_shape[0], input_shape[1])).generate(train=False, mosaic=mosaic)
        epoch_size = max(1, num_train // Batch_size)
        epoch_size_val = num_val // Batch_size
        for param in model.backbone.parameters():
            param.requires_grad = False
        for epoch in range(Init_Epoch, Freeze_Epoch):
            fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda)
            lr_scheduler.step()
    if True:
        lr = 1e-4
        Batch_size = 32
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100
        optimizer = optim.Adam(net.parameters(), lr)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic,
                                        is_train=True)
            val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
            gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (input_shape[0], input_shape[1])).generate(train=True, mosaic=mosaic)
            gen_val = Generator(Batch_size, lines[num_train:],
                                (input_shape[0], input_shape[1])).generate(train=False, mosaic=mosaic)

        epoch_size = max(1, num_train // Batch_size)
        epoch_size_val = num_val // Batch_size
        for param in model.backbone.parameters():
            param.requires_grad = True
        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, Cuda)
            lr_scheduler.step()

