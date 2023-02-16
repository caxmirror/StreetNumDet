#Part3 定义数据集
from ctypes.wintypes import INT, LONG
import os, sys, glob, shutil, json
from pickletools import long1, long4
import cv2

from PIL import Image
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label 
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        # 原始SVHN中类别10为数字0
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl)  + (6 - len(lbl)) * [10] #将lbl用数字10填充至6个
        
        return img, torch.from_numpy(np.array(lbl[0:6]))

    def __len__(self):
        return len(self.img_path)

train_path = glob.glob('C:/Project/python/Jupyter/Practise/tianchi/data/mchar_train/mchar_train/*.png')
train_path.sort()
train_json = json.load(open('C:/Project/python/Jupyter/Practise/tianchi/data/mchar_train.json'))
train_label = [train_json[x]['label'] for x in train_json]

train_loader = torch.utils.data.DataLoader(
        SVHNDataset(train_path, train_label,
                   transforms.Compose([
                       transforms.Resize((64, 128)), #固定尺寸
                       transforms.ColorJitter(0.3, 0.3, 0.2), #随机颜色变换
                       transforms.RandomRotation(5), #随机旋转
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])), 
    batch_size=10, # 每批样本个数
    shuffle=True, # 是否打乱顺序
    num_workers=0 # 读取的线程个数
)

val_path = glob.glob('C:/Project/python/Jupyter/Practise/tianchi/data/mchar_val/mchar_val/*.png')
val_path.sort()
val_json = json.load(open('C:/Project/python/Jupyter/Practise/tianchi/data/mchar_val.json'))
val_label = [val_json[x]['label'] for x in val_json]

val_loader = torch.utils.data.DataLoader(
        SVHNDataset(val_path, val_label,
                   transforms.Compose([
                       transforms.Resize((64, 128)), #固定尺寸
                       transforms.ColorJitter(0.3, 0.3, 0.2), #随机颜色变换
                       transforms.RandomRotation(5), #随机旋转
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])), 
    batch_size=10, # 每批样本个数
    shuffle=False, # 是否打乱顺序
    num_workers=0 # 读取的线程个数
)


#Part4 定义模型
import torch
torch.manual_seed(0) #固定随机数种子，确保参数（训练效果）一致
torch.backends.cudnn.deterministic = False #GPU使用默认算法
torch.backends.cudnn.benchmark = True #GPU寻找最优算法

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

# 定义模型
class SVHN_Model1(nn.Module): #输入数据为3x64x128
    def __init__(self):
        super(SVHN_Model1, self).__init__() #super()可以理解为指向父类的指针
        # CNN提取特征模块
        self.cnn = nn.Sequential( #顺序构建网络
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2)), #in_channel, out_channel, 卷积核大小, 滑动步长
            nn.ReLU(),  #非线性激活函数，为了避免神经网络中全部是线性函数，那样就不需要神经网络了
            nn.MaxPool2d(2), #最大池化MaxPooling，大小为2x2
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(), 
            nn.MaxPool2d(2),
        )
        # 
        self.fc1 = nn.Linear(32*3*7, 11)
        self.fc2 = nn.Linear(32*3*7, 11) #有点问题
        self.fc3 = nn.Linear(32*3*7, 11)
        self.fc4 = nn.Linear(32*3*7, 11)
        self.fc5 = nn.Linear(32*3*7, 11)
        self.fc6 = nn.Linear(32*3*7, 11)
    
    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        c6 = self.fc6(feat)
        return c1, c2, c3, c4, c5, c6
    

model = SVHN_Model1()
#Part5 训练和验证模型
import time
# 损失函数
criterion = nn.CrossEntropyLoss(size_average=False) #求batch中的loss之和

# 优化器
optimizer = torch.optim.Adam(model.parameters(), 0.001)
best_loss = 1000.0
val_loss, timer = [], []

# 迭代30个Epoch
for epoch in range(10):
    time_start = time.time()
    train(train_loader, model, criterion, optimizer, epoch)
    val_loss = validate(val_loader, model, criteron)
    
    if val_los < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(),'./model.pt')
        
    time_end = time.time()
    time_cost = time_end - time_start
    timer.append(time_cost)
    print("epoch%d,time_cost = %d" %(epoch,time_cost))
    
def train(train_loader, model, criterion, optimizer, epoch):
    model.train() #启用batch normalization和drop out
    
    for i,(images,labels) in enumerate(train_loader):  #train_loader中的数据分为两部分,data[0]表示图片，data[1]表示标识
        c0, c1, c2, c3, c4, c5 = model(images)
        if i<5 :
            print(c0,c1,c2,c3,c4,c5) #测试：c0~c5输出是否是【0，10】的正整数
            print(images.shape)#测试：输入图像的维度
        loss = criterion(c0, torch.tensor(labels[:, 0],dtype=torch.long)) + \
                criterion(c1, torch.tensor(labels[:, 1],dtype=torch.long)) + \
                criterion(c2, torch.tensor(labels[:, 2],dtype=torch.long)) + \
                criterion(c3, torch.tensor(labels[:, 3],dtype=torch.long)) + \
                criterion(c4, torch.tensor(labels[:, 4],dtype=torch.long)) + \
                criterion(c5, torch.tensor(labels[:, 5],dtype=torch.long))
        loss /= 6 #计算损失函数,6个数字全部对上损失=0，全部不对损失=1
        optimizer.zero_grad() #清零现有梯度
        loss.backward() #当前梯度后向传递
        optimizer.step() #更新权值
        
def validate(val_loader, model, criterion):
    model.eval()
    with torch.no_grad():
        for i,(images,labels) in enumerate(val_loader):  
            c0, c1, c2, c3, c4, c5 = model(images)
            loss = criterion(c0, torch.tensor(labels[:, 0],dtype=torch.long)) + \
                    criterion(c1, torch.tensor(labels[:, 1],dtype=torch.long)) + \
                    criterion(c2, torch.tensor(labels[:, 2],dtype=torch.long)) + \
                    criterion(c3, torch.tensor(labels[:, 3],dtype=torch.long)) + \
                    criterion(c4, torch.tensor(labels[:, 4],dtype=torch.long)) + \
                    criterion(c5, torch.tensor(labels[:, 5],dtype=torch.long))
            loss /= 6 #计算损失函数,6个数字全部对上损失=0，全部不对损失=1

            val_loss.append(loss.item()) #误差损失
    return np.mean(val_loss)

    