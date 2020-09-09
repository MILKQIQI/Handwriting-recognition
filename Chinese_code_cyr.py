# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:53:48 2020

@author: 臧
"""
import os
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

EPOCH = 20 #训练几次
BATCH_SIZE = 50 #数据集划分
LR = 0.001 #学习率
DOWNLOAD_MNIST = False 
start=time.clock()#程序开始时间

def classes_txt(root, out_path, num_class=None):
    '''
    write image paths (containing class name) into a txt file.
    :param root: data set path
    :param out_path: txt file path
    :param num_class: how many classes needed
    :return: None
    '''
    dirs = os.listdir(root) # 列出根目录下所有类别所在文件夹名
    if not num_class:		# 不指定类别数量就读取所有
        num_class = len(dirs)

    if not os.path.exists(out_path): # 输出文件路径不存在就新建
        f = open(out_path, 'w')
        f.close()
	# 如果文件中本来就有一部分内容，只需要补充剩余部分
	# 如果文件中数据的类别数比需要的多就跳过
    with open(out_path, 'r+') as f:
        try:
            end = int(f.readlines()[-1].split('/')[-2]) + 1
        except:
            end = 0
        if end < num_class - 1:
            dirs.sort()
            dirs = dirs[end:num_class]
            for dir in dirs:
                files = os.listdir(os.path.join(root, dir))
                for file in files:
                    f.write(os.path.join(root, dir, file) + '\n')

class MyDataset(Dataset):
    def __init__(self, txt_path, num_class, transforms=None):
        super(MyDataset, self).__init__()
        images = [] # 存储图片路径
        labels = [] # 存储类别名，在本例中是数字
        # 打开上一步生成的txt文件
        with open(txt_path, 'r') as f:
            for line in f:
                if int(line.split('/')[-2]) >= num_class:  # 只读取前 num_class 个类 #\\
                    break
                line = line.strip('\n')
                images.append(line)
                labels.append(int(line.split('/')[-2]))#\\
        self.images = images
        self.labels = labels
        self.transforms = transforms # 图片需要进行的变换，ToTensor()等等

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB') # 用PIL.Image读取图像
        label = self.labels[index]
        if self.transforms is not None:
            image = self.transforms(image) # 进行变换
        return image, label

    def __len__(self):
        return len(self.labels)



    
class NetSmall(nn.Module):
#    def __init__(self):
#        super(NetSmall, self).__init__()
#        self.conv1 = nn.Sequential(
#                nn.Conv2d(
#                        in_channels = 1, # 输入图片的高度
#                        out_channels = 16, # 16个特征卷积过滤器
#                        kernel_size = 5, #卷积宽度（长度）
#                        stride = 1, # 步长
#                        padding = 2, # 扩展图片边缘长宽度
#                            # if stirder = 1, padding = (kernel_size-1)/2
#                ), # ->(16, 64, 64))
#                nn.ReLU(),
#                nn.MaxPool2d(kernel_size = 2), # 池化层→筛选重要信息 ！!取一定区域内最大值!！
#        ) # -> (16, 32, 32))
#        self.conv2 = nn.Sequential(
#                nn.Conv2d(16, 32, 5, 1, 2),
#                nn.ReLU(),
#                nn.MaxPool2d(2)
#        ) # -> (32, 16, 16))
#        self.out = nn.Linear(32 * 16 * 16, 100)
#    def forward(self, x):
#        x = self.conv1(x)
#        x = self.conv2(x)
#        x = x.view(x.size(0), -1)
#        output = self.out(x)
#        return output
#    
    def __init__(self):
        super(NetSmall, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) # 3个参数分别是in_channels，out_channels，kernel_size(filter)，还可以加padding
        # stride = 1时,输出大小 = 输入大小-kernel_size+1
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 512)
        self.fc2 = nn.Linear(512, 84)
        self.fc3 = nn.Linear(84, 100) #84,x,其中x与num_class一致

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 2704)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 以下为主程序
# 首先将训练集和测试集文件途径和文件名以txt保存在一个文件夹中
# 同一文件夹下需要存在1checkpoint文件夹2test文件夹3train文件夹4test.txt5train.txt
root = 'C:/Users/Msi/PycharmProjects/手写数字识别/data' # 这是我文件的储存位置
classes_txt(root + '/train', root+'/train.txt')
classes_txt(root + '/test', root+'/test.txt')

# 由于我的数据集图片尺寸不一，因此要进行resize，这里还可以加入数据增强，灰度变换，随机剪切等等
transform = transforms.Compose([transforms.Resize((64,64)), # 将图片大小重设为 64 * 64
                                transforms.Grayscale(),
                                transforms.ToTensor()])

train_set = MyDataset(root + '/train.txt', num_class=100, transforms=transform) # num_class 选取100种汉字  提出图片和标签
test_set = MyDataset(root + '/test.txt', num_class =100, transforms = transform)#num_class与self.fc3中84,x一致
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) # 装进迭代器中
test_loader = DataLoader(test_set, batch_size=6033, shuffle=True) #batch_size指test的样本数，100-6033，3755-223991

device = torch.device('cuda:0')
for step, (x,y) in enumerate(test_loader):
    test_x, labels_test = x.to(device), y.to(device)
    
model = NetSmall()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss() 
model.to(device)
# 训练集

for epoch in range(EPOCH):
    startepoch=time.clock()
    for step, (x,y) in enumerate(train_loader):
        picture, labels = x.to(device), y.to(device)
        
        output = model(picture)
        loss = loss_func(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            test_output = model(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels_test).sum().item() / labels_test.size(0)
            print('Epoch:', epoch, '| train loss:%.4f' % loss.data, '| test accuracy:', accuracy)
    #检查点保存
    elapsed_epoch=(time.clock()-startepoch)
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss},
    str(root)+'/checkpoint/cp_epoch_'+str(epoch)+'.tar')#检查点路径保存，可改动
    print('Finish saving checkpoint epoch:'+str(epoch)+' | Time used:'+str(elapsed_epoch))
    print('Dir='+str(root)+'/checkpoint/cp_epoch_'+str(epoch)+'.tar')

print('Finish training')

#保存推理模型
torch.save(model.state_dict(),root+'/chinese_rec_params.pth')
#保存整个模型
torch.save(model,root+'/chinese_rec_full.pkl')
alltime=time.clock()-start
print('Finish saving all files | All time used = '+str(alltime))
print('Model_Dir='+root)





