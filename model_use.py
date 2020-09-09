# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 16:20:45 2020

@author: cyr
"""

import os
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import linecache
import matplotlib.pyplot as plt
#模型checkpoint文件存放地址,待识别txt文件地址
model_path='C:/Users/cyr/Desktop/imgidentification/wordident/word/use/data/checkpoint/cp_epoch_19.tar'
rectxt_path='C:/Users/cyr/Desktop/imgidentification/wordident/word/use/data/chinese/test.txt'
dict_path='C:/Users/cyr/Desktop/imgidentification/wordident/word/use/data/dictionary.txt'
#训练网络
class NetSmall(nn.Module):
    def __init__(self):
        super(NetSmall, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) # 3个参数分别是in_channels，out_channels，kernel_size，还可以加padding
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
# 预测函数
# 目前只实现了单张图片识别
# def inference():
print('Start inference...')
transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])

f = open(rectxt_path)  # 填写存放待识别文件路径txt
num_line = sum(line.count('\n') for line in f)  # 读取txt行数
f.seek(0, 0)
line = int(torch.rand(1).data * num_line - 10)  # -10 for '\n's are more than lines
while line > 0:
    f.readline()
    line -= 1
img_path = f.readline().rstrip('\n')
f.close()
    #label = int(img_path.split('/')[-2])#通过文件夹获取标签
    #print('label:\t%4d' % label)#输出图片标签
input = Image.open(img_path).convert('1')  # RGB模式，1模式（二值化）
input = transform(input)
input = input.unsqueeze(0)
model = NetSmall()
model.eval()
checkpoint = torch.load(model_path)  # 填写pth模型文件路径
model.load_state_dict(checkpoint['model_state_dict'])
output = model(input)
_, pred = torch.max(output.data, 1)
    
print('predict:\t%4d' % pred)#输出预测标签值
chinese = linecache.getline(dict_path,(pred+1)*2)#填写字典txt
print('predict:\t' + chinese)#根据字典输出对应汉字
#执行单张图片预测
#inference()






