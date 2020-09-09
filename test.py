"""
2020.8.22
@author: cq

测试灰度变换的结果
"""

# 导入模块
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse
import matplotlib.pyplot as plt

Image_0 = Image.open('E:/Pycharmproject/手写数字识别/6.jpg').convert('RGB')

# img = Image.fromarray(Image_0)

Img1 = transforms.Grayscale(Image_0)

plt.imshow(Image_0)
plt.show()


# transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
#                                     transforms.Grayscale(),
#                                     transforms.ToTensor()])