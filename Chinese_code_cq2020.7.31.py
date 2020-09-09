"""
2020.7.30
@author: cq

手写数字的识别
待解决的问题：1.只能进行单个汉字的识别，未能批量输出汉字识别结果,已完成
            2.结果只输出到控制台，未进行txt形式的输出，已完成
            3.模型对数据集中的识别效果较好，当手写后拍照进行识别，则无法取得较好的成果，需要在训练集中加入手写汉字

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

parse = argparse.ArgumentParser(description='Params for training. ')
# 数据集根目录
parse.add_argument('--root', type=str, default='E:/Pycharmproject/手写数字识别/data6', help='path to data set')
# 模式，3选1
parse.add_argument('--mode', type=str, default='train', choices=['train', 'validation', 'inference'])
# checkpoint 路径，保存推理模型
parse.add_argument('--log_path', type=str, default=os.path.abspath('.') + '/log.pth', help='dir of checkpoints')

parse.add_argument('--restore', type=bool, default=False, help='whether to restore checkpoints')  # 没有保存过模型或首次训练,False

parse.add_argument('--batch_size', type=int, default=16, help='size of mini-batch')
parse.add_argument('--image_size', type=int, default=64, help='resize image')
parse.add_argument('--epoch', type=int, default=100)
# 我的数据集类别数是3755，所以给定了一个选择范围
parse.add_argument('--num_class', type=int, default=100, choices=range(10, 3755))
args = parse.parse_args()

# 超参数定义
EPOCH = 10  # 训练次数
BATCH_SIZE = 50  # 数据集划分
LR = 0.001  # 学习率

# 提取数据集的路径
def class_txt(root, out_path, num_class=None):
    """
    :param root: 根目录
    :param out_path:txt存储的路径
    :param num_class: 需要读取的类别数目
    :return: None
    """

    dirs = os.listdir(root)  # 将root下面的子文件夹名存入一个list中
    if not num_class:  # 未指定类别数量就读取所有
        num_class = len(dirs)
        f = open(out_path, "w")
        f.close()
    if not os.path.exists(out_path):  # 如果outpath没有txt文件的存在，则新建
        f = open(out_path, 'w')
        f.close()

    with open(out_path, 'r+') as f:
        try:
            end = int(f.readlines()[-1].split('/')[-2]) + 1  ### 将类别文件名字作为类别数量
        except:
            end = 0
        if end < num_class - 1: # 如果文件中数据的类别数比需要的少，需补充剩余部分，反之跳过
            dirs.sort()
            dirs = dirs[end:num_class]
            for dir in dirs:
                files = os.listdir(os.path.join(root, dir))  # 类别文件的路径
                for file in files:
                    f.write(os.path.join(root, dir, file) + '\n')  # 将每张图片的路径写入txt中
"""
数据集的设置
"""
class MyDataset(Dataset):
    def __init__(self, txt_path, num_class, transforms=None):
        """

        :param txt_path: 打开class_txt函数生成的txt文件
        :param num_class: 需要读取的类别数目
        :param transforms:
        """
        super(MyDataset, self).__init__()
        images = []  # 存储图片路径
        labels = []  # 存储类别名，类别为数字
        # 打开上一步生成的txt文件
        with open(txt_path, 'r') as f:
            for line in f:
                if int(line.split("\\")[-2]) >= num_class:   # 只读取前num_class个类
                    break
                line = line.strip('\n')
                images.append(line)
                labels.append(int(line.split('\\')[-2]))
        self.images = images
        self.labels = labels
        self.transforms = transforms  # 图片需要进行的变换，ToTensor()等等

    def __getitem__(self, index):
        """
        在训练的时候返回输入网络的数据，图片和标签等等

        :param index:
        :return:
        """
        image = Image.open(self.images[index]).convert('RGB')  # 用PIL.Image读取图像，并用convert转化为RGB像素值
        label = self.labels[index]
        if self.transforms is not None:
            image = self.transforms(image)  # 进行变换
        return image, label

    def __len__(self):
        return len(self.labels)

class NetSmall(nn.Module):
    """
    搭建神经网络

    """
    def __init__(self):
        super(NetSmall,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)  # in_channels, out_channels, kernel_size
        # stride = 1时,输出大小 = 输入大小-kernel_size+1
        self.pool = nn.MaxPool2d(2, 2)  # 池化
        self.conv2 = nn.Conv2d(6, 16, 5)  # 卷积
        self.fc1 = nn.Linear(2704, 512)  # 全连接
        self.fc2 = nn.Linear(512, 84)
        self.fc3 = nn.Linear(84, args.num_class)  # 命令行参数

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 2704)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# """
# 主程序
# """
# if __name__ == '__main__':
#     # 首先将训练集和测试集文件途径和文件名以txt保存在一个文件夹中，路径自行定义
#     root = 'C:/Users/Msi/PycharmProjects/手写数字识别/data' # 这是我文件的储存位置
#     class_txt(root + '/train', root+'/train.txt')
#     class_txt(root + '/test', root+'/test.txt')
#
#     # 由于数据集图片尺寸不一，因此要进行resize（重设大小）
#     transform = transforms.Compose([transforms.Resize((64,64)), # 将图片大小重设为 64 * 64
#                                     transforms.Grayscale(),
#                                     transforms.ToTensor()])
#
#     # 提取训练集和测试集图片的路径
#     train_set = MyDataset(root + '/train.txt', num_class=100, transforms=transform) # num_class 选取100种汉字  提出图片和标签
#     test_set = MyDataset(root + '/test.txt', num_class=100, transforms=transform)
#     # 放入迭代器中
#     train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
#     test_loader = DataLoader(test_set, batch_size=5473, shuffle=True)
#     # 选择使用的设备
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     print(device)
#
#     # 这里的5473是因为测试集为5973张图片，当进行迭代时取第二批500个图片进行测试
#     for step, (x, y) in enumerate(test_loader):
#         test_x, labels_test = x.to(device), y.to(device)
#
#     """
#     参数优化
#     """
#     parse = argparse.ArgumentParser(description='Params for training. ')
#     parse.add_argument('--num_class', type=int, default=100, choices=range(10, 3755))
#     args = parse.parse_args()
#     model = NetSmall()
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR) # 参数优化
#     loss_func = nn.CrossEntropyLoss() #分类误差计算函数
#     device = torch.device('cpu')
#     model.to(device)
#
#     """
#     模型训练
#     """
#     for epoch in range(EPOCH):
#         for step, (x, y) in enumerate(train_loader):
#             picture, labels = x.to(device), y.to(device)
#
#             output = model(picture)
#             loss = loss_func(output, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             if step % 50 == 0:
#                 test_output = model(test_x)
#                 pred_y = torch.max(test_output, 1)[1].data.squeeze()
#                 accuracy = (pred_y == labels_test).sum().item() / labels_test.size(0)
#                 print('Epoch:', epoch, '| train loss:%.4f' % loss.data, '| test accuracy:', accuracy)
#                 # 输出训练次数、误差、准确率
#     print('Finish training')
def train():
    # 由于我的数据集图片尺寸不一，因此要进行resize，这里还可以加入数据增强，灰度变换，随机剪切等等
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])

    train_set = MyDataset(args.root + '/train.txt', num_class=args.num_class, transforms=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    # 选择使用的设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = NetSmall()
    model.to(device)
    # 训练模式
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 由命令行参数决定是否从之前的checkpoint开始训练
    if args.restore:
        checkpoint = torch.load(args.log_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        epoch = checkpoint['epoch']
    else:
        loss = 0.0
        epoch = 0

    while epoch < args.epoch:
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            # 这里取出的数据就是 __getitem__() 返回的数据
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outs = model(inputs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 200 == 199:  # every 200 steps
                print('epoch %5d: batch: %5d, loss: %f' % (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
            # 保存 checkpoint
        if epoch % 10 == 9:
            print('Save checkpoint...')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss},
                       args.log_path)
        epoch += 1

    print('Finish training')


def validation():
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])
    test_set = MyDataset(args.root + '/test.txt', num_class=args.num_class, transforms=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = NetSmall()
    model.to(device)
    checkpoint = torch.load(args.log_path, map_location=torch.device('cpu'))  # 加载整个模型结构以及参数
    model.load_state_dict(checkpoint['model_state_dict'])  # 仅加载参数
    model.eval()
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # inputs, labels = data[0].cuda(), data[1].cuda()
            inputs, labels = data[0], data[1]
            outputs = model(inputs)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            # correct += sum(int(predict == labels)).item()
            # 根据评论区反馈，如果上面这句报错，可以换成下面这句试试：
            correct += (predict == labels).sum().item()
            if i % 100 == 99:
                print('batch: %5d,\t acc: %f' % (i + 1, correct / total))
    print('Accuracy: %.2f%%' % (correct / total * 100))


def inference():
    print('Start inference...')
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])

    # f = open(args.root + '/test.txt')
    # num_line = sum(line.count('\n') for line in f)
    # f.seek(0, 0)
    # # 修改
    # # 批量读取文件路径，将其存入一个列表中
    # labels = []
    # ans = []
    # file = open(r'pre.txt', mode='w')
    # lines = f.readlines()
    # f.close()
    # label = -1
    #
    # for i in range(len(lines)):
    #     if label > 100:     # 可改，预测100类，总共3755类
    #         pass
    #     else:
    #         img_path = lines[i].rstrip('\n')
    #         print(img_path)
    #         label = int(img_path.split("\\")[-2])
    #         print('label:\t%4d' % label)
    #         input = Image.open(img_path).convert('RGB')
    #         input = transform(input)
    #
    #         # 网络默认接受4维数据，即[Batch, Channel, Heigth, Width]，所以要加1个维度
    #         input = input.unsqueeze(0)
    #         model = NetSmall()
    #         model.eval()
    #         checkpoint = torch.load(args.log_path)
    #         model.load_state_dict(checkpoint['model_state_dict'])
    #         output = model(input)
    #         _, pred = torch.max(output.data, 1)
    #         print('predict:\t%4d' % pred)
    #
    #         # 将结果存入list中
    #         labels.append(label)
    #         ans.append(int(pred.numpy()[0]))
    #
    #         # 将结果写入txt中
    #         file.write(img_path)
    #         file.write('\n')
    #         file.write('label：')
    #         file.write(str(label))
    #         file.write('\n')
    #         file.write('predict:')
    #         file.write(str(pred.numpy()[0]))
    #         file.write('\n')
    # file.close()
    #
    # # 计算预测精度
    # count = 0
    # for i in range(len(labels)):
    #     if labels[i] == ans[i]:
    #         count = count + 1
    #     else:
    #         pass
    # print('Accuracy of Predict: %.2f%%' % (count / len(labels) * 100))
    # 读取自己写的

    img_path = 'E:/Pycharmproject/手写数字识别/6.jpg'
    print(img_path)
    label = img_path.split("/")[-1]
    print(label)
    input = Image.open(img_path).convert('RGB')
    input = transform(input)

    # 网络默认接受4维数据，即[Batch, Channel, Heigth, Width]，所以要加1个维度
    input = input.unsqueeze(0)
    model = NetSmall()
    model.eval()
    checkpoint = torch.load(args.log_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    output = model(input)
    _, pred = torch.max(output.data, 1)
    print('predict:\t%4d' % pred)


if __name__ == '__main__':

    class_txt(args.root + '/train', args.root + '/train.txt', num_class=args.num_class)
    class_txt(args.root + '/test', args.root + '/test.txt', num_class=args.num_class)
    args.restore = True
    args.mode = 'inference'
    args.num_class = 100

    if args.mode == 'train':
        train()
    elif args.mode == 'validation':
        validation()
    elif args.mode == 'inference':
        inference()
















