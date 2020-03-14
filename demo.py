# -*- coding:utf-8 -*-
import sys
import numpy as np
# import matplotlib.pyplot as plt
import torch.nn as nn
import models
import torch
from torchvision import datasets, models, transforms
import torchvision
import config
from PIL import Image

# 模型搭建
pthfile = 'trained_models2/interface-fault_25.pth'
model = torch.load(pthfile)

# 数据预处理
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

batch_size = config.BATCH_SIZE
num_classes = config.NUM_CLASSES

# 创建数据集
valid_directory = config.VALID_DATASET_DIR
valid_datasets = datasets.ImageFolder(valid_directory,transform=data_transform)
valid_data_size = len(valid_datasets)
valid_data = torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)

classes = ['falut','interface']
def prediect(img_path):
    device = torch.device("cpu")
    net = model.to(device)
    torch.no_grad()
    img = Image.open(img_path)
    img.show()
    img_ = data_transform(img).unsqueeze(0)
    img_ = img_.to(device)
    outputs = net(img_)
    _, predicted = torch.max(outputs, 1)
    print('this picture maybe :',classes[predicted[0]])

if __name__=='__main__':
    img_path = 'data2/val/fault/2-5-43-normal stress.bmp'
    prediect(img_path)
