#!/usr/bin/python
# -*- coding:utf-8 -*-

# 为什么需要Tensor数据类型： transforms.ToTensor()为我们封装了很多模型训练需要的参数和方法
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img_path = 'data/train/ants/0013035.jpg'
img = Image.open(img_path)
# 创建需要的工具
myTensor = transforms.ToTensor()
img_tensor = myTensor(img)

writers = SummaryWriter('logs')
writers.add_image('img_tensor', img_tensor)

writers.close()

print(img_tensor.shape)
print(img_tensor)