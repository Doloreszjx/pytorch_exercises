#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter('logs')

img_path = 'data/train/ants/0013035.jpg'
img_PIL = Image.open(img_path)
img_np = np.array(img_PIL)

# 默认图片的形状是CHW - channel, height, weight
# 但是numpy的shape是HWC
# 因此需要手动设置dataformats为HWC
writer.add_image(tag='ants_img_1', img_tensor=img_np, global_step=2, dataformats='HWC')

# for i in range(100):
#     writer.add_scalar('y=2x', 2*i, i)

writer.close()