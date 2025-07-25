#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

class DataPrepare(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.img_path = os.path.join(self.root_dir, self.label_dir)
        self.img_path_list = os.listdir(self.img_path)
        print(self.img_path_list)

    def __getitem__(self, index):
        img_name = self.img_path_list[index]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir

        return img, label
    def __len__(self):
        return len(self.img_path_list)

root_dir = 'data/train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'

ants_data = DataPrepare(root_dir, ants_label_dir)
bees_data = DataPrepare(root_dir, bees_label_dir)

