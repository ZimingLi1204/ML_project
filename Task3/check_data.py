import sys

sys.path.append("..")
import torch.nn as nn
import torch
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np
import os
import scipy.io as scio
import gc
from tqdm.auto import tqdm
from utils.utils import get_promt

import os
import matplotlib.pyplot as plt

class Maskdataset(Dataset):

    def __init__(self,  img: np.array, mask: np.array, name: list, slice_id: list, category: list):
        '''
        img: N * 512 * 512.
        mask: 每个data的groundtruth 0-1mask, N * 512 * 512
        name: 每个data对应的CT编号 1-10, 21-40   list中每个元素格式为 str(00xx) 如 "0001"      N * 1
        slice_id: 每个data对应的切片编号 每个CT有80-150个切片    list中每个元素格式为 str(xxx)    N * 1
        category: 每个mask对应的类别 范围为1-13     N * 1
        '''
        self.img = img
        self.mask = mask
        self.name = name
        self.slice_id = slice_id
        self.category = category.astype(np.int64) - 1
        
        
        # N = self.img.shape[0]
        # self.mean = np.mean(self.img, axis=0)
        # self.std = np.sqrt(np.sum(((self.img - self.mean)**2), axis=0))
        # print(self.mean.shape, self.std.shape)
        # print(self.mean.reshape(1, 512, 512).repeat(N, axis=0).shape)
        # self.img = (self.img-self.mean.reshape(1, 512, 512).repeat(N, axis=0)) / self.std.reshape(1, 512, 512).repeat(N, axis=0)
        
        
        #self.data_merge = np.concatenate([self.mask.reshape(-1, 1, 512, 512), self.img.reshape(-1, 1, 512, 512)], axis=1)
        self.data_merge = np.multiply(self.img, self.mask)
        self.data_merge = self.data_merge.reshape(-1, 1, 512, 512)
        print(self.data_merge.shape)
        # breakpoint()
        
                
        
    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, index):

        img = torch.from_numpy(self.img[index]).float()
        mask = torch.from_numpy(self.mask[index]).float()
        promt = get_promt(self.img[index], self.mask[index], promt_type='box').astype(np.float32)
        data_merge = self.data_merge[index]
        category = self.category[0][index]
        return data_merge, category, promt, img, mask


writer = SummaryWriter(log_dir="log")
img = np.load("../BTCV_dataset1/pre_processed_dataset1_train.npz")["img"]
mask = np.load("../BTCV_dataset1/pre_processed_dataset1_train.npz")["mask"]
category = scio.loadmat("../BTCV_dataset1/pre_processed_dataset1_train.mat")["category"]
slice_id = scio.loadmat("../BTCV_dataset1/pre_processed_dataset1_train.mat")["slice_id"]
name = scio.loadmat("../BTCV_dataset1/pre_processed_dataset1_train.mat")["name"]

dataset = Maskdataset(img=img, mask=mask, name=name, slice_id=slice_id,
                                category=category)

img = torch.from_numpy(img_data[0, :, :]).float()
plt.imshow(img)
plt.show()
plt.imsave("q1.jpg", img)
breakpoint()

for iter in range(len(img_data)):
    img = img_data[iter, :, :]
    print(img.shape)
    writer.add_image("raw_img", img, iter, dataformats="HW")
