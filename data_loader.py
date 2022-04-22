import os
import os.path as osp

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

class VPDDataset(Dataset):
    def __init__(self, data_dir, desc="train", time_size=118):
        self.data_dir = os.path.join(data_dir, desc)
        self.label_dir = os.path.join(data_dir, "label")
        self.time_size = time_size
        self.file_path_name = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.file_path_name)

    def __getitem__(self, idx):
        data = np.load(self.data_dir + '/' + self.file_path_name[idx])
        label = np.load(self.label_dir + '/' + self.file_path_name[idx]).reshape((1))
        
        if data.shape[1] > self.time_size:
            data = data[:, :self.time_size, :]
        elif data.shape[1] < self.time_size:
            tmp = np.zeros((data.shape[0], self.time_size - data.shape[1], data.shape[2]))            
            data = np.concatenate((data, tmp), axis = 1)
        
        return data, label