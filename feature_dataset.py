import os
import os.path as osp

import torch
from torch.utils.data import Dataset

import numpy as np


class WAVLMDataset(Dataset):
    def __init__(self, data_dir, desc="train", time_size=118):
        self.data_dir = os.path.join(data_dir, "WAVLM", desc)
        self.label_dir = os.path.join(data_dir, "WAVLM", "label")
        self.time_size = time_size
        self.file_path_name = os.listdir(self.data_dir)
        self.data, self.labels = [], []

        for idx in range(len(self.file_path_name)):
            data = np.load(self.data_dir + '/' + self.file_path_name[idx])
            label = np.load(self.label_dir + '/' + self.file_path_name[idx]).reshape((1))
            
            if data.shape[1] > self.time_size:
                data = data[:, :self.time_size, :]
            elif data.shape[1] < self.time_size:
                tmp = np.zeros((data.shape[0], self.time_size - data.shape[1], data.shape[2]))            
                data = np.concatenate((data, tmp), axis=1)
        
            self.data.append(data)
            self.labels.append(label)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.file_path_name)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]
        
        return data, label

class STFTDataset(Dataset):
    def __init__(self, data_dir, desc="train", time_size=596):
        self.data_dir = os.path.join(data_dir, "STFT", desc)
        self.label_dir = os.path.join(data_dir, "STFT", "label")
        self.time_size = time_size
        self.file_path_name = os.listdir(self.data_dir)
        self.data, self.labels = [], []

        for idx in range(len(self.file_path_name)):
            data = np.load(self.data_dir + '/' + self.file_path_name[idx]).swapaxes(1,2)
            label = np.load(self.label_dir + '/' + self.file_path_name[idx]).reshape((1))
            
            if data.shape[1] > self.time_size:
                data = data[:, :self.time_size, :]
            elif data.shape[1] < self.time_size:
                tmp = np.zeros((data.shape[0], self.time_size - data.shape[1], data.shape[2]))            
                data = np.concatenate((data, tmp), axis=1)
        
            self.data.append(data)
            self.labels.append(label)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.file_path_name)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]
        
        return data, label

class W2VDataset(Dataset):
    def __init__(self, data_dir, desc="train", time_size=236):
        self.data_dir = os.path.join(data_dir, "W2VL", desc)
        self.label_dir = os.path.join(data_dir, "W2VL", "label")
        self.time_size = time_size
        self.file_path_name = os.listdir(self.data_dir)
        self.data, self.labels = [], []

        for idx in range(len(self.file_path_name)):
            data = np.load(self.data_dir + '/' + self.file_path_name[idx]).swapaxes(1,2)
            label = np.load(self.label_dir + '/' + self.file_path_name[idx]).reshape((1))
            
            if data.shape[1] > self.time_size:
                data = data[:, :self.time_size, :]
            elif data.shape[1] < self.time_size:
                tmp = np.zeros((data.shape[0], self.time_size - data.shape[1], data.shape[2]))            
                data = np.concatenate((data, tmp), axis=1)
        
            self.data.append(data)
            self.labels.append(label)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.file_path_name)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]
        
        return data, label
