import os
import os.path as osp

import torch
from torch.utils.data import Dataset

import numpy as np

class VPDDataset(Dataset):
    def __init__(self, data_dir, desc="train", time_size=118):
        self.data_dir = os.path.join(data_dir, desc)
        self.label_dir = os.path.join(data_dir, "label")
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


def get_smote(train_dataset, smote_model):
    (_, prev_shape_1, prev_shape_2) = train_dataset.data[0].shape
    inputs, labels = np.array(train_dataset.data), np.array(train_dataset.labels)

    flatten_inputs = []
    for idx in range(inputs.shape[0]):
        flatten_inputs.append(inputs[idx, :].flatten())
    flatten_inputs = np.array(flatten_inputs)

    inputs_res, labels_res = smote_model.fit_resample(flatten_inputs, labels)
    
    reshaped_inputs = []
    for idx in range(inputs_res.shape[0]):
        reshaped_inputs.append(inputs_res[idx, :].reshape(1, prev_shape_1, prev_shape_2))
    reshaped_inputs = np.array(reshaped_inputs)
    labels_res = labels_res.reshape(-1, 1)

    train_dataset.labels = np.concatenate((train_dataset.labels, labels_res), axis=0)
    train_dataset.data = np.concatenate((train_dataset.data, reshaped_inputs), axis=0)

    return train_dataset
