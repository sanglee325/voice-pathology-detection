import os
import os.path as osp

import torch
from torch.utils.data import Dataset

import numpy as np

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

def get_smote_kfold(x_data, y_data, smote_model):
    x_data, y_data = x_data.cpu().detach().numpy(), y_data.cpu().detach().numpy()
    _, prev_shape_1, prev_shape_2 = x_data[0].shape
    flatten_inputs = []
    for idx in range(x_data.shape[0]):
        flatten_inputs.append(x_data[idx].flatten())
    flatten_inputs = np.array(flatten_inputs)

    inputs_res, labels_res = smote_model.fit_resample(flatten_inputs, y_data)
    
    reshaped_inputs = []
    for idx in range(inputs_res.shape[0]):
        reshaped_inputs.append(inputs_res[idx, :].reshape(1, prev_shape_1, prev_shape_2))
    reshaped_inputs = np.array(reshaped_inputs)
    labels_res = labels_res.reshape(-1, 1)

    x_smote = torch.from_numpy(np.concatenate((x_data, reshaped_inputs), axis=0))
    y_smote = torch.from_numpy(np.concatenate((y_data, labels_res), axis=0))
    smote_dataset = torch.utils.data.TensorDataset(x_smote, y_smote)

    return smote_dataset
