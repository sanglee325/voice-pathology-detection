import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as ttf
from ipywidgets import IntProgress

import os
import os.path as osp

from tqdm import tqdm
from PIL import Image
import numpy as np
import pdb

"""
The well-accepted SGD batch_size & lr combination for CNN classification is 256 batch size for 0.1 learning rate.
When changing batch size for SGD, follow the linear scaling rule - halving batch size -> halve learning rate, etc.
This is less theoretically supported for Adam, but in my experience, it's a decent ballpark estimate.
"""

time_size = 118;
batch_size = 32
lr = 0.001
epochs = 101 # Just for the early submission. We'd want you to train like 50 epochs for your main submissions.

#%% Network

class Network(nn.Module):
    
    def __init__(self, num_classes=1):
                  
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            torchvision.models.resnet18(pretrained=True)
            #torchvision.models.resnet18()
            )
        
        self.cls_layer = nn.Sequential(
            nn.Identity(),
            nn.Linear(1000, num_classes),
            nn.Sigmoid()
            )
        
        # self.backbone = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=7, stride=2, padding = (3, 3), bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
            
        #     nn.Conv2d(16, 64, kernel_size=5, stride=2, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
            
        #     nn.Conv2d(64, 256, kernel_size=5, stride=2, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
            
        #     nn.Conv2d(256, 256, kernel_size=5, stride=2, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.AdaptiveMaxPool2d(1),
        #     nn.Flatten()
        #     )
        
        # self.cls_layer = nn.Sequential(
        #     nn.Linear(256, num_classes),
        #     nn.Sigmoid()
        #     )
    
    def forward(self, x, return_feats=False):

        feats = self.backbone(x)
        out = self.cls_layer(feats)

        if return_feats:
            return feats
        else:
            return out

#%% Setup everything for training

model = Network()
model.cuda()

# %% Loading_Dataset

class Loading_Dataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.file_path_name = []
        
        for (root, directories, files) in os.walk(data_dir):
            for file in files:
                if '.npy' in file:
                    self.file_path_name.append(os.path.join(file))

    def __len__(self):
        return len(self.file_path_name)

    def __getitem__(self, idx):
        data = np.load(self.data_dir + '/' + self.file_path_name[idx])
        # label = np.zeros((2));
        # tmp_arg = np.load(self.label_dir + '/' + self.file_path_name[idx])
        # label[tmp_arg] = 1;
        label = np.load(self.label_dir + '/' + self.file_path_name[idx]).reshape((1))
        
        if data.shape[1] > time_size:
            data = data[:, :time_size, :]
        elif data.shape[1] < time_size:
            tmp = np.zeros((data.shape[0], time_size - data.shape[1], data.shape[2]))            
            data = np.concatenate((data, tmp), axis = 1)
        
        # data = data.reshape(data.shape[0], data.shape[2], data.shape[1])
        # data = data/data.mean(axis=1)
        # data = data.reshape(data.shape[0], data.shape[2], data.shape[1])
        return data, label

# class Loading_Dataset_test(Dataset):
#     def __init__(self, data_dir, transform=None, target_transform=None):
#         self.data_dir = data_dir
#         self.file_path_name = []
        
#         for (root, directories, files) in os.walk(data_dir):
#             for file in files:
#                 if '.npy' in file:
#                     self.file_path_name.append(os.path.join(file))

#     def __len__(self):
#         return len(self.file_path_name)

#     def __getitem__(self, idx):
#         data = np.load(self.data_dir + '/' + self.file_path_name[idx])
#         return data

#%% Dataset & DataLoader

DATA_DIR = "C:/Users/DTLab/Desktop/CMU_Class/Large scale multimedia analysis [LSMA]/Project/features"
TRAIN_DIR = osp.join(DATA_DIR + "/train") 
VAL_DIR = osp.join(DATA_DIR + "/val")
TEST_DIR = osp.join(DATA_DIR + "/test")
LABEL_DIR = osp.join(DATA_DIR + "/label")

train_dataset = Loading_Dataset(TRAIN_DIR, LABEL_DIR)
val_dataset = Loading_Dataset(VAL_DIR, LABEL_DIR)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

#%% Setup everything for training

model = Network()
model.cuda()

num_trainable_parameters = 0
for p in model.parameters():
    num_trainable_parameters += p.numel()
print("Number of Params: {}".format(num_trainable_parameters))

# TODO: What criterion do we use for this task?
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * epochs))

scaler = torch.cuda.amp.GradScaler()

#%% Let's train!

for epoch in range(epochs):
    # Quality of life tip: leave=False and position=0 are needed to make tqdm usable in jupyter
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 

    num_correct = 0
    total_loss = 0

    for i, (x, y) in enumerate(train_loader):
        
        optimizer.zero_grad()

        x = x.cuda()
        y = y.cuda()

        # Don't be surprised - we just wrap these two lines to make it work for FP16
        with torch.cuda.amp.autocast():     
            outputs = model(x.float())
            # pdb.set_trace();
            loss = criterion(outputs.float(), y.float())

        # Update # correct & loss as we go
        # print(outputs)
        num_correct += int((torch.round(outputs.float()) == torch.round(y.float())).sum())
        total_loss += float(loss)

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / ((i + 1) * batch_size)),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct,
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
        
        # Another couple things you need for FP16. 
        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() # This is something added just for FP16

        scheduler.step() # We told scheduler T_max that we'd call step() (len(train_loader) * epochs) many times.

        batch_bar.update() # Update tqdm bar
    batch_bar.close() # You need this to close the tqdm bar

    # You can add validation per-epoch here if you would like

    print("Epoch {}/{}: Train Acc {:.04f}%, Train Loss {:.04f}, Learning Rate {:.04f}".format(
        epoch + 1,
        epochs,
        100 * num_correct / (len(train_loader) * batch_size),
        float(total_loss / len(train_loader)),
        float(optimizer.param_groups[0]['lr'])))
    
    if epoch % 10 == 0:
        torch.save(model.state_dict(), './model' + str(epoch) + '_2.pth')
        
        # Validation per 10 epochs
        
        model.eval()
        batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')
        num_correct = 0
        for i, (x_val, y_val) in enumerate(val_loader):

            x_val = x_val.cuda()
            y_val = y_val.cuda()

            with torch.no_grad():
                outputs = model(x_val.float())
            num_correct += int((torch.round(outputs.float()) == torch.round(y_val.float())).sum())
            batch_bar.set_postfix(acc="{:.04f}%".format(100 * num_correct / ((i + 1) * batch_size)))

            batch_bar.update()
            
        batch_bar.close()
        print("\n\nValidation: {:.04f}%".format(100 * num_correct / len(val_dataset)))
        print("\n\n")

# #%% Classification Task: Test

# test_dataset = Loading_Dataset_test(TEST_DIR)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)#, num_workers=1)


# model.eval()
# batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, position=0, leave=False, desc='Test')

# res = []
# for i, (x) in enumerate(test_loader):

#     x = x.cuda()

#     with torch.no_grad():
#         outputs = model(x)
#         pred = torch.argmax(outputs, axis=1)
#         res.extend(pred.tolist())

#     batch_bar.update()
    
# batch_bar.close()

# with open("submission.csv", "w+") as f:
#     f.write("id,label\n")
#     for i in range(len(test_dataset)):
#         f.write("{},{}\n".format(str(i).zfill(6) + ".jpg", res[i]))