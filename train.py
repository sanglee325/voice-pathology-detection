import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as ttf

import os
import os.path as osp
import argparse

from tqdm import tqdm
import numpy as np
import pdb
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 

from models.cnn import BinaryResNet18
from data_loader import VPDDataset, get_smote
from utils import set_reproducibility, set_logpath, save_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
if torch.cuda.is_available():
    N_GPUS = torch.cuda.device_count()
else:
    N_GPUS = 0

def parse_args():
    # model seting options
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=50, type=int,
                        help='total epochs to run')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
    
    # save path
    parser.add_argument('--name', default='0', type=str, help='name of run')
    parser.add_argument('--log_path', default="./logs/", type=str,
                        help='path for results')
    parser.add_argument('--data_path', default="./features/", type=str,
                        help='path for results')

    parser.add_argument('--smote', '-s', action='store_true', help='oversampling')

    return parser.parse_args()

def train(args, epoch, model, train_loader, optimizer, criterion, scaler, scheduler):
    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 

    num_correct = 0
    total_loss = 0
    total = 0

    for i, (x, y) in enumerate(train_loader):
        
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)

        with torch.cuda.amp.autocast():     
            outputs = model(x.float())
            loss = criterion(outputs.float(), y.float())

        num_correct += int((torch.round(outputs.float()) == torch.round(y.float())).sum())
        total += len(x)
        total_loss += float(loss)

        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / ((i + 1) * args.batch_size)),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct,
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer) 
        scaler.update()

        scheduler.step()

        batch_bar.update()
        batch_bar.close()

        train_acc = 100 * num_correct / total
        train_loss = float(total_loss / total)
        lr_rate = float(optimizer.param_groups[0]['lr'])

    print("Epoch {}/{}: Train Acc {:.04f}%, Train Loss {:.04f}, Learning Rate {:.04f}".format(
                                        epoch + 1, args.epochs, train_acc, train_loss, lr_rate))
    
def validate(args, model, val_loader):
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Validation')

    num_correct = 0
    total = 0
    for i, (x, y) in enumerate(val_loader):

        x, y = x.to(device, dtype=torch.float), y.to(device, dtype=torch.float)

        with torch.no_grad():
            outputs = model(x)

        num_correct += int((torch.round(outputs.float()) == torch.round(y.float())).sum())
        total += len(x)
        batch_bar.set_postfix(acc="{:.04f}%".format(100 * num_correct / ((i + 1) * args.batch_size)))

        batch_bar.update()
        
    batch_bar.close()

    val_acc = 100 * num_correct / total
    print("Validation: {:.04f}%".format(val_acc))

    return val_acc


if __name__ == '__main__':
    args = parse_args()
    set_reproducibility(args, args.seed)

    
    logpath = args.log_path
    logfile_base = f"{args.name}_S{args.seed}_B{args.batch_size}_LR{args.lr}_E{args.epochs}"
    logdir = logpath + logfile_base

    set_logpath(logpath, logfile_base)
    print('save path: ', logdir)

    time_size = 118

    model = BinaryResNet18(pretrained=True)
    model.to(device)

    DATA_DIR = args.data_path
    train_dataset = VPDDataset(DATA_DIR, desc="train", time_size=time_size)
    val_dataset = VPDDataset(DATA_DIR, desc="val", time_size=time_size)

    if args.smote:
        smote_model = SMOTE(random_state=args.seed, k_neighbors=5)
        smote_dataset = get_smote(train_dataset, smote_model)
        train_loader = DataLoader(smote_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    num_trainable_parameters = 0
    for p in model.parameters():
        num_trainable_parameters += p.numel()
    print("Number of Params: {}".format(num_trainable_parameters))

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * args.epochs))

    scaler = torch.cuda.amp.GradScaler()

    BEST_VAL = 0
    best_model = model

    for epoch in range(args.epochs):
        train(args, epoch, model, train_loader, optimizer, criterion, scaler, scheduler)
        val_acc = validate(args, model, val_loader)
        
        if BEST_VAL <= val_acc:
            save_checkpoint(args, val_acc, model, optimizer, epoch, logdir)
            best_model = model
            BEST_VAL = val_acc