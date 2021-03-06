import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import torchvision
import torchvision.transforms as ttf

import os
import os.path as osp
import argparse

from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE 

from models.cnn import CNNNetwork
from models.resnet import BinaryResNet18
from models.lstm import LSTMNetwork
from data_loader import get_smote_kfold
from feature_dataset import WAVLMDataset, WVLMLDataset, W2VDataset, STFTDataset, MFCC13Dataset, MFCC40Dataset
from utils import set_reproducibility, set_logpath, save_checkpoint
from utils import Logger

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
    parser.add_argument('--epochs', default=20, type=int,
                        help='total epochs to run')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--arch', default='resnet18', type=str, help='Model Architecture')
    
    # save path
    parser.add_argument('--name', default='0', type=str, help='name of run')
    parser.add_argument('--log_path', default="./logs/", type=str,
                        help='path for results')
    parser.add_argument('--data_path', default="./data/", type=str,
                        help='path for results')

    parser.add_argument('--smote', '-s', action='store_true', help='oversampling')
    parser.add_argument('--extractor', default="WAVLM", type=str,
                        help='Feature extractor type')
    parser.add_argument('--pretrained', '-p', action='store_true', help='pretrained resnet')

    return parser.parse_args()

def train_epoch(args, epoch, model, train_loader, optimizer, criterion, scaler, scheduler, logger=None):
    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 

    num_correct = 0
    total_loss = 0
    total = 0

    for i, (x, y) in enumerate(train_loader):
        
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)

        if args.arch == 'lstm':
            x = x.squeeze()

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

    msg = "Epoch {}/{}: Train Acc {:.04f}%, Train Loss {:.04f}, Learning Rate {:.04f}".format(
                                        epoch + 1, args.epochs, train_acc, train_loss, lr_rate)
    if logger:
        logger.log(msg)
    else:
        print(msg)

    

def train(args, x_data, y_data, logger=None):
    total_acc = 0
    kf = KFold(n_splits=10)

    for fold, (train_index, test_index) in enumerate(kf.split(x_data, y_data)):
        ### Dividing data into folds
        x_train_fold = x_data[train_index]
        x_test_fold = x_data[test_index]
        y_train_fold = y_data[train_index]
        y_test_fold = y_data[test_index]

        train_dataset = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        val_dataset = torch.utils.data.TensorDataset(x_test_fold, y_test_fold)
        
        if args.smote:
            smote_model = SMOTE(random_state=args.seed, k_neighbors=5)
            smote_dataset = get_smote_kfold(x_train_fold, y_train_fold, smote_model)
            train_loader = DataLoader(smote_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,shuffle = False)

        if args.arch == 'resnet18':
            model = BinaryResNet18(pretrained=args.pretrained)
        elif args.arch == 'cnn':
            model = CNNNetwork()
        elif args.arch == 'lstm':
            _, _, input_size = x_data[0].shape
            model = LSTMNetwork(input_size=input_size, time_size=time_size)
        model.to(device)
        num_trainable_parameters = 0
        for p in model.parameters():
            num_trainable_parameters += p.numel()
        logger.log("Number of Params: {}".format(num_trainable_parameters))

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * args.epochs))

        scaler = torch.cuda.amp.GradScaler()

        logger.log('Fold number {} / {}'.format(fold + 1 , kf.get_n_splits()))

        BEST_VAL = 0
        best_model = model
        for epoch in range(args.epochs):
            train_epoch(args, epoch, model, train_loader, optimizer, criterion, scaler, scheduler, logger=None)
            val_acc = validate(args, model, val_loader, logger)
        
            if BEST_VAL <= val_acc:
                save_checkpoint(args, val_acc, model, optimizer, epoch, logger.logdir, index=fold)
                best_model = model
                BEST_VAL = val_acc

        total_acc += BEST_VAL

    total_acc = (total_acc / kf.get_n_splits())

    return total_acc
    
def validate(args, model, val_loader, logger=None):
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Validation')

    num_correct = 0
    total = 0
    for i, (x, y) in enumerate(val_loader):

        x, y = x.to(device, dtype=torch.float), y.to(device, dtype=torch.float)

        if args.arch == 'lstm':
            x = x.squeeze()

        with torch.no_grad():
            outputs = model(x)

        num_correct += int((torch.round(outputs.float()) == torch.round(y.float())).sum())
        total += len(x)
        batch_bar.set_postfix(acc="{:.04f}%".format(100 * num_correct / ((i + 1) * args.batch_size)))

        batch_bar.update()
        
    batch_bar.close()

    val_acc = 100 * num_correct / total

    msg = "Validation: {:.04f}%".format(val_acc)
    if logger:
        logger.log(msg)
    else:
        print(msg)

    return val_acc


if __name__ == '__main__':
    args = parse_args()
    set_reproducibility(args, args.seed)

    
    logpath = args.log_path
    logfile_base = f"{args.name}_{args.arch}_S{args.seed}_B{args.batch_size}_LR{args.lr}_E{args.epochs}"

    logname = logfile_base
    logger = Logger(logname, logpath)
    logdir = logger.logdir

    print('save path: ', logdir)

    DATA_DIR = args.data_path

    if args.extractor == "WAVLM":
        time_size = 118
        train_dataset = WAVLMDataset(DATA_DIR, desc="train", time_size=time_size)
        val_dataset = WAVLMDataset(DATA_DIR, desc="val", time_size=time_size)
    elif args.extractor == "WVLML":
        time_size = 118
        train_dataset = WVLMLDataset(DATA_DIR, desc="train", time_size=time_size)
        val_dataset = WVLMLDataset(DATA_DIR, desc="val", time_size=time_size)
    elif args.extractor == "STFT":
        time_size = 596
        train_dataset = STFTDataset(DATA_DIR, desc="train", time_size=time_size)
        val_dataset = STFTDataset(DATA_DIR, desc="val", time_size=time_size)
    elif args.extractor == "W2VL":
        time_size = 236
        train_dataset = W2VDataset(DATA_DIR, desc="train", time_size=time_size)
        val_dataset = W2VDataset(DATA_DIR, desc="val", time_size=time_size)
    elif args.extractor == "mfcc13":
        time_size = 475
        train_dataset = MFCC13Dataset(DATA_DIR, desc="train", time_size=time_size)
        val_dataset = MFCC13Dataset(DATA_DIR, desc="val", time_size=time_size)
    elif args.extractor == "mfcc40":
        time_size = 475
        train_dataset = MFCC40Dataset(DATA_DIR, desc="train", time_size=time_size)
        val_dataset = MFCC40Dataset(DATA_DIR, desc="val", time_size=time_size)

    x_data = torch.from_numpy(np.concatenate((train_dataset.data, val_dataset.data)))
    y_data = torch.from_numpy(np.concatenate((train_dataset.labels, val_dataset.labels)))

    total_acc = train(args, x_data, y_data, logger)

    logger.log('Total accuracy 10 fold cross validation: {:.3f}%'.format(total_acc))