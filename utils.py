import os
import shutil
import random

import torch
import torch.nn as nn

import numpy as np


def set_reproducibility(args, seed):
    if args.seed is None:
        seed = np.random.randint(10000)
        args.seed = seed

    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(seed)
    else:    
        torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    return seed

def set_logpath(dirpath, fn):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    logdir = dirpath + fn
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if len(os.listdir(logdir)) != 0:
        ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
                        "Will you proceed [y/N]? ")
        if ans in ['y', 'Y']:
            shutil.rmtree(logdir)
        else:
            exit(1)

def save_checkpoint(args, acc, model, optim, epoch, logdir, index=False):
    # Save checkpoint.
    print('Saving..')

    if isinstance(model, nn.DataParallel):
        model = model.module

    state = {
        'net': model.state_dict(),
        'optimizer': optim.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }

    if index:
        ckpt_name = 'ckpt_epoch' + str(epoch) + '_' + str(args.seed) + '.pth'
    else:
        ckpt_name = 'ckpt_' + str(args.seed) + '.pth'

    ckpt_path = os.path.join(logdir, ckpt_name)
    torch.save(state, ckpt_path)
