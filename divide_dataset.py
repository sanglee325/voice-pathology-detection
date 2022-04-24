import os
import argparse
import shutil

import numpy as np


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--path', default="./features/", type=str,
                        help='path for results')
args = parser.parse_args()

file_list = os.listdir(args.path)
filenames = [file for file in file_list if file.endswith(".npy")]
filenames.sort()

train_dir = os.path.join(args.path, 'train')
val_dir = os.path.join(args.path, 'val')
label_dir = os.path.join(args.path, 'label')

if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(val_dir):
    os.mkdir(val_dir)

idx_0, idx_1 = 0, 0
idx_train, idx_val = [], []
for idx, (filename) in enumerate(filenames):
    label_file = os.path.join(label_dir, filename)
    label = np.load(label_file)
    if label == 0 and idx_0 < 10:
        idx_0 += 1
        idx_val.append(idx)
    if label == 1 and idx_1 < 10:
        idx_1 += 1
        idx_val.append(idx)
    

# 0 is healthy
print(idx_0, idx_1)
print(len(idx_val))

for idx, (filename) in enumerate(filenames):
    if idx in idx_val:
        src = os.path.join(args.path, filename)
        dest = os.path.join(val_dir, filename)
    else:
        src = os.path.join(args.path, filename)
        dest = os.path.join(train_dir, filename)
    shutil.move(src, dest)
