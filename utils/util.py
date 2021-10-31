import argparse
import csv
import json
import os
import glob
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from model.modelloader import COVID_PVTv2, COVID_ViT




def read_filepaths(file, root):
    """Collect data from annotation files.
    """
    print('Collecting data from : {}'. format(file))

    paths, labels = [], []

    with open(file, 'r') as f:
        lines = f.read().splitlines()
        for idx, line in enumerate(lines):
            if len(line.split(' ')) == 3:
                _, path, label = line.split(' ')
            elif len(line.split(' ')) == 4:
                _, path, label, dataset = line.split(' ')
            elif len(line.split(' ')) == 5:
                _, _, path, label, dataset = line.split(' ')
            elif len(line.split(' ')) == 6:
                _, _, path1, path2, label, dataset = line.split(' ')
                path = path1 + path2

            label = label.lower()

            # Ignore duplicates filenames
            if path in paths:
                continue

            # Ignore not exists filenames
            img_path = root + path
            if not os.path.exists(img_path):
                continue

            paths.append(path)
            labels.append(label)

    print('Data collected! {}'. format(file))

    return paths, labels



def seeding(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
    SEED = config.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if (config.cuda):
        torch.cuda.manual_seed(SEED)



def one_hot_enc(target, batch_size, n_classes):
    """One hot encode annos.
    """
    y = torch.zeros(batch_size, n_classes, dtype=torch.float32)
    for yi in range(y.shape[0]):
        y[yi, target] = 1
    return y



def get_checkpoints(modelname):
    """List all of checkpoints.
    """
    list = glob.glob(f'../COVID_19/checkpoints/{modelname}/*')
    for i, x in enumerate(list):
        [print(i+1 ,':    ' ,x.split('\\')[-1])]



def optimizer_to_cuda(optimizer, device):
    """Moving optimizer to GPU after loading
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def augmentation2raw(config, inp, show=False):
    """Convert augmentated image to raw image.
    """
    if config.preprocess_type == 'torchio':
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.25, 0.25, 0.25])
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    if show == False:
        return inp
    else:
        plt.imshow(inp)
        plt.pause(0.001)



def get_model(config, model_name):
    if  model_name == 'model_ViT':
        return COVID_ViT(config)
    if  model_name == 'model_PVT_V2':
        return COVID_PVTv2(config)
    else:
        raise ValueError(f'Expected model_PVT_V2 or model_ViT, found {model_name}')
