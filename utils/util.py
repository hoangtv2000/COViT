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

"""    def predict(self, epoch):
        self.model.eval()

        predictions = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                data = data.to(self.device)
                logits = self.model(data, None)

                maxes, prediction = torch.max(logits, 1)  # get the index of the max log-probability
                # log.info()
                predictions.append(f"{target[0]},{prediction.cpu().numpy()[0]}")

        pred_name = os.path.join(self.checkpoint_dir, f'validation_predictions_epoch_{epoch:d}_.csv')

        # Write prediction to .csv
        with open(pred_name, 'w') as fout:
            for item in predictions:
                # print(item)
                fout.write(item)
                fout.write('\n')

        return predictions"""

def write_score(writer, iter, mode, metrics):
    writer.add_scalar(mode + '/loss', metrics.data['loss'], iter)
    writer.add_scalar(mode + '/acc', metrics.data['correct'] / metrics.data['total'], iter)


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
    y = torch.zeros(batch_size, n_classes, dtype=torch.long)
    for yi in range(y.shape[0]):
        y[yi, target] = 1
    return y



def get_checkpoints(modelname):
    """List all of checkpoints.
    """
    list = glob.glob(f'../COVID_19/checkpoints/model_{modelname}/*')
    for i, x in enumerate(list):
        [print(i+1 ,':    ' ,x.split('\\')[-1])]



def optimizer_to_cuda(optimizer, device):
    """Moving optimizer to GPU after loading
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
