"""File to load dataset.
"""

from torch.utils.data import DataLoader
from .cxr_dataloader import CovidXRDataset
import pandas as pd


def present_dataset(config):
    """Present all subset from dataset.
    """
    train_loader = CovidXRDataset(config = config, mode='train', get_full=True)
    val_loader = CovidXRDataset(config = config, mode='val')
    test_loader = CovidXRDataset(config = config, mode='test')

    train_data = {'paths': train_loader.paths, 'labels': train_loader.labels}
    train_df = pd.DataFrame(train_data)
    num_samples = len(train_df.loc[train_df['labels'] == 'covid-19'])

    val_data = {'paths': val_loader.paths, 'labels': val_loader.labels}
    val_df = pd.DataFrame(val_data)

    test_data = {'paths': test_loader.paths, 'labels': test_loader.labels}
    test_df = pd.DataFrame(test_data)

    print('Name of the dataset:', config.dataset.name)
    print('Collected from the description of these github: https://github.com/lindawangg/COVID-Net'\
          , end='\n{}\n'.format('-'*50))

    print('Labels and quantities of samples in train dataset: \n ', train_df['labels'].value_counts())
    print(f'For each epoch we randomly get {num_samples} samples for each class in train dataset', end='\n{}\n'.format('-'*50))

    print('Labels and quantities of samples in val dataset: \n ', val_df['labels'].value_counts(), end='\n{}\n'.format('-'*50))

    print('Labels and quantities of samples in test dataset: \n ', test_df['labels'].value_counts(), end='\n{}\n'.format('-'*50))


def get_balance_train_data(config):
    train_params = {'batch_size': config.dataloader.train.batch_size,
                    'shuffle': True,
                    'num_workers': config.dataloader.train.num_workers,
                    'pin_memory': True}

    train_loader = CovidXRDataset(config = config, mode='train', get_full=False)
    train_pneufold, train_normfold = train_loader.get_split_fold()

    train_generator = DataLoader(train_loader, **train_params)

    return train_generator, train_pneufold, train_normfold



def get_val_data(config):
    val_params = {'batch_size': config.dataloader.val.batch_size,
                  'shuffle': True,
                  'num_workers': config.dataloader.val.num_workers,
                  'pin_memory': True}

    val_loader = CovidXRDataset(config = config, mode='val')
    val_generator = DataLoader(val_loader, **val_params)

    return val_generator



def get_test_data(config, resize4gradcam=False):
    test_params = {'batch_size': config.dataloader.test.batch_size,
                   'shuffle': True,
                   'num_workers': config.dataloader.test.num_workers}

    test_loader = CovidXRDataset(config = config, mode='test', resize4gradcam=resize4gradcam)
    test_generator = DataLoader(test_loader, **test_params)

    return test_generator
