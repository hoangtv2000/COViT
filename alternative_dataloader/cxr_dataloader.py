import os
import random
import torch
from torch.utils.data import Dataset
import pandas as pd
import glob
import numpy as np
from utils.util import read_filepaths, split_dataframe
from PIL import Image
from .image_aug import *
from torchvision import transforms


class CovidXRDataset(Dataset):
    def __init__(self, config, mode, get_full=True, resize4gradcam=False, p=0, n=0, reset_state=False):
        self.config = config
        self.get_full = get_full

        self.resize4gradcam = resize4gradcam

        self.root = self.config.dataset.input_data
        self.img_folder = self.config.dataset.img_folder

        self.class_dict = self.config.dataset.class_dict
        self.num_classes = self.config.dataset.num_classes

        self.img_size = self.config.dataset.img_size
        self.gradcam_img_size = self.config.dataset.gradcam_img_size

        self.mode = mode
        self.do_aug = False

        if reset_state:
            p = n = 0


        if self.mode == 'train' and self.get_full:
            self.paths, self.labels = read_filepaths(str(self.root + 'train_split.txt'), self.img_folder)
            self.do_aug = True


        elif self.mode == 'train' and self.get_full == False:
            PATHS, LABELS = read_filepaths(str(self.root + 'train_split.txt'), self.img_folder)
            train_data = {'paths': PATHS, 'labels': LABELS}
            train_df = pd.DataFrame(train_data)

            num_sample = len(train_df.loc[train_df['labels'] == 'covid-19'])

            # New approach: split them to folds and load sequentially.

            train_covid = train_df.loc[train_df['labels'] == 'covid-19']
            lst_train_pneu = split_dataframe(train_df.loc[train_df['labels'] == 'pneumonia'], chunk_size=num_sample)
            lst_train_normal = split_dataframe(train_df.loc[train_df['labels'] == 'normal'], chunk_size=num_sample)

            self.len_fold_pneu = len(lst_train_pneu)
            self.len_fold_norm = len(lst_train_normal)

            for i in range(self.len_fold_pneu):
                for j in range(self.len_fold_norm):
                    if (i == p and j == n):
                        concat_subpd = pd.concat([lst_train_normal[j], lst_train_pneu[i], train_covid])
                        self.paths, self.labels = concat_subpd['paths'].tolist(), concat_subpd['labels'].tolist()

            # Old approach: getting randomly n sample for each label.
            # sub_df = []

            # for class_ in train_df['labels'].unique():
            #     per_class_ = train_df.query("labels == @class_")
            #     sub_df.append(per_class_.sample(num_sample, replace=False, random_state=1))
            # train_df = pd.concat(sub_df, axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)
            # self.paths, self.labels = train_df['paths'].tolist(), train_df['labels'].tolist()

            self.do_aug = True


        elif self.mode == 'val':
            self.paths, self.labels = read_filepaths(str(self.root + 'val_split.txt'), self.img_folder)


        elif self.mode == 'test':
            self.paths, self.labels = read_filepaths(str(self.root + 'test_split.txt'), self.img_folder)


    def get_split_fold(self):
        return self.len_fold_pneu, self.len_fold_norm


    def __len__(self):
        return len(self.paths)


    def __getitem__(self, index):
        img_path = self.img_folder + self.paths[index]
        img_tensor = self.load_image(img_path)
        label_tensor = torch.tensor(self.class_dict[self.labels[index]], dtype=torch.long)

        return img_tensor, label_tensor


    def load_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        if self.resize4gradcam == True:
            img = img.resize(self.gradcam_img_size)
        else:
            img = img.resize(self.img_size)

        if self.config.preprocess_type == 'base':
            if self.do_aug:
                transform = base_train_transformation(self.img_size[0])
            else:
                transform = base_test_transformation()

        elif self.config.preprocess_type == 'autoaug':
            if self.do_aug:
                transform = autoaug_transformation(self.img_size[0])
            else:
                transform = base_test_transformation()

        elif self.config.preprocess_type == 'torchio':
            convert_tensor = transforms.ToTensor()
            img = convert_tensor(img).unsqueeze(3)
            if self.do_aug:
                transform = torchio_train_transformation()
            else:
                transform = torchio_test_transformation()
            return transform(img).squeeze(3)

        else:
            raise ValueError('You must set config one of three augmentation_type: "base" "autoaug" "torchio"!')

        return transform(img)
