"""Adapted: https://github.com/kohpangwei/group_DRO/blob/master/data/celebA_dataset.py"""
from pathlib import Path
import argparse
import os

import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensor
from tabulate import tabulate
from PIL import Image
import jpeg4py as jpeg
import pandas as pd
import numpy as np
import albumentations
import torch
import datetime


class CelebADataset(Dataset):
    """
    Source
    CelebA dataset (already cropped and centered).
    Note: idx and filenames are off by one.
    """

    def __init__(self, split, root_dir, target_name, confounder_names,
                  augment_data=False, target_resolution=None, loading_type='PIL',
                 skew_group_ids=[], crop_type=0):
        self.root_dir = Path(root_dir) / 'celeba'
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.augment_data = augment_data
        self.loading_type = loading_type
        self.crop_type = crop_type
        self.image_shape = (3, target_resolution, target_resolution)
        self.split = split

        # Read in attributes
        self.attrs_df = pd.read_csv(
            os.path.join(self.root_dir, 'list_attr_celeba.csv'))

        # Read in train/val/test splits
        self.split_df = pd.read_csv(
            os.path.join(self.root_dir, 'list_eval_partition.csv'))

        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        # Select split
        self.attrs_df = self.attrs_df[self.split_df['partition'] == self.split_dict[split]]
        self.split_array = self.split_df['partition'].values

        # Split out filenames and attribute names
        self.data_dir = self.root_dir / 'img_align_celeba'
        self.filename_array = self.attrs_df['image_id'].values
        self.attrs_df = self.attrs_df.drop(labels='image_id', axis='columns')
        self.attr_names = self.attrs_df.columns.copy()

        # Then cast attributes to numpy array and set them to 0 and 1
        # (originally, they're -1 and 1)
        self.attrs_df = self.attrs_df.values
        self.attrs_df[self.attrs_df == -1] = 0

        # Get the y values
        target_idx = self.attr_idx(self.target_name)
        self.y_array = self.attrs_df[:, target_idx]
        self.n_classes = 2

        # Map the confounder attributes to a number 0,...,2^|confounder_idx|-1
        self.confounder_idx = [self.attr_idx(a) for a in self.confounder_names]
        self.n_confounders = len(self.confounder_idx)
        confounders = self.attrs_df[:, self.confounder_idx]
        confounder_id = confounders @ np.power(2, np.arange(len(self.confounder_idx)))

        self.confounder_array = confounder_id

        self.n_groups = self.n_classes * pow(2, len(self.confounder_idx))
        self.group_ids = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')

        self.group_counts, bin_edges = np.histogram(self.group_ids, bins=range(self.n_groups+1), density=False)
        self.group_dist, bin_edges = np.histogram(self.group_ids, bins=range(self.n_groups+1), density=True)

        if np.sum(self.group_dist) != 1:
            raise ValueError

        ########################
        #### Dataset stats ####
        #######################
        self.group_stats = np.zeros((self.n_groups, 3))
        self.group_stats[:, 0] = self.group_counts
        self.group_stats[:, 1] = self.group_dist
        for group_id in range(self.n_groups):
            indices = np.nonzero(np.asarray(self.group_ids == group_id))[0]
            self.group_stats[group_id, 2] = np.mean(self.y_array[indices])

        self.df_stats = pd.DataFrame(self.group_stats, columns=['n', 'frac', 'class_balance'])
        self.df_stats['group_id'] = self.df_stats.index
        self.df_stats['binary'] = self.df_stats['group_id'].apply(lambda x: '{0:b}'.format(x).zfill(int(np.log(self.n_groups) + 1)))

        print("Num examples", len(self.y_array))
        print("Class balance: ", np.mean(self.y_array))

        # Print dataset stats
        print(tabulate(self.df_stats, headers='keys', tablefmt='psql'))
        self.transform = self.get_transform(target_resolution=target_resolution, crop_type=crop_type)

    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, index):

        # Load image
        img_filename = self.filename_array[index]
        img_filepath = self.data_dir / img_filename

        if self.loading_type == 'PIL':
            img = Image.open(img_filepath)
            img = np.array(img)
        elif self.loading_type == 'jpeg': # This was used for experiments in paper.
            img = jpeg.JPEG(str(img_filepath)).decode()

        # Transform
        img = self.transform(**{"image": img})['image']

        # Get label and group id
        y = torch.tensor(self.y_array[index], dtype=torch.long)
        group_id = torch.tensor(self.group_ids[index], dtype=torch.long)

        return img, y, group_id

    def get_transform(self, target_resolution, crop_type):
        """Transforms based on Group DRO paper

            Note: Switched to albumentations, rather than torchvision.transforms, are used to speed up transformations"""

        CROP_RESIZE = 0
        CROP_NORESIZE = 2

        orig_w = 178
        orig_h = 218
        orig_min_dim = int(min(orig_w, orig_h))

        if crop_type == CROP_RESIZE:
            transform = albumentations.Compose([
                                    albumentations.CenterCrop(orig_min_dim,orig_min_dim, always_apply=True),
                                    albumentations.Resize(height=target_resolution, width=target_resolution,
                                                          interpolation=1, p=1, always_apply=True),
                                    albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                                             max_pixel_value=255, p=1.0, always_apply=True),
                                    ToTensor()])
        # Used for experiments without pretraining. In this setup there is no resizing.
        if crop_type == CROP_NORESIZE:
            transform = albumentations.Compose([
                                    albumentations.CenterCrop(orig_min_dim,orig_min_dim, always_apply=True),
                                    albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                                             max_pixel_value=255, p=1.0, always_apply=True),
                                    ToTensor()])

        return transform

