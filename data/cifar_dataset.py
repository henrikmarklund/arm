import json
import os

import albumentations
import numpy as np
import torch

from collections import defaultdict
from pathlib import Path

from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

def load_corruption(path):
    data = np.load(path)
    return np.array(np.array_split(data, 5))

class CIFARDataset(Dataset):
    def __init__(self, split, root_dir):

        if split == 'train':
            self.root_dir = Path(root_dir) / 'CIFAR-10-C-new/train/'
            corruptions = ['gaussian_noise', 'shot_noise', 'defocus_blur', 'glass_blur', 'zoom_blur', 'snow', 'frost', 'brightness', 'contrast', 'pixelate']
            other_idx = [0, 1, 2, 5, 6, 7]

        if split == 'val':
            self.root_dir = Path(root_dir) / 'CIFAR-10-C-new/val/'
            corruptions = ['speckle_noise', 'gaussian_blur', 'saturate']
            other_idx = [3, 9]

        if split == 'test':
            self.root_dir = Path(root_dir) / 'CIFAR-10-C/'
            corruptions = ['impulse_noise', 'motion_blur', 'fog', 'elastic_transform']
            other_idx = [4, 8]

        print("loading cifar-10-c")
        other = [load_corruption(self.root_dir / (corruption + '.npy')) for corruption in ['spatter', 'jpeg_compression']]
        other = np.concatenate(other, axis=0)[other_idx]

        data = [load_corruption(self.root_dir / (corruption + '.npy')) for corruption in corruptions]
        data = np.concatenate(data, axis=0)

        self._X = np.concatenate([other, data], axis=0)

        n_images_per_group = self._X.shape[1]

        self.n_groups = self._X.shape[0]
        self.groups = list(range(self.n_groups))
        self.image_shape = (3, 32, 32)
        self._X = self._X.reshape((-1, 32, 32, 3))
        self.num_classes = 10

        if split == 'test':
            n_images = 10000
            self._y = np.load(self.root_dir / 'labels.npy')[:n_images]
            self._y = np.tile(self._y, self.n_groups)
            self.group_ids = np.array([[i]*n_images for i in range(self.n_groups)]).flatten()
        else:
            n_images = 1000
            other_labels = [load_corruption(self.root_dir / (corruption + '_labels.npy')) for corruption in ['spatter', 'jpeg_compression']]
            other_labels = np.concatenate(other_labels, axis=0)[other_idx]
            data_labels = [load_corruption(self.root_dir / (corruption + '_labels.npy')) for corruption in corruptions]
            data_labels = np.concatenate(data_labels, axis=0)
            self._y = np.concatenate([other_labels, data_labels], axis=0).flatten()
            self.group_ids = np.array([[i]*n_images for i in range(self.n_groups)]).flatten()

        self._len = len(self.group_ids)
        print("loaded")

        self.group_counts, _ = np.histogram(self.group_ids,
                                            bins=range(self.n_groups + 1),
                                            density=False)
        self.transform = get_transform()
        print("split: ", split)
        print("n groups: ", self.n_groups)
        print("Dataset size: ", len(self._y))
        print("Smallest group: ", np.min(self.group_counts))
        print("Largest group: ", np.max(self.group_counts))

    def __len__(self):
        return self._len
    def __getitem__(self, index):
        x = self.transform(**{'image': self._X[index]})['image']
        y = torch.tensor(self._y[index], dtype=torch.long)
        g = torch.tensor(self.group_ids[index], dtype=torch.long)

        return x, y, g


def get_transform():
    transform = albumentations.Compose([
        albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                     		 std=[0.229, 0.224, 0.225], max_pixel_value=255,
                                 p=1.0, always_apply=True),
        ToTensor()
    ])
    return transform
