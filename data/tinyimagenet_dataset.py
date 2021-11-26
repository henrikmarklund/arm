import json
import os
import re

import albumentations
import numpy as np
import torch

from collections import defaultdict
from pathlib import Path

from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
from PIL import Image


class ImageNetDataset(Dataset):

    def __init__(self, split, root_dir):
        if split == 'train':
            self.root_dir = Path(root_dir) / 'Tiny-ImageNet-C-new/train/'
            corruptions = ['gaussian_noise', 'shot_noise', 'defocus_blur', 'glass_blur', 'zoom_blur', 'snow', 'brightness', 'contrast', 'pixelate']
            frost_idx = [1, 2, 3]
            jpeg_idx = [1, 2, 3]
        if split == 'val':
            self.root_dir = Path(root_dir) / 'Tiny-ImageNet-C-new/val'
            corruptions = ['speckle_noise', 'gaussian_blur', 'saturate']
            frost_idx = [4]
            jpeg_idx = [5]
        if split == 'test':
            self.root_dir = Path(root_dir) / 'Tiny-ImageNet-C/'
            corruptions = ['impulse_noise', 'motion_blur', 'fog', 'elastic_transform']
            frost_idx = [5]
            jpeg_idx = [4]
        print("loading tiny-imagenet-c")

        data = []
        for level in frost_idx:
            data.extend(self.construct_imdb('frost', level))
        for level in jpeg_idx:
            data.extend(self.construct_imdb('jpeg_compression', level))
        for corruption in corruptions:
            for level in [1, 2, 3, 4, 5]:
                data.extend(self.construct_imdb(corruption, level))
        self._X = data # np.concatenate([spatter, jpeg, data], axis=0)
        self.n_groups = len(frost_idx) + len(jpeg_idx) + 5*len(corruptions)
        self.groups = list(range(self.n_groups))

        self.image_shape = (3, 64, 64)
        if split == 'test':
            self.group_ids = np.array([[i]*10000 for i in range(self.n_groups)]).flatten()
        else:
            self.group_ids = np.array([[i]*2000 for i in range(self.n_groups)]).flatten()

        self._len = len(self.group_ids)
        print("loaded")

        self.group_counts, _ = np.histogram(self.group_ids,
                                            bins=range(self.n_groups + 1),
                                            density=False)

        self.transform = get_transform(split)


        print("split: ", split)
        print("n groups: ", self.n_groups)
        print("Dataset size: ", len(self._X))

        print("Smallest group: ", np.min(self.group_counts))
        print("Largest group: ", np.max(self.group_counts))

    def construct_imdb(self, corruption, level):
        """Constructs the imdb."""
        # Compile the split data path
        split_path = os.path.join(self.root_dir, corruption, str(level))
        re_pattern = r"^n[0-9]+$"
        # Images are stored per class in subdirs (format: n<number>)
        class_ids = sorted(f for f in os.listdir(split_path) if re.match(re_pattern, f))
        # Map ImageNet class ids to contiguous ids
        class_id_cont_id = {v: i for i, v in enumerate(class_ids)}
        # Construct the image db
        imdb = []
        for class_id in class_ids:
            cont_id = class_id_cont_id[class_id]
            im_dir = os.path.join(split_path, class_id)
            for im_name in os.listdir(im_dir):
                imdb.append(
                    {"im_path": os.path.join(im_dir, im_name), "class": cont_id}
                )
        return imdb

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        img = Image.open(self._X[index]["im_path"])
        img = np.array(img)
        img = self.transform(**{"image": img})['image']
        y = torch.tensor(self._X[index]["class"], dtype=torch.long)
        g = torch.tensor(self.group_ids[index], dtype=torch.long)
        return img, y, g


def get_transform(split):
    transform = albumentations.Compose([
        albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225], max_pixel_value=255,
                                 p=1.0, always_apply=True),
        ToTensor(),
    ])
    return transform
