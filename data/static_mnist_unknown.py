import torchvision.transforms.functional as TF
import numpy as np
import scipy as sp
import pandas as pd
import colorsys

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision
import torch
from tabulate import tabulate
from tensorflow import keras
from pathlib import Path

# Rotation config
CONFIGS = {}
config = {}
config['group_type'] = 'rotation'
n_groups = 14
config['n_groups'] = n_groups
config['group_values'] = np.array(range(n_groups)) * 10
config['group_probs'] = np.zeros(n_groups)
config['group_probs'][:3] = 70/100 # 0 - 20
config['group_probs'][3:6] = 20/100 # 30 - 50
config['group_probs'][6:9] = 6/100 # 60 - 80
config['group_probs'][9:12] = 3/100 # 90 - 110
config['group_probs'][12:] = 1/100

CONFIGS['rotation'] = config.copy()

TRAIN_SIZE = 60000
IMG_SIZE = 28


def preprocess(X, y):
    return X.reshape([-1, 28, 28, 1]).astype(np.float64), y

def to_rgb(X):

    return np.concatenate([X,X,X], axis=3)

def rescale(X):
    return X.astype(np.float32) / 255.

def rotate(X, rotation, single_image=False):
    if single_image:
        return np.array(sp.ndimage.rotate(X, rotation, reshape=False, order=0))
    else:
        return np.array(
            [sp.ndimage.rotate(X[i], rotation[i], reshape=False, order=0)
             for i in range(X.shape[0])]
        )

def get_data(shuffle=True):
    """Returns train, val and test for mnist"""
    (X_train, y_train), (X_test, y_test) = [preprocess(*data) for data in
                                            keras.datasets.mnist.load_data()]

    if shuffle:
        train_perm = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[train_perm], y_train[train_perm]

        train_frac = 0.90
        n_train = int(len(X_train) * train_frac)

        X_val = X_train[n_train:]
        y_val = y_train[n_train:]

        X_train = X_train[:n_train]
        y_train = y_train[:n_train]

        test_perm = np.random.permutation(X_test.shape[0])
        X_test, y_test = X_test[test_perm], y_test[test_perm]


    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

class StaticMNISTUnknown(Dataset):

    def __init__(self, data, split, args,
                 data_folder='datasets'):

        super(StaticMNISTUnknown, self).__init__()

        self.images, self.labels = data
        self.original_size = len(self.images)

        self.num_classes = 10

        self.all_indices = range(self.original_size)

        # Generate the right dataset based on config
        # Create skew
        np.random.seed(1)
        config = CONFIGS['rotation']
        self.image_shape = (1, 28, 28)
        self.config = config

        self.group_type = config['group_type']
        self.group_values = config['group_values']

        if split == 'train':
            self.indices, self.group_ids = self._get_train_skew(config)
        else:
            self.indices, self.group_ids = self._get_test(config)

        # Retrieve set
        self.images, self.labels = self.images[self.indices], self.labels[self.indices]


        # Map to groups
        # Get group ids
        self.groups = np.unique(self.group_ids)
        self.n_groups = config['n_groups']

        self.group_stats = np.zeros((self.n_groups, 2))

        self.group_counts, bin_edges = np.histogram(self.group_ids, bins=range(self.n_groups+1), density=False)
        self.group_dist, bin_edges = np.histogram(self.group_ids, bins=range(self.n_groups+1), density=True)

        for group_id in range(self.n_groups):
            indices = np.nonzero(np.asarray(self.group_ids == group_id))[0]
            num_in_group = len(indices)
            self.group_stats[group_id, 0] = num_in_group # Num in group
            self.group_stats[group_id, 1] = num_in_group / len(self.labels) # Frac in group

        print("len dataset: ", len(self.labels))

        self.df_stats = pd.DataFrame(self.group_stats, columns=['n', 'frac'])
        self.df_stats['group_id'] = self.df_stats.index
        self.df_stats['binary'] = self.df_stats['group_id'].apply(lambda x: '{0:b}'.format(x).zfill(int(np.log(self.n_groups))))

        self.binarize = False

        # Print dataset stats
        print("Number of examples", len(self.indices))
        print(tabulate(self.df_stats, headers='keys', tablefmt='psql'))


    def _get_test(self, skew_config):
        """Returns the test set by duplicating the original
            MNIST test set for each rotation angle.

            There is no skew.

            TODO: Clean up this function"""

        group_ids = []
        indices = []

        for group_id in range(skew_config['n_groups']):

            num_examples = int(self.original_size / 3)
            group_ids.extend([group_id] * num_examples)
            indices.extend(self.all_indices)

        group_ids = np.array(group_ids)
        indices = np.array(indices)

        return indices, group_ids

    def _get_train_skew(self, skew_config):
        """Returns a skewed train set"""


        num_examples_total = len(self.labels)

        indices = []
        group_ids = []
        for group_id in range(skew_config['n_groups']):

            group_prob = skew_config['group_probs'][group_id]

            if group_prob == 0:
                continue

            num_examples = int(group_prob * self.original_size / 5)

            print("group type: ", self.group_type)

            indices_for_group = np.random.choice(self.original_size, size=num_examples)
            group_ids.append(len(indices_for_group) * [group_id])

            print("Group id", group_id)

            indices.append(indices_for_group)

        group_ids = np.concatenate(group_ids)
        indices = np.concatenate(indices)

        return indices, group_ids

    def __len__(self):
        """Returns number of examples in the dataset"""
        return len(self.labels)

    def __getitem__(self, index):

        group_id = self.group_ids[index]

        img = self.images[index]

        self.apply_transform = True
        if self.apply_transform:

            group_value = self.group_values[group_id]
            img = rotate(img, group_value, single_image=True)

        img = rescale(img) # =/ 256
        if self.binarize:
            img = np.random.binomial(1, img).astype(np.float32)

        img = torch.tensor(img, dtype=torch.float)

        # Put color channel first
        img = img.permute(-1, 0, 1)

        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)


        return img, label, group_id

