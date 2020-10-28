import torchvision.transforms.functional as TF
import numpy as np
import scipy as sp
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision
import torch
from tabulate import tabulate
from tensorflow import keras

# Rotation config
config = {}
config['do_rotate'] = True
config['skew_labels'] = False
config['skew_rotations'] = True
config['rotations'] = np.array(range(72)) * 5
config['rotation_probs'] = np.zeros(72)
config['rotation_probs'][:24] = 1/100
config['rotation_probs'][24:48] = 10/100
config['rotation_probs'][48:] = 89/100

TRAIN_SIZE = 60000
IMG_SIZE = 28


def preprocess(X, y):
    return X.reshape([-1, 28, 28, 1]).astype(np.float64), y

def rescale(X):
    return X.astype(np.float32) / 256.

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

class StaticMNIST(Dataset):

    def __init__(self, data, split,
                 data_folder='datasets'):

        super(StaticMNIST, self).__init__()

        self.image_shape = (1, 28, 28)
        self.images, self.labels = data
        self.original_size = len(self.images)

        self.all_indices = range(self.original_size)

        # Generate the right dataset based on config
        # Create skew
        np.random.seed(1)
        skew_config = config
        self.do_rotate = skew_config['do_rotate']
        if split == 'train':
            self.indices, self.rotations, self.rotation_ids = self._get_train_skew(skew_config)
        else:
            self.indices, self.rotations, self.rotation_ids = self._get_test(skew_config)

        # Retrieve set
        self.images, self.labels = self.images[self.indices], self.labels[self.indices]

        # Map to groups
        # Get group ids
        self.group_ids, self.n_groups = self._get_group_ids(skew_config)
        self.groups = np.unique(self.group_ids)

        self.group_stats = np.zeros((self.n_groups, 3))

        self.group_counts, bin_edges = np.histogram(self.group_ids, bins=range(self.n_groups+1), density=False)
        self.group_dist, bin_edges = np.histogram(self.group_ids, bins=range(self.n_groups+1), density=True)

        for group_id in range(self.n_groups):
            indices = np.nonzero(np.asarray(self.group_ids == group_id))[0]
            num_in_group = len(indices)
            self.group_stats[group_id, 0] = num_in_group # Num in group
            self.group_stats[group_id, 1] = num_in_group / len(self.labels) # Frac in group

            # Check if group correspond to even? TODO: Write in cleaner way.
            self.group_stats[group_id, 2] = np.mean(self.rotations[indices])

        self.df_stats = pd.DataFrame(self.group_stats, columns=['n', 'frac',  'rotation'])
        self.df_stats['group_id'] = self.df_stats.index
        self.df_stats['binary'] = self.df_stats['group_id'].apply(lambda x: '{0:b}'.format(x).zfill(int(np.log(self.n_groups))))

        # Print dataset stats
        print("Number of examples", len(self.indices))
        print(tabulate(self.df_stats, headers='keys', tablefmt='psql'))

    def _get_group_ids(self, skew_config):
        """Returns the group ids for each example

            TODO: Clean up this function"""

        group_ids = self.rotation_ids
        n_rotations = len(skew_config['rotations'])
        n_groups = n_rotations

        return group_ids, n_groups



    def _get_test(self, skew_config):
        """Returns the test set by duplicating the original
            MNIST test set for each rotation angle.

            There is no skew.

            TODO: Clean up this function"""

        rotations = []
        rotation_ids = []
        indices = []

        for rotation_id, rotation in enumerate(skew_config['rotations']):

            rotations.extend([rotation] * self.original_size)
            rotation_ids.extend([rotation_id] * self.original_size)
            indices.extend(self.all_indices)

        rotations = np.array(rotations)
        rotation_ids = np.array(rotation_ids)
        indices = np.array(indices)

        return indices, rotations, rotation_ids

    def _get_train_skew(self, skew_config):
        """Returns a skewed train set"""

        num_examples_total = len(self.labels)

        indices = []
        rotations = []
        rotation_ids = []
        for rotation_id, rotation in enumerate(skew_config['rotations']):
            rotation_prob = skew_config['rotation_probs'][rotation_id]
            group_prob = rotation_prob
            num_examples = int(rotation_prob * self.original_size / 4)
            indices_for_rotation = np.random.choice(self.original_size, size=num_examples)
            rotations.append(len(indices_for_rotation) * [rotation])
            rotation_ids.append(len(indices_for_rotation) * [rotation_id])
            indices.append(indices_for_rotation)

        rotations = np.concatenate(rotations)
        rotation_ids = np.concatenate(rotation_ids)
        indices = np.concatenate(indices)

        return indices, rotations, rotation_ids

    def __len__(self):
        """Returns number of examples in the dataset"""
        return len(self.labels)

    def __getitem__(self, index):

        img = self.images[index]
        if self.do_rotate:
            rotation = self.rotations[index]
            img = rotate(img, rotation, single_image=True)

        img = rescale(img) # =/ 256
        img = torch.tensor(img, dtype=torch.float)

        # Put color channel first
        img = img.unsqueeze(0)
        img = img.squeeze(-1)

        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)


        group_id = self.group_ids[index]


        return img, label, group_id

