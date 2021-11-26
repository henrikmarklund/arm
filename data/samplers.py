import random

from torch.utils.data.sampler import Sampler
import numpy as np
import torch

def get_one_hot(values, num_classes):
    return np.eye(num_classes)[values]


class GroupSampler:
    r"""
        Samples batches of data from predefined groups.
    """

    def __init__(self, dataset, meta_batch_size, support_size,
                 drop_last=None, uniform_over_groups=True):

        self.dataset = dataset
        self.indices = range(len(dataset))

        self.group_ids = dataset.group_ids
        self.groups = dataset.groups
        self.num_groups = dataset.n_groups

        self.meta_batch_size = meta_batch_size
        self.support_size = support_size
        self.batch_size = meta_batch_size * support_size
        self.drop_last = drop_last
        self.dataset_size = len(self.dataset)
        self.num_batches = len(self.dataset) // self.batch_size

        self.groups_with_ids = {}
        self.actual_groups = []

        # group_count will have one entry per group
        # with the size of the group
        self.group_count = []
        for group_id in self.groups:
            ids = np.nonzero(self.group_ids == group_id)[0]
            self.group_count.append(len(ids))
            self.groups_with_ids[group_id] = ids

        self.group_count = np.array(self.group_count)
        self.group_prob = self.group_count / np.sum(self.group_count)
        self.uniform_over_groups = uniform_over_groups

    def __iter__(self):

        n_batches = len(self.dataset) // self.batch_size
        if self.uniform_over_groups:
            sampled_groups = np.random.choice(self.groups, size=(n_batches, self.meta_batch_size))
        else:
            # Sample groups according to the size of the group
            sampled_groups = np.random.choice(self.groups, size=(n_batches, self.meta_batch_size), p=self.group_prob)

        group_sizes = np.zeros(sampled_groups.shape)

        for batch_id in range(self.num_batches):

            sampled_ids = [np.random.choice(self.groups_with_ids[sampled_groups[batch_id, sub_batch]],
                                size=self.support_size,
                                replace=True,
                                p=None)
                                for sub_batch in range(self.meta_batch_size)]



            # Flatten
            sampled_ids = np.concatenate(sampled_ids)

            yield sampled_ids

        self.sub_distributions = None

    def __len__(self):
        return len(self.dataset) // self.batch_size

