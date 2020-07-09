import random

from torch.utils.data.sampler import WeightedRandomSampler, Sampler
import numpy as np
import torch

def get_one_hot(values, num_classes):

    return np.eye(num_classes)[values]

class ClusteredMixSampler:
    r"""
        Samples batches from mixes of groups.
    """

    def __init__(self, dataset, meta_batch_size, support_size,
                 drop_last=None, args=None):
        """
            Args:
                dataset:
                meta_batch_size:
                support_size:
                use_dist_over_groups: This is used in the known groups setting
        """

        self.dataset = dataset
        self.indices = range(len(dataset))
        self.group_ids = dataset.group_ids
        self.num_groups = dataset.n_groups

        # Pre compute probabilities
        self.group_ids_one_hot = get_one_hot(self.group_ids, self.num_groups)
        self.group_count = np.sum(self.group_ids_one_hot, axis=0)
        self.group_ids_probs_pre = self.group_ids_one_hot * (1 / self.group_count)


        self.meta_batch_size = meta_batch_size
        self.support_size = support_size
        self.batch_size = meta_batch_size * support_size
        self.drop_last = drop_last
        self.dataset_size = len(self.dataset)
        self.num_batches = len(self.dataset) // self.batch_size

    def set_all_sub_dists(self, dists_np):
        """ Saves a set of distrubtions. Each of which is over the whole dataset.,

            dists_np: Numpy array with shape (num_batches, meta_batch_size, dataset_size)"""

        self.all_sub_dists = dists_np

    def set_sub_dists_ids(self, dist_ids):
        """ Sets distribution over all examples

            Args:
                dists_ids: Np array containing sub dist ids
                    Shape: (num_batches, meta_batch_size)"""

        self.sub_dists_ids = dist_ids

    def set_group_sub_dists(self, dists):
        """ Sets distribution over groups"""
        self.group_sub_dists = dists

    def _get_dist(self, p_over_groups):
        """
            Args:
                p_over_groups: Distribution over groups

            Return: Distribution over the dataset.
        """

        p_over_examples = np.sum(self.group_ids_probs_pre * p_over_groups, axis=1)

        return p_over_examples

    def __iter__(self):

        for batch_id in range(self.num_batches):

            sampled_ids = [np.random.choice(self.indices, size=self.support_size,
                                replace=False,
                                p=self._get_dist(self.group_sub_dists[batch_id, sub_batch]))
                                for sub_batch in range(self.meta_batch_size)]

            sampled_ids = np.concatenate(sampled_ids)

            # Flatten

            yield sampled_ids

        self.sub_distributions = None

    def __len__(self):
        return len(self.dataset) // self.batch_size


class ClusteredGroupSampler:
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

class ConstantGroupSampler:
    r"""
        Samples batches of data from predefined groups.

        This one is in practice not used. Currently, just holds the val and test datasets, but sampling is done separately.
    """

    def __init__(self, dataset, batch_size, use_known_groups=True,
                 drop_last=None, replacement=False):
        """
        """

        self.dataset = dataset
        self.indices = range(len(dataset))
        self.use_known_groups = use_known_groups

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.dataset_size = len(self.dataset)
        self.num_batches = len(self.dataset) // self.batch_size
        self.replacement = replacement

        self.ids_for_current_group = None
        self.current_group_id = None

        self.groups_with_ids = {}
        self.group_count = []
        for group_id in dataset.groups:
            ids = np.nonzero(dataset.group_ids == group_id)[0]
            self.groups_with_ids[group_id] = ids

    def set_group(self, group_id):
        self.current_group_id = group_id
        ids = self.groups_with_ids[group_id]
        self.ids_for_current_group = ids

    def __iter__(self):

        if self.replacement is False:
            num_batches = len(self.ids_for_current_group) // self.batch_size + 1

            current_group_size = len(self.ids_for_current_group)
            ids = self.ids_for_current_group[np.random.permutation(current_group_size)] # Random permutation that affects CML.
            #ids = self.ids_for_current_group # Random permutation that affects CML.

            for batch_id in range(num_batches):

                if batch_id == num_batches - 1:
                    sampled_ids = ids[batch_id*self.batch_size:]
                else:
                    sampled_ids = ids[batch_id*self.batch_size:(batch_id+1)*self.batch_size]

                yield sampled_ids

        else:
            for batch_id in range(self.num_batches):

                sampled_ids = list(np.random.choice(self.ids_for_current_group,
                                    size=self.batch_size,
                                    replace=True))

                yield sampled_ids

    def __len__(self):
        if len(self.ids_for_current_group) % self.batch_size == 0:
            return len(self.ids_for_current_group) // self.batch_size
        else:
            return len(self.ids_for_current_group) // self.batch_size + 1

class ConstantMixSampler(WeightedRandomSampler):
    """Samples examples from a specific sub distribution

        E.g if there are 4 groups, then the sub distribution may be [0.4,0.2,0.4,0]

    """

    def __init__(self, dataset, replacement=True):

        super().__init__(np.ones(len(dataset)), len(dataset), replacement=replacement)

        self.dataset = dataset
        self.num_samples = len(dataset)
        self.indices = range(len(dataset))

        self.group_ids = dataset.group_ids
        self.num_groups = dataset.n_groups

        # Pre compute probabilities
        self.group_ids_one_hot = get_one_hot(self.group_ids, self.num_groups)
        self.group_count = np.sum(self.group_ids_one_hot, axis=0) # If there are e.g 4 groups, this will be a vector with 4 entries.
        # The following is later multiplied by the distribution over groups to get the distribution over the dataset.
        self.group_ids_probs_pre = self.group_ids_one_hot * (1 / self.group_count)

        self.sub_distribution = None

    def set_sub_dist(self, dist):
        """Sets distribution over groups"""
        self.sub_distribution = dist
        self.weights = torch.as_tensor(self._get_dist(dist), dtype=torch.double)

    def set_uniform_dist_over_groups(self):
        """Makes all groups equally likely"""

        sub_dist = np.ones(self.num_groups) / self.num_groups
        self.set_sub_dist(sub_dist)
        print("sub dist set uniform: ", sub_dist)

    def _get_dist(self, p_over_groups):
        """
            Args:
                p_over_groups: Distribution over groups

            Return: Distribution over examples
        """

        p_over_examples = np.sum(self.group_ids_probs_pre * p_over_groups, axis=1)

        return p_over_examples

