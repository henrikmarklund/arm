import numpy as np
import torch

from .static_mnist_unknown import StaticMNISTUnknown
from .tinyimagenet_dataset import ImageNetDataset
from .cifar_dataset import CIFARDataset
from .femnist_dataset import FEMNISTDataset
from . import static_mnist_unknown
from .samplers import GroupSampler

def get_loader(dataset, sampler_type, uniform_over_groups=False,
               meta_batch_size=None,support_size=None, shuffle=True,
               pin_memory=True, num_workers=8, args=None):
    """Returns a data loader that sample meta_batches of data where each
            meta batch contains a set of support batches. Each support batch
            contain examples all having the same angle

    """

    if sampler_type == 'group': # Sample support batches from multiple sub distributions
        batch_sampler = GroupSampler(dataset, meta_batch_size, support_size,
                                          uniform_over_groups=uniform_over_groups)

        batch_size = 1
        shuffle = None
        sampler=None
        drop_last = False
    else:

        batch_size = meta_batch_size * support_size

        if uniform_over_groups:
            group_weights = 1 / dataset.group_counts
            weights = group_weights[dataset.group_ids]
            sampler = torch.utils.data.WeightedRandomSampler(weights, len(dataset), replacement=True)
            batch_sampler = None
            drop_last = True
            shuffle = None
        else: # Sample each example uniformly

            print("standard sampler")

            sampler = None
            batch_sampler = None
            if args is not None:
                drop_last = bool(args.drop_last)
            else:
                drop_last = False
            if shuffle == 0:
                shuffle=False
            else:
                shuffle=True
            print("shuffle: ", shuffle)

    loader = torch.utils.data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  sampler=sampler,
                                  batch_sampler=batch_sampler,
                                  pin_memory=pin_memory,
                                  num_workers=num_workers,
                                  drop_last=drop_last)
    return loader


def get_dataset(args, only_train=False):


    if args.dataset == 'femnist':
        train_dataset = FEMNISTDataset('train', args.data_dir, args)
        val_dataset = FEMNISTDataset('val', args.data_dir, args)
        test_dataset = FEMNISTDataset('test', args.data_dir, args)

    elif args.dataset == 'tinyimg':
        train_dataset = ImageNetDataset('train', args.data_dir)
        val_dataset = ImageNetDataset('val', args.data_dir)
        test_dataset = ImageNetDataset('test', args.data_dir)

    elif args.dataset == 'cifar-c':
        train_dataset = CIFARDataset('train', args.data_dir)
        val_dataset = CIFARDataset('val', args.data_dir)
        test_dataset = CIFARDataset('test', args.data_dir)

    elif args.dataset == 'mnist':
        train, val, test = static_mnist_unknown.get_data()
        train_dataset = StaticMNISTUnknown(train, 'train', args)
        val_dataset = StaticMNISTUnknown(val, 'val', args)
        test_dataset = StaticMNISTUnknown(test, 'test', args)

    if only_train:
        return train_dataset
    else:
        return train_dataset, val_dataset, test_dataset


def get_loaders(args, only_train=False):
    train_loader, train_eval_loader, val_loader, test_loader = None, None, None, None

    if only_train:
        train_dataset = get_dataset(args, only_train=True)
    else:
        train_dataset, val_dataset, test_dataset = get_dataset(args, only_train=False)

    print("dataset: ", train_dataset)

    if train_dataset is not None:
        train_loader = get_loader(train_dataset, sampler_type=args.sampler, uniform_over_groups=args.uniform_over_groups,
                                  meta_batch_size=args.meta_batch_size,
                                  support_size=args.support_size,
                                  shuffle=args.shuffle_train,
                                  pin_memory=args.pin_memory, num_workers=args.num_workers,
                                  args=args)

    # The test loader will sample examples from a sub distribution that is set during evaluation
    # You can update this sub distribution during evaluation
    # The following are not really in use
    eval_sampling_type = 'group'
    if not only_train:
        if train_dataset is not None:
            train_eval_loader = get_loader(train_dataset, eval_sampling_type, uniform_over_groups=False,
                                  meta_batch_size=args.meta_batch_size,
                                  support_size=args.support_size,
                                  shuffle=False,
                                  pin_memory=args.pin_memory, num_workers=args.num_workers)
        if val_dataset is not None:
            val_loader = get_loader(val_dataset, eval_sampling_type, uniform_over_groups=False,
                                  meta_batch_size=args.meta_batch_size,
                                  support_size=args.support_size,
                                  shuffle=False,
                                  pin_memory=args.pin_memory, num_workers=args.num_workers)
        if test_dataset is not None:
            test_loader = get_loader(test_dataset, eval_sampling_type, uniform_over_groups=False,
                                  meta_batch_size=args.meta_batch_size,
                                  support_size=args.support_size,
                                  shuffle=False,
                                  pin_memory=args.pin_memory, num_workers=args.num_workers)

        return train_loader, train_eval_loader, val_loader, test_loader
    else:
        return train_loader

