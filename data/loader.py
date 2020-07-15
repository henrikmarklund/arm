import numpy as np
import torch

from .celeba_dataset import CelebADataset
from .static_mnist import StaticMNIST
from .femnist_dataset import FEMNISTDataset
from . import static_mnist
from .samplers import ClusteredMixSampler, ConstantMixSampler, ConstantGroupSampler, ClusteredGroupSampler

def get_one_hot(values, num_classes):    # putting this here for now so you can get it working in one copy paste
    return np.eye(num_classes)[values]

def get_loader(dataset, sampling_type=None, batch_size=None, meta_batch_size=None,
               support_size=None, shuffle=True, meta_distribution=None,
               pin_memory=True, num_workers=8, args=None):
    """Returns a data loader that sample meta_batches of data where each
            meta batch contains a set of support batches. Each support batch
            contain examples all having the same angle

    """

    if sampling_type == 'meta_batch_mixtures': # Sample support batches from multiple sub distributions
        batch_sampler = ClusteredMixSampler(dataset, meta_batch_size, support_size,
                                         args=args)

        batch_size = 1
        shuffle = None
        sampler=None
        drop_last = False

    elif sampling_type == 'meta_batch_groups': # Sample support batches from multiple sub distributions
        batch_sampler = ClusteredGroupSampler(dataset, meta_batch_size, support_size,
                                          uniform_over_groups=args.uniform_over_groups)

        batch_size = 1
        shuffle = None
        sampler=None
        drop_last = False
        print("meta batch group")

    elif sampling_type == 'constant_mixture': # Sample batches from a sub distribution
        sampler = ConstantMixSampler(dataset, replacement=True)
        batch_sampler = None
        drop_last=True
        shuffle = None
        print("constant mixture")

    elif sampling_type == 'constant_group': # Sample batches from specific group
        batch_sampler = ConstantGroupSampler(dataset, batch_size, replacement=False)

        batch_size = 1
        shuffle = None
        sampler=None
        drop_last = False
        print("constant group")

    elif sampling_type == 'uniform_over_groups':
        # Sample batches from the sub distribution that is uniform over groups
        # Put each group uniformly
        sampler = ConstantMixSampler(dataset, replacement=True)
        batch_sampler = None
        sampler.set_uniform_dist_over_groups()
        drop_last = True
        shuffle = None
        print("uniform over groups")
    elif sampling_type == 'regular': # Sample each example uniformly
        sampler = None
        batch_sampler = None
        if args is not None:
            drop_last = bool(args.drop_last)
        else:
            drop_last = False
        print("regular: ")
        print("Drop last", drop_last)
        if shuffle == 0: shuffle=False

        print("Shuffle: ", shuffle)

    loader = torch.utils.data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  sampler=sampler,
                                  batch_sampler=batch_sampler,
                                  pin_memory=pin_memory,
                                  num_workers=num_workers,
                                  drop_last=drop_last)
    return loader


def get_dataset(args):

    if args.dataset == 'celeba':
        train_dataset = CelebADataset('train', args.data_dir, args.target_name, args.confounder_names,
                       target_resolution=args.target_resolution,
                                loading_type=args.loading_type, crop_type=args.crop_type)
        val_dataset = CelebADataset('val', args.data_dir, args.target_name, args.confounder_names,
                       target_resolution=args.target_resolution,
                                loading_type=args.loading_type, crop_type=args.crop_type)
        test_dataset = CelebADataset('test', args.data_dir, args.target_name, args.confounder_names,
                       target_resolution=args.target_resolution, loading_type=args.loading_type, crop_type=args.crop_type)

    elif args.dataset == 'femnist':
        train_dataset = FEMNISTDataset('train', args.data_dir)
        val_dataset = FEMNISTDataset('val', args.data_dir)
        test_dataset = FEMNISTDataset('test', args.data_dir)

    elif args.dataset == 'mnist':
        train, val, test = static_mnist.get_data()
        train_dataset = StaticMNIST(train, 'train')
        val_dataset = StaticMNIST(val, 'val')
        test_dataset = StaticMNIST(test, 'test')

    return train_dataset, val_dataset, test_dataset


def get_loaders(args):

    train_dataset, val_dataset, test_dataset = get_dataset(args)
    batch_size = args.meta_batch_size * args.support_size
    train_loader = get_loader(train_dataset, sampling_type=args.sampling_type,
                              batch_size=batch_size,
                              meta_batch_size=args.meta_batch_size,
                              support_size=args.support_size,
                              shuffle=args.shuffle_train,
                              pin_memory=args.pin_memory, num_workers=args.num_workers,
                              args=args)

    # The test loader will sample examples from a sub distribution that is set during evaluation
    # You can update this sub distribution during evaluation

    eval_sampling_type = 'constant_group' if args.eval_corners_only else 'constant_mixture'

    if 'eval_deterministic' in args and args.eval_deterministic:
        eval_sampling_type = 'regular'

    train_eval_loader = get_loader(train_dataset, eval_sampling_type,
                          batch_size, shuffle=False,
                          pin_memory=args.pin_memory, num_workers=args.num_workers)
    val_loader = get_loader(val_dataset, eval_sampling_type,
                          batch_size, shuffle=False,
                          pin_memory=args.pin_memory, num_workers=args.num_workers)
    test_loader = get_loader(test_dataset, eval_sampling_type,
                          batch_size, shuffle=False,
                          pin_memory=args.pin_memory, num_workers=args.num_workers)

    return train_loader, train_eval_loader, val_loader, test_loader

