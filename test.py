import os
from datetime import datetime
import argparse
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from tqdm import trange, tqdm
import wandb
from sklearn import metrics
from dro_loss import LossComputer
from binned_dists import get_binned_dists

import data
import utils

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"


def get_one_hot(values, num_classes):
    return np.eye(num_classes)[values]

def test(args, eval_on):

    # Cuda
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tags = ['supervised', f'{args.dataset}', f'use_context_{args.use_context}']

    if args.debug:
        tags.append('debug')

    if args.log_wandb:
        wandb.init(name=args.experiment_name,
                   project=f"arm_test_{args.dataset}",
                   tags=tags,
                   reinit=True
                   )
        wandb.config.update(args, allow_val_change=True)

    # Get data
    train_loader, train_eval_loader, val_loader, test_loader = data.get_loaders(args)
    args.n_groups = train_loader.dataset.n_groups

    if args.seed is not None:
        print('setting seed', args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.binning:
        corner_dists = np.array(get_one_hot(range(args.n_groups), args.n_groups))
        binned_groups = get_binned_dists(args.n_test_dists)
        if eval_on == 'train':
            empirical_dist = [train_loader.dataset.group_dist]
        elif eval_on == 'val':
            empirical_dist = [val_loader.dataset.group_dist]
        elif eval_on == 'test':
            empirical_dist = [test_loader.dataset.group_dist]
        eval_dists = [corner_dists, binned_groups, empirical_dist]

    else:
        corner_dists = np.array(get_one_hot(range(args.n_groups), args.n_groups))
        if eval_on == 'train':
            empirical_dist = [train_loader.dataset.group_dist]
        elif eval_on == 'val':
            empirical_dist = [val_loader.dataset.group_dist]
        elif eval_on == 'test':
            empirical_dist = [test_loader.dataset.group_dist]

        if args.n_test_dists > 0:
            random_dists = np.random.dirichlet(np.ones(args.n_groups), size=args.n_test_dists)
            eval_dists = np.concatenate([corner_dists, random_dists, empirical_dist])
        else:
            eval_dists = np.concatenate([corner_dists, empirical_dist])


    # Get model
    model = utils.get_model(args, image_shape=train_loader.dataset.image_shape)
    state_dict = torch.load(args.ckpt_path)


    # Remove unnecessary weights from the model.
    new_state_dict = state_dict[0]
    if "context_net.classifier.weight" in new_state_dict:
        del new_state_dict["context_net.classifier.weight"]

    if "context_net.classifier.bias" in new_state_dict:
        del new_state_dict["context_net.classifier.bias"]

    new_state_dict_2 = {}
    for key in new_state_dict.keys():
        new_key = key.replace('module.', '')
        new_state_dict_2[new_key] = new_state_dict[key]
    new_state_dict = new_state_dict_2

    model.load_state_dict(new_state_dict)

    model = model.to(args.device)
    model.eval()

    # Train
    if args.eval_deterministic:
        if eval_on == 'train':
            stats = utils.evaluate_each_corner(args, model,
                    train_eval_loader, split='train')
        elif eval_on == 'val':
            stats = utils.evaluate_each_corner(args, model,
                    val_loader, split='val')
        elif eval_on == 'test':
            stats = utils.evaluate_each_corner(args, model,
                    test_loader, split='test')


    else:
        if eval_on == 'train':
            worst_case_acc, avg_acc, empirical_case_acc, stats = utils.evaluate_mixtures(args, model,
                    train_eval_loader, eval_dists, split='train')
        elif eval_on == 'val':
            worst_case_acc, avg_acc, empirical_case_acc, stats = utils.evaluate_mixtures(args, model,
                    val_loader, eval_dists,  split='val')
        elif eval_on == 'test':
            worst_case_acc, avg_acc, empirical_case_acc, stats = utils.evaluate_mixtures(args, model,
                    test_loader, eval_dists, split='test')

    return stats


# Arguments
parser = argparse.ArgumentParser()

# Model args
parser.add_argument('--model', type=str, default='ContextualConvNet')

parser.add_argument('--binning', type=int, default=0)
parser.add_argument('--eval_deterministic', type=int, default=0,
                                   help='Eval deterministic')
parser.add_argument('--drop_last', type=int, default=0)

parser.add_argument('--pretrained', type=int, default=1,
                                   help='Pretrained resnet')
# If model is Convnet
parser.add_argument('--prediction_net', type=str, default='resnet50',
                    choices=['resnet18', 'resnet34', 'resnet50', 'convnet'],
                    help='')

parser.add_argument('--n_context_channels', type=int, default=3, help='Used when using a convnet/resnet')
parser.add_argument('--use_context', type=int, default=0, help='Seed')

# Data args
parser.add_argument('--dataset', type=str, default='celeba', choices=['mnist','celeba'])
parser.add_argument('--data_dir', type=str, default='../data/')

# CelebA Data
parser.add_argument('--target_resolution', type=int, default=224,
                    help='Resize image to this size before feeding in to model')
parser.add_argument('--target_name', type=str, default='Blond_Hair',
                    help='The y value we are trying to predict')
parser.add_argument('--confounder_names', type=str, nargs='+',
                    default=['Male'],
                    help='Binary attributes from which we construct the groups. This is called confounder names \
                    for now since we are using part of Group DRO data loading')
parser.add_argument('--eval_on', type=str, nargs='+',
                    default=['test'],
                    help='Binary attributes from which we construct the groups. This is called confounder names \
                    for now since we are using part of Group DRO data loading')

# Data sampling
parser.add_argument('--meta_batch_size', type=int, default=2)
parser.add_argument('--support_size', type=int, default=50, help='Support size. Same as what we call batch size in the appendix.')
parser.add_argument('--shuffle_train', type=int, default=1,
                    help='Only relevant when do_clustered_sampling = 0 \
                    and --uniform_over_groups 0')

# Clustered sampling
parser.add_argument('--sampling_type', type=str, default='regular',
        choices=['meta_batch_mixtures', 'meta_batch_groups', 'uniform_over_groups', 'regular'],
                    help='Sampling type')
parser.add_argument('--eval_corners_only', type=int, default=0,
                    help='Are evaluating mixtures or corners?')

parser.add_argument('--loading_type', type=str, choices=['PIL', 'jpeg'], default='jpeg',
                    help='Whether to use PIL or jpeg4py when loading images. Jpeg is faster. See README for deatiles')

# Evalaution
parser.add_argument('--n_test_dists', type=int, default=100,
                    help='Number of test distributions to evaluate on. These are sampled uniformly.')
parser.add_argument('--n_test_per_dist', type=int, default=3000,
                    help='Number of examples to evaluate on per test distribution')

# Logging
parser.add_argument('--seed', type=int, default=0, help='Seed')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--log_wandb', type=int, default=1)

parser.add_argument('--num_workers', type=int, default=8, help='Num workers for pytorch data loader')
parser.add_argument('--pin_memory', type=int, default=1, help='Pytorch loader pin memory. \
                    Best practice is to use this')

parser.add_argument('--crop_type', type=int, default=0)

parser.add_argument('--ckpt_folders', type=str, nargs='+', default='')

parser.add_argument('--context_net', type=str, default='convnet')

parser.add_argument('--pret_add_channels', type=int, default=1,
                                   help='Relevant when using context and pretrained resnet')

parser.add_argument('--experiment_name', type=str, default='')


args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    # Cuda
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.cuda = True
    else:
        args.device = torch.device('cpu')
        args.cuda = False


    ### Check if checkpoints exist
    for ckpt_folder in args.ckpt_folders:
        ckpt_path = Path('output') / 'checkpoints' / ckpt_folder / f'best_weights.pkl'
        state_dict = torch.load(ckpt_path)
        print("Found: ", ckpt_path)


    all_test_stats = [] # Store all results in list.

    for i, ckpt_folder in enumerate(args.ckpt_folders):

        args.seed = i + 10 # Mainly to make sure seed for training and testing is not the same. Not critical.
        args.ckpt_path = Path('output') / 'checkpoints' / ckpt_folder / f'best_weights.pkl'
        args.experiment_name += f'_{args.seed}'

        if args.log_wandb and i != 0:
            wandb.join() # Initializes new wandb run

        train_stats, val_stats, test_stats = None, None, None

        if 'train' in args.eval_on:
            train_stats = test(args, eval_on='train')
        if 'val' in args.eval_on:
            val_stats = test(args, eval_on='val')
        if 'test' in args.eval_on:
            test_stats = test(args, eval_on='test')
            all_test_stats.append((i, test_stats))


        print("----SEED used when evaluating -----: ", args.seed)
        print("----CKPT FOLDER -----: ", ckpt_folder)
        print("TRAIN STATS:\n ", train_stats)
        print('-----------')

        print("VAL STATS: \n ", val_stats)
        print('-----------')

        print("TEST STATS: \n ", test_stats)


    print("all test stats: ", all_test_stats)

