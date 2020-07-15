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
import torchvision

import data
import utils
from dro_loss import LossComputer
from specified_dists import dist_sets
from binned_dists import get_binned_dists

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"


# Arguments
parser = argparse.ArgumentParser()

# Training / Optimization args
parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')

parser.add_argument('--learning_rate', type=float, default=1e-4)

parser.add_argument('--use_lr_schedule', type=int, default=0)
parser.add_argument('--weight_decay', type=float, default=1e-4)


# Distributionally Robust NNs
parser.add_argument('--use_robust_loss', type=int, default=0,
                    help='Use robust loss algo from DRNN paper')
parser.add_argument('--robust_step_size', type=float, default=0.01,
                    help='Applicable when using robust loss algo from DRNN paper')

# Model args
parser.add_argument('--context_net', type=str, default='convnet')
parser.add_argument('--pretrained', type=int, default=1,
                                   help='Pretrained resnet')
parser.add_argument('--use_context', type=int, default=0, help='Whether or not to condition the model.')

parser.add_argument('--prediction_net', type=str, default='convnet',
                    choices=['resnet18', 'resnet34', 'resnet50', 'convnet'],
                    help='')

parser.add_argument('--n_context_channels', type=int, default=3, help='Used when using a convnet/resnet')


# Evaluation
parser.add_argument('--binning', type=int, default=0)

# Data args
parser.add_argument('--dataset', type=str, default='celeba', choices=['celeba'])
parser.add_argument('--data_dir', type=str, default='../data/')

# CelebA Data (from DRNN paper)
parser.add_argument('--target_resolution', type=int, default=224,
                    help='Resize image to this size before feeding in to model')
parser.add_argument('--target_name', type=str, default='Blond_Hair',
                    help='The y value we are trying to predict')
parser.add_argument('--confounder_names', type=str, nargs='+',
                    default=['Male'],
                    help='Binary attributes from which we construct the groups. This is called confounder names \
                    for now since we are using part of Group DRO data loading')

# Data sampling
parser.add_argument('--meta_batch_size', type=int, default=2)
parser.add_argument('--support_size', type=int, default=50, help='Support size (what we call batch_size in the appendix)')
parser.add_argument('--shuffle_train', type=int, default=1,
                    help='Only relevant when do_clustered_sampling = 0 \
                    and --uniform_over_groups 0')
parser.add_argument('--loading_type', type=str, choices=['PIL', 'jpeg'], default='jpeg',
                    help='Whether to use PIL or jpeg4py when loading images. Jpeg is faster. See README for details')

# Clustered sampling
parser.add_argument('--sampling_type', type=str, default='regular',
        choices=['meta_batch_mixtures', 'meta_batch_groups', 'uniform_over_groups', 'regular'],
                    help='Sampling type')
parser.add_argument('--eval_corners_only', type=int, default=0,
                    help='Are we evaluating mixtures or corners?')
parser.add_argument('--alpha', type=float, default=[1,1,1,1], nargs=4,
                    help='The alpha values for the dirichlet distribution. All 1s gives the uniform distribution')

parser.add_argument('--uniform_over_groups', type=int, default=0,
                    help='Sample groups uniformly. This is relevant when sampling_type == meta_batch_groups')

# Evalaution
parser.add_argument('--n_test_dists', type=int, default=30,
                    help='Number of test distributions to evaluate on. These are sampled uniformly.')
parser.add_argument('--n_test_per_dist', type=int, default=1000,
                    help='Number of examples to evaluate on per test distribution')

# Logging
parser.add_argument('--seed', type=int, default=None, help='Seed')
parser.add_argument('--experiment_name', type=str, default='debug')
parser.add_argument('--epochs_per_eval', type=int, default=1)
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--log_wandb', type=int, default=1)

parser.add_argument('--num_workers', type=int, default=8, help='Num workers for pytorch data loader')
parser.add_argument('--pin_memory', type=int, default=1, help='Pytorch loader pin memory. \
                    Best practice is to use this')

parser.add_argument('--crop_type', type=int, default=0)
parser.add_argument('--crop_size_factor', type=float, default=1)
parser.add_argument('--drop_last', type=int, default=0)

args = parser.parse_args()


# Save folder
datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
ckpt_dir = Path('output') / 'checkpoints' / f'{args.experiment_name}_{args.seed}_{datetime_now}'

args.ckpt_dir = ckpt_dir

if args.log_wandb:
    tags = ['supervised', f'{args.dataset}', f'use_context_{args.use_context}']
    if args.debug: tags.append('debug')

    project_name = f"arm_{args.dataset}"

    wandb.init(name=args.experiment_name,
               project=project_name,
               tags=tags)
    wandb.config.update(args)

# For reproducibility. See https://pytorch.org/docs/stable/notes/randomness.html for details.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_one_hot(values, num_classes):
    """Helper function"""
    return np.eye(num_classes)[values]

def main():

    # Cuda
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.cuda = True
    else:
        args.device = torch.device('cpu')
        args.cuda = False

    # Make as reproducible as possible.
    # Please note that pytorch does not let us make things completely reproducible across machines.
    # See https://pytorch.org/docs/stable/notes/randomness.html
    if args.seed is not None:
        print('setting seed', args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Get data
    train_loader, train_eval_loader, val_loader, _ = data.get_loaders(args)
    args.n_groups = train_loader.dataset.n_groups

    # Sample and specify distributions to evaluate on.
    if args.binning:
        # Binning is used when doing ARM. I.e when use_context = True.
        corner_dists = np.array(get_one_hot(range(args.n_groups), args.n_groups))
        binned_groups = get_binned_dists()
        empirical_dist = [val_loader.dataset.group_dist]
        eval_dists = [corner_dists, binned_groups, empirical_dist]

    else:
        corner_dists = np.array(get_one_hot(range(args.n_groups), args.n_groups))
        empirical_dist = [val_loader.dataset.group_dist]

        if args.n_test_dists > 0: # Multiple random dists from the dirichlet when doing ARM
            random_dists = np.random.dirichlet(np.ones(args.n_groups), size=args.n_test_dists)
            eval_dists = np.concatenate([corner_dists, random_dists, empirical_dist])
        else: # Used for ERM and DRO.
            eval_dists = np.concatenate([corner_dists, empirical_dist])

    # Get model
    model = utils.get_model(args, image_shape=train_loader.dataset.image_shape)
    model = model.to(args.device)

    # Loss Fn
    if args.use_robust_loss:
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loss_computer = LossComputer(loss_fn, is_robust=True,
                                     dataset=train_loader.dataset,
                                     step_size=args.robust_step_size,
                                     device=args.device,
                                     args=args)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # The following is taken directly from Group DRNN Repo
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay)

    # Train loop
    dfs_results = []
    best_worst_case_acc = 0
    best_worst_case_acc_epoch = 0
    avg_val_acc = 0
    empirical_val_acc = 0

    selected_sub_dists = None

    for epoch in trange(args.num_epochs):
        total_loss = 0
        total_accuracy = 0
        total_examples = 0

        model.train()


        # Sample all sub distributions for the epoch (num_dists=num_batches * meat_batch_size)
        # We precompute to speed up training, such that batches can be preloaded
        if args.sampling_type == 'meta_batch_mixtures':
            num_batches = len(train_loader)

            alpha = np.array(args.alpha) * np.ones((args.n_groups, 1)) # Alpha vector for the dirichlet distribution
            group_sub_dists = np.random.dirichlet(alpha, size=(num_batches, args.meta_batch_size))
            train_loader.batch_sampler.set_group_sub_dists(group_sub_dists)

        #####################
        ##### Train Loop ####
        #####################

        for batch_id, (images, labels, group_ids) in enumerate(tqdm(train_loader, desc='train loop')):

            # Put on GPU
            images = images.to(args.device)
            labels = labels.to(args.device)

            # Forward
            logits = model(images)

            if args.use_robust_loss:
                group_ids = group_ids.to(args.device)
                loss = loss_computer.loss(logits, labels, group_ids,
                                          is_training=True)
            else:
                loss = loss_fn(logits, labels)

            # Evaluate
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            accuracy = np.mean(preds == labels.detach().cpu().numpy().reshape(-1))
            total_accuracy += accuracy * labels.shape[0]
            total_loss += loss.item() * labels.shape[0]
            total_examples += labels.shape[0]

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # Decay learning rate after one epoch
        if args.use_lr_schedule:
            if (args.dataset == 'celeba' and epoch == 0):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-5

        if args.log_wandb:
            wandb.log({"Train Loss": total_loss / total_examples,
                       "Train Accuracy": total_accuracy / total_examples, "epoch": epoch})

        if epoch % args.epochs_per_eval == 0:
            worst_case_acc, avg_acc, empirical_case_acc, _  = utils.evaluate_mixtures(args, model,
                                                        val_loader, eval_dists, epoch, split='val')

            print(f"Epoch {epoch}, Worst Case Acc {worst_case_acc}")

            # Track early stopping values with respect to worst case.
            if worst_case_acc > best_worst_case_acc:
                best_worst_case_acc = worst_case_acc
                best_worst_case_acc_epoch = epoch
                avg_val_acc = avg_acc
                empirical_val_acc = empirical_case_acc

                save_model(model, ckpt_dir, epoch, args.device)

            # Log early stopping values
            if args.log_wandb:
                wandb.log({"Best Worst Case Val Acc": best_worst_case_acc,
                           "Best worst Case Acc Epoch": best_worst_case_acc_epoch,
                           "Early Stop (WC). Average Val Acc": avg_val_acc,
                           "Early Stop (WC). Empirical Val Acc": empirical_val_acc,
                           "epoch": epoch})


def save_model(model, ckpt_dir, epoch, device):
    """Save model"""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f'{epoch}_weights.pkl'
    model_state = model.to('cpu').state_dict(),
    torch.save(model_state, ckpt_path)

    # Overwrite the best weights with the latest
    ckpt_path = ckpt_dir / f'best_weights.pkl'
    torch.save(model_state, ckpt_path)
    model.to(device)

if __name__ == "__main__":
    main()
