import os
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange, tqdm
import wandb

import data
import utils
from algorithm import init_algorithm

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"


####################
###### TRAIN #######
####################

def run_epoch(algorithm, loader, train, progress_bar=True):

    epoch_labels = []
    epoch_logits = []
    epoch_group_ids = []

    if progress_bar:
        loader = tqdm(loader, desc=f'{"train" if train else "eval"} loop')


    for images, labels, group_ids in loader:

        # Put on GPU
        images = images.to(algorithm.device)
        labels = labels.to(algorithm.device)



        # Forward
        if train:
            logits, batch_stats = algorithm.learn(images, labels, group_ids)
            if logits is None: # DANN
                continue
        else:
            logits = algorithm.predict(images)

        epoch_labels.append(labels.to('cpu').clone().detach())
        epoch_logits.append(logits.to('cpu').clone().detach())
        epoch_group_ids.append(group_ids.to('cpu').clone().detach())

    return torch.cat(epoch_logits), torch.cat(epoch_labels), torch.cat(epoch_group_ids)

def train(args):

    # Get data
    train_loader, _, val_loader, _ = data.get_loaders(args)
    args.n_groups = train_loader.dataset.n_groups

    algorithm = init_algorithm(args, train_loader.dataset)
    saver = utils.Saver(algorithm, args.device, args.ckpt_dir)

    # Train loop
    best_worst_case_acc = 0

    for epoch in trange(args.num_epochs):
        epoch_logits, epoch_labels, epoch_group_ids = run_epoch(algorithm, train_loader, train=True, progress_bar=args.progress_bar)

        if epoch % args.epochs_per_eval == 0:
            stats = eval_groupwise(args, algorithm, val_loader, epoch, split='val', n_samples_per_group=args.n_samples_per_group)

            # Track early stopping values with respect to worst case.
            if stats['val/worst_case_acc'] > best_worst_case_acc:
                best_worst_case_acc = stats['val/worst_case_acc']
                saver.save(epoch, is_best=True)


            # Log early stopping values
            if args.log_wandb:
                wandb.log({"val/best_worst_case_acc": best_worst_case_acc})

            print(f"\nEpoch: ", epoch, "\nWorst Case Acc: ", stats['val/worst_case_acc'])

##############################
###### Evaluate / Test #######
##############################

def get_group_iterator(loader, group, support_size, n_samples_per_group=None):
    example_ids = np.nonzero(loader.dataset.group_ids == group)[0]
    example_ids = example_ids[np.random.permutation(len(example_ids))] # Shuffle example ids

    # Create batches
    batches = []
    X, Y, G = [], [], []
    counter = 0
    for i, idx in enumerate(example_ids):
        x, y, g = loader.dataset[idx]
        X.append(x); Y.append(y); G.append(g)
        if (i + 1) % support_size == 0:
            X, Y, G = torch.stack(X), torch.tensor(Y, dtype=torch.long), torch.tensor(G, dtype=torch.long)
            batches.append((X, Y, G))
            X, Y, G = [], [], []

        if n_samples_per_group is not None and i == (n_samples_per_group - 1):
            break
    if X:
        X, Y, G = torch.stack(X), torch.tensor(Y, dtype=torch.long), torch.tensor(G, dtype=torch.long)
        batches.append((X, Y, G))

    return batches

def eval_groupwise(args, algorithm, loader, epoch=None, split='val', n_samples_per_group=None):
    """ Test model on groups and log to wandb

        Separate script for femnist for speed."""

    groups = []
    accuracies = np.zeros(len(loader.dataset.groups))
    num_examples = np.zeros(len(loader.dataset.groups))

    if args.adapt_bn:
        algorithm.train()
    else:
        algorithm.eval()

    # Loop over each group
    for i, group in tqdm(enumerate(loader.dataset.groups), desc='Evaluating', total=len(loader.dataset.groups)):
        counter = 0
        group_iterator = get_group_iterator(loader, group, args.support_size, n_samples_per_group)

        logits, labels, group_ids = run_epoch(algorithm, group_iterator, train=False, progress_bar=False)
        preds = np.argmax(logits, axis=1)

        # Evaluate
        accuracy = np.mean((preds == labels).numpy())
        num_examples[group] = len(labels)
        accuracies[group] = accuracy

        if args.log_wandb:
            if epoch is None:
                wandb.log({f"{split}/acc": accuracy, # Gives us Acc vs Group Id
                           f"{split}/group_id": group})
            else:
                wandb.log({f"{split}/acc_e{epoch}": accuracy, # Gives us Acc vs Group Id
                           f"{split}/group_id": group})

    # Log worst, average and empirical accuracy
    worst_case_acc = np.amin(accuracies)
    worst_case_group_size = num_examples[np.argmin(accuracies)]

    num_examples = np.array(num_examples)
    props = num_examples / num_examples.sum()
    empirical_case_acc = accuracies.dot(props)
    average_case_acc = np.mean(accuracies)

    total_size = num_examples.sum()

    stats = {
                f'{split}/worst_case_acc': worst_case_acc,
                f'{split}/worst_case_group_size': worst_case_group_size,
                f'{split}/average_acc': average_case_acc,
                f'{split}/total_size': total_size,
                f'{split}/empirical_acc': empirical_case_acc
            }

    if epoch is not None:
        stats['epoch'] = epoch

    if args.log_wandb:
        wandb.log(stats)

    return stats

