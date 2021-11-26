from datetime import datetime
from pathlib import Path
import random

import numpy as np
import argparse
import wandb
import torch

import train
import data

def test(args, algorithm, seed, eval_on):

    # Get data
    train_loader, train_eval_loader, val_loader, test_loader = data.get_loaders(args)

    stats = {}
    loaders = {'train': train_eval_loader,
                'val': val_loader,
                'test': test_loader}
    for split in eval_on:
        set_seed(seed + 10, args.cuda)
        loader = loaders[split]
        split_stats = train.eval_groupwise(args, algorithm, loader, split=split, n_samples_per_group=args.test_n_samples_per_group)
        stats[split] = split_stats

    return stats


def get_parser():
    # Arguments
    parser = argparse.ArgumentParser()

    # Train / test
    parser.add_argument('--train', type=int, default=1, help="Train models")
    parser.add_argument('--test', type=int, default=1, help="Test models")
    parser.add_argument('--ckpt_folders', type=str, nargs='+') # only applicable when train is 0 and test is 1

    parser.add_argument('--progress_bar', type=int, default=0, help="Test models")

    # Training / Optimization args
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)

    # Data args
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'femnist', 'cifar-c', 'tinyimg'])
    parser.add_argument('--data_dir', type=str, default='../data/')


    # Data sampling
    parser.add_argument('--sampler', type=str, default='standard',
            choices=['standard', 'group'],
                        help='Standard or group sampler')
    parser.add_argument('--uniform_over_groups', type=int, default=0,
                        help='Sample across groups uniformly')
    parser.add_argument('--meta_batch_size', type=int, default=2, help='Number of classes')
    parser.add_argument('--support_size', type=int, default=50, help='Support size: same as what we call batch size in the appendix')
    parser.add_argument('--shuffle_train', type=int, default=1,
                        help='Only relevant when no group sampling = 0 \
                        and --uniform_over_groups 0')
    parser.add_argument('--drop_last', type=int, default=0)
    parser.add_argument('--loading_type', type=str, choices=['PIL', 'jpeg'], default='jpeg',
                        help='Whether to use PIL or jpeg4py when loading images. Jpeg is faster. See README for deatiles')

    parser.add_argument('--num_workers', type=int, default=8, help='Num workers for pytorch data loader')
    parser.add_argument('--pin_memory', type=int, default=1, help='Pytorch loader pin memory. \
                        Best practice is to use this')

    # Model args
    parser.add_argument('--model', type=str, default='convnet',
                        choices=['resnet50', 'convnet'])
    parser.add_argument('--pretrained', type=int, default=1,
                                       help='Pretrained resnet')

    # Method
    parser.add_argument('--algorithm', type=str, default='ERM', choices=['ERM', 'DRNN', 'ARM-CML', 'ARM-BN', 'ARM-LL', 'DANN', 'MMD'])

    # ARM-CML
    parser.add_argument('--n_context_channels', type=int, default=3, help='Used when using a convnet/resnet')
    parser.add_argument('--context_net', type=str, default='convnet')
    parser.add_argument('--pret_add_channels', type=int, default=1)
    parser.add_argument('--adapt_bn', type=int, default=0)


    # Evalaution
    parser.add_argument('--n_samples_per_group', type=int, default=None,
                        help='Number of examples to evaluate on per test distribution')
    parser.add_argument('--test_n_samples_per_group', type=int, default=None,
                        help='Number of examples to evaluate on per test distribution')
    parser.add_argument('--epochs_per_eval', type=int, default=1)

    # Test
    parser.add_argument('--eval_on', type=str, nargs="*", default=['test'])

    # DANN
    parser.add_argument('--lambd', type=float, default=0.01)
    parser.add_argument('--d_steps_per_g_step', type=int, default=1)

    # Logging
    parser.add_argument('--seeds', type=int, nargs="*", default=[0], help='Seeds')
    parser.add_argument('--plot', type=int, default=0, help='Plot or not')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--log_wandb', type=int, default=0)


    return parser


def set_seed(seed, cuda):

    # Make as reproducible as possible.
    # Please note that pytorch does not let us make things completely reproducible across machines.
    # See https://pytorch.org/docs/stable/notes/randomness.html
    print('setting seed', seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class ScoreKeeper:


    def __init__(self, splits, n_seeds):

        self.splits = splits
        self.n_seeds = n_seeds

        self.results = {}
        for split in splits:
            self.results[split] = {}

    def log(self, stats):
        for split in stats:
            split_stats = stats[split]
            for key in split_stats:
                value = split_stats[key]
                metric_name = key.split('/')[1]

                if metric_name not in self.results[split]:
                    self.results[split][metric_name] = []

                self.results[split][metric_name].append(value)

    def print_stats(self, metric_names=['worst_case_acc', 'average_acc', 'empirical_acc']):

        for split in self.splits:
            print("Split: ", split)

            for metric_name in metric_names:

                values = np.array(self.results[split][metric_name])
                avg = np.mean(values)
                standard_error =  np.std(values) / np.sqrt(self.n_seeds - 1)

                print(f"{metric_name}: {avg}, standard error: {standard_error}")


if __name__ == '__main__':

    start_time = datetime.now()

    args = get_parser().parse_args()

    # Cuda
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.cuda = True
    else:
        args.device = torch.device('cpu')
        args.cuda = False

    # For reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.train:

        score_keeper = ScoreKeeper(args.eval_on, len(args.seeds))
        print("args seeds: ", args.seeds)
        ckpt_dirs = []

        for ind, seed in enumerate(args.seeds):
            print("seeeed: ", seed)
            set_seed(seed, args.cuda)
            tags = ['supervised', args.dataset, args.algorithm]

            # Save folder
            datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
            name = args.dataset + args.exp_name + '_' + str(seed)
            args.ckpt_dir = Path('output') / 'checkpoints' / f'{name}_{datetime_now}'
            ckpt_dirs.append(args.ckpt_dir)
            print("CKPT DIR: ", args.ckpt_dir)

            if args.debug: tags.append('debug')

            if args.log_wandb:
                if ind != 0:
                    wandb.join()
                run = wandb.init(name=name,
                           project=f"arm_{args.dataset}",
                           tags=tags,
                           allow_val_change=True,
                           reinit=True)
                wandb.config.update(args, allow_val_change=True)

            train.train(args)

            # Test the model just trained on
            if args.test:
                args.ckpt_path = args.ckpt_dir / f'best.pkl'
                algorithm = torch.load(args.ckpt_path).to(args.device)
                stats = test(args, algorithm, seed, eval_on=args.eval_on)
                score_keeper.log(stats)


        print("Ckpt dirs: \n ", ckpt_dirs)
        score_keeper.print_stats()

    elif args.test and args.ckpt_folders: # test a set of already trained models

        # Check if checkpoints exist
        for ckpt_folder in args.ckpt_folders:
            ckpt_path = Path('output') / 'checkpoints' / ckpt_folder / f'best.pkl'
            algorithm = torch.load(ckpt_path)
            print("Found: ", ckpt_path)

        score_keeper = ScoreKeeper(args.eval_on, len(args.ckpt_folders))
        for i, ckpt_folder in enumerate(args.ckpt_folders):

            # test algorithm
            seed = args.seeds[i]
            args.ckpt_path = Path('output') / 'checkpoints' / ckpt_folder / f'best.pkl' # final_weights.pkl
            algorithm = torch.load(args.ckpt_path).to(args.device)
            stats = test(args, algorithm, seed, eval_on=args.eval_on)
            score_keeper.log(stats)

        score_keeper.print_stats()


    end_time = datetime.now()
    runtime = (end_time - start_time).total_seconds() / 60.0
    print("\nTotal runtime: ", runtime)
