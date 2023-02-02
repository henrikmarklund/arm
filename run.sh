#!/bin/bash

#SBATCH --job-name=arm5
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8

source /home/${USER}/.bashrc
conda activate maicon


SEEDS="0"
SHARED_ARGS="\
    --dataset mnist \
    --num_epochs 200 \
    --eval_on val test \
    --n_samples_per_group 300 \
    --seeds ${SEEDS} \
    --meta_batch_size 6 \
    --epochs_per_eval 10 \
    --optimizer adam \
    --learning_rate 1e-4 \
    --weight_decay 0 \
    --log_wandb 0 \
    --train 1 \
    --support_size 5
    "

srun python run.py --algorithm ARM-CML --sampler group --uniform_over_groups 1 --n_context_channels 12 --exp_name arm_cml $SHARED_ARGS
# python run.py --eval_on test --test 1 --train 0 --ckpt_folders mnistarm_cml_0_20230123-210851 --log_wandb 0  ## train 50

# python run.py --eval_on test --test 1 --train 0 --ckpt_folders mnistarm_cml_0_20230125-222512 --log_wandb 0 --support_size 50 ## train 1
# python run.py --eval_on test --test 1 --train 0 --ckpt_folders mnistarm_cml_0_20230125-223530 --log_wandb 0  ## train 5

