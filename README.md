*Henrik will be offline Dec 2 - Dec 14 so responses to some questions may be slower during this time period.*

# Adaptive Risk Minimization: Learning to Adapt to Domain Shift
*Code for the upcoming Neurips 2021 paper. Some code and hyperparameters differ from the current arxiv version (v3).*

The structure of this repo and the way certain details around the training loop and evaluation loop is set up is inspired by and adapted from the [DomainBed repo](https://github.com/facebookresearch/DomainBed/tree/main/domainbed) and the [Wilds repo](https://github.com/p-lambda/wilds).

* Environment
* Logging Results
* Experiments Setup
    * Train
    * Evaluate

## Environment

python version: 3.6.5

Using pip
 - `pip install -r requirements.txt` or `pip3 install -r requirements.txt`

## Logging results.
Weights and Biases, which is an alternative to Tensorboard, is used to log results in the cloud. This is used for both training and evaluating on the test set.
To get it running quickly without WandB, we have set --log_wandb 0 below. Much of the results will be printed in the console. We recommend using WandB which is free for researchers.

## Data

Femnist
The train/val/test data split used in the paper can be found here: https://drive.google.com/file/d/1xvT13Sl3vJIsC2I7l7Mp8alHkqKQIXaa/view?usp=sharing

CIFAR-C
- Test data can be downloaded here: https://zenodo.org/record/2535967#.YCUsMukzZ0s
- The training and validation split used in the paper can be found here: https://drive.google.com/file/d/1blM7LHGR62-dVJjNAScsJMlzeiQS9DX1/view?usp=sharing

TinyImg
- Test data can be downloaded here: https://zenodo.org/record/2536630#.YCUsBOkzZ0s
- The training and validation split used in the paper can be found here: https://drive.google.com/file/d/13hd39InVa5WqPUpuoJtl9kSSwyDFyFNc/view?usp=sharing

## Experiments Setup

Showing example args for MNIST here. See all_commands.sh for more details.

### 1. Train

##### Shared args
```
SEEDS="0"
SHARED_ARGS="--dataset mnist --num_epochs 200 --n_samples_per_group 300 --epochs_per_eval 10 --seeds ${SEEDS} --meta_batch_size 6 --epochs_per_eval 10 --log_wandb 0 --train 1"
```

##### ERM
```
python run.py --exp_name erm $SHARED_ARGS
```

##### UW (Upweighted)
```
python run.py --uniform_over_groups 1 --exp_name uw $SHARED_ARGS
```

##### DRNN (Distributionally Robust Neural Networks)
```
python run.py --algorithm drnn --uniform_over_groups 1 --exp_name drnn $SHARED_ARGS
```

##### ARM-CML (Adaptive Risk Minimization - Contextual Meta-learner)
```
python run.py --algorithm ARM-CML --sampler group --uniform_over_groups 1 --n_context_channels 12 --exp_name arm_cml $SHARED_ARGS
```

##### ARM-LL (Adaptive Risk Minimization - Learned Loss)
```
python run.py --algorithm ARM-LL --sampler group --uniform_over_groups 1 --exp_name arm_ll $SHARED_ARGS
```

##### ARM-BN (Adaptive Risk Minimization - Batchnorm)
```
python run.py --algorithm ARM-LL --sampler group --uniform_over_groups 1 --exp_name arm_bn $SHARED_ARGS
```

#### CML ablation
```
python run.py --algorithm ARM-CML --sampler regular --experiment_name cml_ablation $SHARED_ARGS
```

### 2. Evaluate

Your trained models are saved in `output/checkpoints/{dataset}_{exp_name}_{seed}_{datetime}/`

An example of checkpoint could be:
- `output/checkpoints/mnist_erm_0_20200529-130211/best_weights.pkl`

To evaluate a set of checkpoints, you run:
```
python run.py --eval_on test --test 1 --train 0 --ckpt_folders CKPT_FOLDER1 CKPT_FOLDER2 CKPT_FOLDER3 --log_wandb 0`
```

E.g., you could run
```
python run.py --eval_on test --test 1 -- train 0 --ckpt_folders mnist_erm_0_1231414 mnist_erm_1_1231434 mnist_erm_2_2_1231414 --log_wandb 0`
```

`--ckpt_folders` is a list of the folders

You can vary support size with `--support_size`.

