
# Adaptive Risk Minimization: A Meta-Learning Approach for Tackling Group Shift

* Environment
* Logging Results
* MNIST Experiment
    * Train
    * Evaluate
* FEMNIST Experiment
    * Download data
    * Train
    * Evaluate
* CelebA Experiment
    * Download CelebA
    * Libjpeg-Turbo for data loading
    * Train
    * Evaluate

## Environment

Using pip
 - `pip install -r requirements.txt` or `pip3 install -r requirements.txt` (recommended)

Using conda
 - `conda env create -f environment.yml`
 - `conda env list`
 - `source activate arm`

## Logging results.
Weights and Biases, which is an alternative to Tensorboard, is used to log results in the cloud. This is used for both training and evaluating on the test set.
To get it running quickly without WandB, we have set --log_wandb 0 below. Much of the results will be printed in the console. We recommend using WandB which is free for researchers.

## MNIST Experiments Setup

### 1. Train

##### Upweighting baseline
`SEED=0` <br/>
`python train_on_groups.py --dataset mnist --sampling_type uniform_over_groups --experiment_name mnist_upweighted_$SEED --num_epochs 200 --n_test_per_dist 300 --epochs_per_eval 10 --seed $SEED --meta_batch_size 6 --epochs_per_eval 10 --log_wandb 0`

##### ERM
`SEED=0`<br/>
`python train_on_groups.py --dataset mnist --sampling_type regular --experiment_name mnist_erm_regular_$SEED --n_test_per_dist 300 --num_epochs 200 --epochs_per_eval 10 --seed $SEED --n_test_per_dist 300 --meta_batch_size 6 --epochs_per_eval 10 --use_context 0 --drop_last 0 --log_wandb 0`

##### DRNN
`SEED=0`<br/>
`python train_on_groups.py --dataset mnist --sampling_type uniform_over_groups --use_context 0 --use_robust_loss 1 --experiment_name dro_uniform_over_groups_$SEED --epochs_per_eval 10 --num_epochs 200 --seed $SEED --n_test_per_dist 300 --meta_batch_size 6 --epochs_per_eval 10 --log_wandb 0`

##### ARM (Ours)
`SEED=0`<br/>
`python train_on_groups.py --dataset mnist --sampling_type meta_batch_groups --uniform_over_groups 1 --use_context 1 --experiment_name mnist_arm_$SEED --epochs_per_eval 10 --num_epochs 200 --seed $SEED --n_test_per_dist 300 --meta_batch_size 6 --epochs_per_eval 10 --log_wandb 0`

#### Context ablation
`SEED=0`<br/>
`python train_on_groups.py --sampling_type uniform_over_groups --experiment_name mnist_random_context_$SEED --n_test_per_dist 300 --num_epochs 200 --epochs_per_eval 10 --seed $SEED --use_context 1 --meta_batch_size 6 --epochs_per_eval 10 --log_wandb 0`

### 2. Evaluate

Your trained models are saved in output/checkpoints/{experiment_name}_{datetime}/

An example of checkpoint could be:
- output/checkpoints/mnist_erm_upweighted_0_20200529-130211/best_weights.pkl

To evaluate a set of checkpoints, you run:
`python test_groups.py --eval_on test --ckpt_folders CKPT_FOLDER1 CKPT_FOLDER2 CKPT_FOLDER3 --log_wandb 0`

E.g., you could run
`python test_groups.py --eval_on test --ckpt_folders mnist_erm_upweighted_0_1231414 mnist_erm_upweighted_1_1231434 mnist_erm_upweighted_2_1231414 --log_wandb 0`

--ckpt_folders is a list of the folders

When evaluating ARM, you need to set --use_context 1.

You can vary support size with --support_size.


## FEMNIST Experiments Setup

### 1. Download Data

- Follow the instructions at `https://github.com/TalwalkarLab/leaf`
- This will entail cloning their repo, and running the preprocessing code in `data/femnist/preprocess.sh`
- Specifically you run `./preprocess.sh -s niid --sf 0.1 -k 100 -t user --smplseed 0 --spltseed 0`
- Note: this may not give the exact same split as in the paper.
- Move the femnist data folder to your data folder `../data/` as described for CelebA. As with CelebA you can also set `data_dir` to point towards you data dir.
- Create validation by creating a folder val (in addition the train and test that is automatically generated). Move the first file in train to val.

### 2. Train.

#### ERM
`SEED=0`<br/>
`python train_on_groups.py --dataset femnist --model ContextualConvNet --use_context 0 --pretrained 0 --experiment_name femnist_erm_regular_$SEED --meta_batch_size 2 --support_size 50 --sampling_type regular --num_epochs 200 --epochs_per_eval 1 --n_test_per_dist 2000 --optimizer sgd --seed $SEED --log_wandb 1`

#### ARM
`SEED=0`<br/>
`python train_on_groups.py --dataset femnist --model ContextualConvNet --use_context 1 --n_context_channels 1 --pretrained 0 --experiment_name femnist_arm_$SEED --meta_batch_size 2 --support_size 50 --sampling_type meta_batch_groups --uniform_over_groups 1 --num_epochs 200 --epochs_per_eval 1 --n_test_per_dist 2000 --optimizer sgd --seed $SEED --log_wandb 1`


### 3. Evaluate
`python test_on_groups.py --dataset femnist --eval_on test --log_wandb 0 --ckpt_folders CKPT_FOLDER1 CKPT_FOLDER2 CKPT_FOLDER3`

## CelebA Experiments Setup

Coming soon.


## Known Issues

 - "Object too deep for desired array"
    - You may get this error, when sampling from the dirichlet in the train script:
 - Solution: 'pip install -U numpy'. Update to the latest numpy, 1.18 or 1.17. This may in turn cause a problem with albumentations.
 This is solved by downloading the latest skimage. 'pip install -U scikit-image'
