Updated repo to reflect latest version of the paper is coming soon.


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

##### DRNN
`SEED=0`<br/>
`python train_on_groups.py --dataset mnist --sampling_type uniform_over_groups --use_context 0 --use_robust_loss 1 --experiment_name dro_uniform_over_groups_$SEED --epochs_per_eval 10 --num_epochs 200 --seed $SEED --n_test_per_dist 300 --meta_batch_size 6 --epochs_per_eval 10 --log_wandb 0`

##### ARM (Ours)
`SEED=0`<br/>
`python train_on_groups.py --dataset mnist --sampling_type meta_batch_groups --uniform_over_groups 1 --use_context 1 --experiment_name mnist_arm_$SEED --epochs_per_eval 10 --num_epochs 200 --seed $SEED --n_test_per_dist 300 --meta_batch_size 6 --epochs_per_eval 10 --log_wandb 0`

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

### 1. Download CelebA

1. Download data from https://www.kaggle.com/jessicali9530/celeba-dataset/data#
2. Create a folder names data ../ in relation to your repo.
2. Put the celeba folder inside the data folder. This should be your folder structure
    - ../data/celeba/
        - img_align_celeba/
        - list_attr_celeba.csv
        - xxx.csv
        - ...

You can also specifiy data directory with `--data_dir /path/to/data/` in case you want to put the data somewhere else.

### 2. Libjpeg-Turbo for data loading
To speed things up we use jpeg4py instead of PIL to load images from disc.

 For this to work you need to install libjpeg-turbo by running:
 `sudo apt-get install libturbojpeg`

 If you don't have sudo access you can run with the flag:

 `--data_loading PIL` and you will load images using PIL which will be slightly slower.

 Note that jpeg4py loading was used for all CelebA experiments and may affect results.

### 3. Train

#### Pretrained

##### Upweighting baseline
`SEED=0`<br/>
`python train_supervised.py --dataset celeba --prediction_net resnet50 --use_context 0 --meta_batch_size 2 --support_size 50 --sampling_type uniform_over_groups --experiment_name celeba_erm_$SEED --num_epochs 50 --n_test_dists 0 --epochs_per_eval 1 --n_test_per_dist 1500  --weight_decay 0.5 --use_lr_schedule 1 --log_wandb 0 --seed $SEED`

##### ERM
`SEED=0`<br/>
`python train_supervised.py --dataset celeba --prediction_net resnet50 --use_context 0 --meta_batch_size 2 --support_size 50 --sampling_type regular --experiment_name celeba_erm_$SEED --num_epochs 50 --n_test_dists 0 --epochs_per_eval 1 --n_test_per_dist 1500  --weight_decay 0.5 --use_lr_schedule 1 --log_wandb 0 --seed $SEED`

##### DRNN
`SEED=0`<br/>
`python train_supervised.py --dataset celeba  --prediction_net resnet50 --use_context 0 --meta_batch_size 2 --support_size 50 --sampling_type uniform_over_groups --experiment_name celeba_dro_$SEED --num_epochs 50 --n_test_dists 0 --epochs_per_eval 1 --n_test_per_dist 1500  --weight_decay 0.5 --use_robust_loss 1 --use_lr_schedule 1 --log_wandb 0 --seed $SEED`

##### ARM (ours)
`SEED=0`<br/>
`python train_supervised.py --dataset celeba  --prediction_net resnet50 --use_context 1 --meta_batch_size 2 --support_size 50 --sampling_type meta_batch_mixtures --experiment_name celeba_arm_$SEED --num_epochs 50 --n_test_dists 30 --epochs_per_eval 1 --n_test_per_dist 1500  --weight_decay 0.5 --use_lr_schedule 1 --binning 1 --log_wandb 0 --seed  $SEED`

### 5. Evaluate

Your trained models are saved in output/checkpoints/{experiment_name}_{datetime}/

An example of checkpoint could be be:
- output/checkpoints/celeba_dro_0_20200529-130211/best_weights.pkl

In the above example, we have that CKPT_FOLDER = celeba_dro_0_20200529-130211.

To evaluate a set of models, you run:
To evaluate a set of checkpoints, you run the following for ERM & DRNN:
`python test.py --eval_on test --eval_deterministic 1 --log_wandb 0 --ckpt_folders CKPT_FOLDER1 CKPT_FOLDER2 CKPT_FOLDER3`

To evaluate a set of checkpoints for ARM you run (15 test distributions per bin)
`python test.py --eval_on test --use_context 1 --binning 1 --log_wandb 0 --ckpt_folders CKPT_FOLDER1 CKPT_FOLDER2 CKPT_FOLDER3 --n_test_dists 15`

When evaluating without pretraining, make sure to set crop_type=2. You can vary support size with --support_size.

## Known Issues

 - "Object too deep for desired array"
    - You may get this error, when sampling from the dirichlet in the train script:
 - Solution: 'pip install -U numpy'. Update to the latest numpy, 1.18 or 1.17. This may in turn cause a problem with albumentations.
 This is solved by downloading the latest skimage. 'pip install -U scikit-image'
