
# MNIST
SEEDS="0 1 2"
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
    --log_wandb 0"


N_CONTEXT_CHANNELS=12 # For CML
python run.py --exp_name erm ${SHARED_ARGS}
python run.py --exp_name uw --uniform_over_groups 1 ${SHARED_ARGS}
python run.py --exp_name drnn --algorithm DRNN --uniform_over_groups 1  $SHARED_ARGS
python run.py --exp_name mmd --algorithm MMD --sampler group --uniform_over_groups 1 $SHARED_ARGS
python run.py --exp_name dann --algorithm DANN $SHARED_ARGS
python run.py --exp_name arm-cml --algorithm ARM-CML --sampler group --uniform_over_groups 1 --n_context_channels $N_CONTEXT_CHANNELS  $SHARED_ARGS
python run.py --exp_name arm-ll --algorithm ARM-LL --sampler group --uniform_over_groups 1  $SHARED_ARGS
python run.py --exp_name arm-bn --algorithm ARM-BN --sampler group --uniform_over_groups 1  $SHARED_ARGS
python run.py --exp_name cml-ablation --algorithm ARM-CML --n_context_channels $N_CONTEXT_CHANNELS  $SHARED_ARGS
python run.py --exp_name ll-ablation --algorithm ARM-LL $SHARED_ARGS
python run.py --exp_name bn-ablation --algorithm ARM-BN $SHARED_ARGS


# FEMNIST
SEEDS="0 1 2"
SHARED_ARGS="\
    --dataset femnist \
    --num_epochs 200 \
    --eval_on val test \
    --seeds ${SEEDS} \
    --meta_batch_size 2 \
    --epochs_per_eval 1 \
    --optimizer sgd \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --log_wandb 0"

N_CONTEXT_CHANNELS=1 # For CML

python run.py --exp_name erm ${SHARED_ARGS}
python run.py --exp_name uw --uniform_over_groups 1 ${SHARED_ARGS}
python run.py --exp_name drnn --algorithm DRNN --uniform_over_groups 1  $SHARED_ARGS
python run.py --exp_name mmd --algorithm MMD --sampler group --uniform_over_groups 1 $SHARED_ARGS
#python run.py --exp_name dann --algorithm DANN $SHARED_ARGS # run separately with optimizer=adam
python run.py --exp_name arm-cml --algorithm ARM-CML --sampler group --uniform_over_groups 1 --n_context_channels $N_CONTEXT_CHANNELS  $SHARED_ARGS
python run.py --exp_name arm-ll --algorithm ARM-LL --sampler group --uniform_over_groups 1  $SHARED_ARGS
python run.py --exp_name arm-bn --algorithm ARM-BN --sampler group --uniform_over_groups 1  $SHARED_ARGS
python run.py --exp_name cml-ablation --algorithm ARM-CML --n_context_channels $N_CONTEXT_CHANNELS  $SHARED_ARGS
python run.py --exp_name ll-ablation --algorithm ARM-LL $SHARED_ARGS
python run.py --exp_name bn-ablation --algorithm ARM-BN $SHARED_ARGS

# CIFAR-C
SEEDS="0 1 2"
SHARED_ARGS="\
    --dataset cifar-c \
    --num_epochs 100 \
    --n_samples_per_group 2000 \
    --test_n_samples_per_group 3000 \
    --eval_on val test \
    --seeds ${SEEDS} \
    --meta_batch_size 3 \
    --support_size 100 \
    --epochs_per_eval 1 \
    --optimizer sgd \
    --learning_rate 1e-2 \
    --weight_decay 1e-4 \
    --log_wandb 0"

N_CONTEXT_CHANNELS=3 # For CML

python run.py --exp_name erm ${SHARED_ARGS}
python run.py --exp_name drnn --algorithm DRNN --uniform_over_groups 1  $SHARED_ARGS
python run.py --exp_name mmd --algorithm MMD --sampler group --uniform_over_groups 1 $SHARED_ARGS
python run.py --exp_name dann --algorithm DANN $SHARED_ARGS
python run.py --exp_name arm-cml --algorithm ARM-CML --sampler group --uniform_over_groups 1 --n_context_channels $N_CONTEXT_CHANNELS --adapt_bn 1  $SHARED_ARGS
python run.py --exp_name arm-ll --algorithm ARM-LL --sampler group --uniform_over_groups 1  $SHARED_ARGS
python run.py --exp_name arm-bn --algorithm ARM-BN --sampler group --uniform_over_groups 1  $SHARED_ARGS
python run.py --exp_name cml-ablation --algorithm ARM-CML --n_context_channels --adapt_bn 1 $N_CONTEXT_CHANNELS  $SHARED_ARGS
python run.py --exp_name ll-ablation --algorithm ARM-LL $SHARED_ARGS
python run.py --exp_name bn-ablation --algorithm ARM-BN $SHARED_ARGS



# Tiny ImageNet-C
SEEDS="0 1 2"
SHARED_ARGS="\
    --dataset tinyimg \
    --num_epochs 50 \
    --n_samples_per_group 2000 \
    --test_n_samples_per_group 3000 \
    --eval_on val test \
    --seeds ${SEEDS} \
    --meta_batch_size 3 \
    --support_size 100 \
    --model resnet50 \
    --epochs_per_eval 1 \
    --optimizer sgd \
    --learning_rate 1e-2 \
    --weight_decay 1e-4 \
    --log_wandb 0"

N_CONTEXT_CHANNELS=3 # For CML

python run.py --exp_name erm ${SHARED_ARGS}
python run.py --exp_name drnn --algorithm DRNN --uniform_over_groups 1  $SHARED_ARGS
python run.py --exp_name mmd --algorithm MMD --sampler group --uniform_over_groups 1 $SHARED_ARGS
python run.py --exp_name dann --algorithm DANN $SHARED_ARGS
python run.py --exp_name arm-cml --algorithm ARM-CML --sampler group --uniform_over_groups 1 --n_context_channels $N_CONTEXT_CHANNELS --adapt_bn 1 $SHARED_ARGS
python run.py --exp_name arm-ll --algorithm ARM-LL --sampler group --uniform_over_groups 1  $SHARED_ARGS
python run.py --exp_name arm-bn --algorithm ARM-BN --sampler group --uniform_over_groups 1  $SHARED_ARGS
python run.py --exp_name cml-ablation --algorithm ARM-CML --n_context_channels $N_CONTEXT_CHANNELS  --adapt_bn 1 $SHARED_ARGS
python run.py --exp_name ll-ablation --algorithm ARM-LL $SHARED_ARGS
python run.py --exp_name bn-ablation --algorithm ARM-BN $SHARED_ARGS
