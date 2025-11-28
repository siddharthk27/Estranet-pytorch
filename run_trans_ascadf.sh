#!/bin/bash

# Experiment (data/checkpoint/directory) config
DATA_PATH=""  # Path to the .h5 file containing the dataset
DATASET="ASCAD"
CKP_DIR="./"
RESULT_PATH="results"

# Optimization config
# This syntax means: Use the env variable if set, otherwise use default
LEARNING_RATE=${LEARNING_RATE:-0.00025}
CLIP=0.25
MIN_LR_RATIO=0.004
INPUT_LENGTH=10000  # or 40000
DATA_DESYNC=200     # 400 for input length 40000

# Training config
TRAIN_BSZ=16
EVAL_BSZ=16
TRAIN_STEPS=400000
WARMUP_STEPS=100000
ITERATIONS=2000
SAVE_STEPS=40000

# Model config
N_LAYER=2
D_MODEL=128
D_HEAD=32
N_HEAD=8
D_INNER=256
N_HEAD_SM=8
D_HEAD_SM=16
DROPOUT=0.05
CONV_KERNEL_SIZE=3  # The kernel size of the first convolutional layer is set to 11
                    # This hyper-parameter sets the kernel size of the remaining
                    # convolutional layers
N_CONV_LAYER=2
POOL_SIZE=20        # 8
D_KERNEL_MAP=512
BETA_HAT_2=150
MODEL_NORM='preLC'
HEAD_INIT='forward'

# Evaluation config
MAX_EVAL_BATCH=100

# Early stopping config
EARLY_STOPPING_PATIENCE=10      # Stop if no improvement for 10 evaluations
EARLY_STOPPING_DELTA=0.0001     # Minimum improvement threshold
DISABLE_EARLY_STOPPING=false    # Set to true to disable early stopping

echo "========================================"
echo "EstraNet Training Script"
echo "========================================"
echo "Dataset: ${DATASET}"
echo "Data path: ${DATA_PATH}"
echo "Mode: $1"
echo "========================================"
echo ""

if [[ $1 == 'train' ]]; then
    EARLY_STOP_FLAG=""
    if [[ "$DISABLE_EARLY_STOPPING" == "true" ]]; then
        EARLY_STOP_FLAG="--disable_early_stopping"
    fi
    
    python train_trans.py \
        --data_path=${DATA_PATH} \
        --dataset=${DATASET} \
        --checkpoint_dir=${CKP_DIR} \
        --warm_start \
        --result_path=${RESULT_PATH} \
        --learning_rate=${LEARNING_RATE} \
        --clip=${CLIP} \
        --min_lr_ratio=${MIN_LR_RATIO} \
        --warmup_steps=${WARMUP_STEPS} \
        --input_length=${INPUT_LENGTH} \
        --data_desync=${DATA_DESYNC} \
        --train_batch_size=${TRAIN_BSZ} \
        --eval_batch_size=${EVAL_BSZ} \
        --train_steps=${TRAIN_STEPS} \
        --iterations=${ITERATIONS} \
        --save_steps=${SAVE_STEPS} \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_head=${D_HEAD} \
        --n_head=${N_HEAD} \
        --d_inner=${D_INNER} \
        --n_head_softmax=${N_HEAD_SM} \
        --d_head_softmax=${D_HEAD_SM} \
        --dropout=${DROPOUT} \
        --conv_kernel_size=${CONV_KERNEL_SIZE} \
        --n_conv_layer=${N_CONV_LAYER} \
        --pool_size=${POOL_SIZE} \
        --d_kernel_map=${D_KERNEL_MAP} \
        --beta_hat_2=${BETA_HAT_2} \
        --model_normalization=${MODEL_NORM} \
        --head_initialization=${HEAD_INIT} \
        --softmax_attn \
        --max_eval_batch=${MAX_EVAL_BATCH} \
        --early_stopping_patience=${EARLY_STOPPING_PATIENCE} \
        --early_stopping_delta=${EARLY_STOPPING_DELTA} \
        ${EARLY_STOP_FLAG} \
        --do_train

elif [[ $1 == 'test' ]]; then
    python train_trans.py \
        --data_path=${DATA_PATH} \
        --dataset=${DATASET} \
        --checkpoint_dir=${CKP_DIR} \
        --result_path=${RESULT_PATH} \
        --input_length=${INPUT_LENGTH} \
        --eval_batch_size=${EVAL_BSZ} \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_head=${D_HEAD} \
        --n_head=${N_HEAD} \
        --d_inner=${D_INNER} \
        --n_head_softmax=${N_HEAD_SM} \
        --d_head_softmax=${D_HEAD_SM} \
        --dropout=${DROPOUT} \
        --conv_kernel_size=${CONV_KERNEL_SIZE} \
        --n_conv_layer=${N_CONV_LAYER} \
        --pool_size=${POOL_SIZE} \
        --d_kernel_map=${D_KERNEL_MAP} \
        --beta_hat_2=${BETA_HAT_2} \
        --model_normalization=${MODEL_NORM} \
        --head_initialization=${HEAD_INIT} \
        --softmax_attn \
        --max_eval_batch=${MAX_EVAL_BATCH}

else
    echo "Usage: bash run_trans_ascadf.sh [train|test]"
fi