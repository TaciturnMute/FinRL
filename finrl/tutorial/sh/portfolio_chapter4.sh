#!/bin/bash

# 数据参数组
STOCKS_DIR="/mnt/finrl/data/csv/DOW_30/"
VIX_DATA_DIR="/mnt/finrl/data/csv/^VIX/"
DJI_DIR="/mnt/finrl/data/csv/DJI/DJI.csv"
TRAIN_START_DATE="2010-01-01"
TRAIN_END_DATE="2021-12-31"
VALIDATION_START_DATE="2022-01-01"
VALIDATION_END_DATE="2022-12-31"
TEST_START_DATE="2023-01-01"
TEST_END_DATE="2024-01-30"
INDICATORS="macd,rsi_30,cci_30,dx_30"  
IF_INDICATOR_NORM="" # store_true

# 环境参数组
HMAX=100
INITIAL_AMOUNT=1000000.0
REWARD_SCALING=0.0001
BUY_COST_PCT=0.001
SELL_COST_PCT=0.001

# 策略参数组
M=2
N=3
CNN_ACTIVATION="relu"
LSTM_INPUT_SIZE=704
LSTM_HIDDEN_SIZE=1024
MLP_ACTIVATION="relu"
SRL_ALIASE="d2rl"
SRL_HIDDEN_DIM=512

# OU噪声参数组
SIGMA=0.05
THETA=0.10
DT=0.1
RANDOMNESS="" # store_false

# agent参数组
BUFFER_SIZE=10000
BATCH_SIZE=4
N_UPDATES=5
GAMMA=0.99
TAU=0.005
POLICY_LR=0.00000005
NOISE_ALIASE="ou"
TRAINING_START=200
TARGET_UPDATE_INTERVAL=1
PRINT_INTERVAL=200
FIGURE_PATH="/mnt/finrl/data/figure/figures_DJIA/"
DEVICE="cuda"
IF_CLIP="" # store_true
TASK="portfolio"
Q_TARGET_MODE="redq"
TOTAL_UPDATES_TIMES_MAXIMUM=138400
MCTS_GAMMA=0.99
EXPAND_LENGTH=50
CHILDREN_MAXIMUM=5
MCTS_C=0.5
RANDOM_SELECT_PROB=0.2


python /mnt/finrl/tutorial/py/portfolio_chapter4.py \
    --stocks_dir $STOCKS_DIR \
    --vix_data_dir $VIX_DATA_DIR \
    --dji_dir $DJI_DIR \
    --train_start_date $TRAIN_START_DATE \
    --train_end_date $TRAIN_END_DATE \
    --validation_start_date $VALIDATION_START_DATE \
    --validation_end_date $VALIDATION_END_DATE \
    --test_start_date $TEST_START_DATE \
    --test_end_date $TEST_END_DATE \
    --indicators $INDICATORS \
    $IF_INDICATOR_NORM \
    --hmax $HMAX \
    --initial_amount $INITIAL_AMOUNT \
    --reward_scaling $REWARD_SCALING \
    --buy_cost_pct $BUY_COST_PCT \
    --sell_cost_pct $SELL_COST_PCT \
    --cnn_activation $CNN_ACTIVATION \
    --lstm_input_size $LSTM_INPUT_SIZE \
    --lstm_hidden_size $LSTM_HIDDEN_SIZE \
    --mlp_activation $MLP_ACTIVATION \
    --srl_aliase $SRL_ALIASE \
    --srl_hidden_dim $SRL_HIDDEN_DIM \
    --sigma $SIGMA \
    --theta $THETA \
    --dt $DT \
    $RANDOMNESS \
    --buffer_size $BUFFER_SIZE \
    --batch_size $BATCH_SIZE \
    --n_updates $N_UPDATES \
    --gamma $GAMMA \
    --tau $TAU \
    --policy_lr $POLICY_LR \
    --noise_aliase $NOISE_ALIASE \
    --training_start $TRAINING_START \
    --target_update_interval $TARGET_UPDATE_INTERVAL \
    --print_interval $PRINT_INTERVAL \
    --figure_path $FIGURE_PATH \
    --device $DEVICE \
    $IF_CLIP \
    --task $TASK \
    --q_target_mode $Q_TARGET_MODE \
    --total_updates_times_maximum $TOTAL_UPDATES_TIMES_MAXIMUM \
    --mcts_gamma $MCTS_GAMMA \
    --expand_length $EXPAND_LENGTH \
    --children_maximum $CHILDREN_MAXIMUM \
    --mcts_C $MCTS_C \
    --random_select_prob $RANDOM_SELECT_PROB \
