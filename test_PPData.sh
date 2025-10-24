#!/bin/bash

# Configuration matching GPT2_PPData.sh
model_name=GPT2
llm_layers=6
master_port=00097
num_process=1
batch_size=24
d_model=16
d_ff=32

# IMPORTANT: Set this to your actual checkpoint path!
# The checkpoint is saved during training with the naming pattern:
# ./checkpoints/{setting}-{model_comment}/checkpoint
# Expected path based on GPT2_PPData.sh:
# ./checkpoints/short_term_forecast_PPData_90_6_GPT2_PPData_ftM_sl90_ll6_pl6_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_Power plant data_0-GPT2-PPData/checkpoint

CHECKPOINT_PATH="./checkpoints/short_term_forecast_PPData_90_6_GPT2_PPData_ftM_sl90_ll6_pl6_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_Power plant data_0-GPT2-PPData/checkpoint"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    echo ""
    echo "Available checkpoints:"
    find ./checkpoints -name "checkpoint" -type f 2>/dev/null
    echo ""
    echo "Please set the correct checkpoint path in this script"
    exit 1
fi

comment='GPT2-PPData'

# Test for short-term forecast (prediction length 6)
echo "=========================================="
echo "Testing GPT2 model for PPData"
echo "Task: Short-term forecast (6 timesteps)"
echo "=========================================="
echo ""
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_test.py \
  --task_name short_term_forecast \
  --checkpoint_path $CHECKPOINT_PATH \
  --root_path ./dataset/ \
  --data_path PPData.csv \
  --model_id PPData_90_6 \
  --model $model_name \
  --data PPData \
  --features M \
  --freq 's' \
  --seq_len 90 \
  --label_len 6 \
  --pred_len 6 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 43 \
  --dec_in 43 \
  --c_out 43 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --llm_layers $llm_layers \
  --llm_dim 4096 \
  --des 'Power plant data' \
  --model_comment $comment

echo ""
echo "=========================================="
echo "Testing completed!"
echo "Results saved to ./test_results/"
echo "=========================================="
