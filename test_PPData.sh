#!/bin/bash

# Configuration
model_name=TimeLLM
llama_layers=32
master_port=00097
num_process=1
batch_size=16
d_model=16
d_ff=32

# IMPORTANT: Set this to your actual checkpoint path!
# The checkpoint is saved during training with the naming pattern:
# ./checkpoints/{setting}-{model_comment}/checkpoint
# Example: ./checkpoints/long_term_forecast_PPData_96_96_TimeLLM_PPData_ftM_sl96_ll48_pl96_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-TimeLLM-PPData/checkpoint

CHECKPOINT_PATH="./checkpoints/long_term_forecast_PPData_96_96_TimeLLM_PPData_ftM_sl96_ll48_pl96_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-TimeLLM-PPData/checkpoint"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Please set the correct checkpoint path in this script"
    exit 1
fi

comment='TimeLLM-PPData'

# Test for prediction length 96
echo "Testing model with prediction length 96..."
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_test.py \
  --task_name long_term_forecast \
  --checkpoint_path $CHECKPOINT_PATH \
  --root_path ./dataset/ \
  --data_path PPData.csv \
  --model_id PPData_96_96 \
  --model $model_name \
  --data PPData \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 43 \
  --dec_in 43 \
  --c_out 43 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --llm_layers $llama_layers \
  --model_comment $comment

echo "Testing completed!"
