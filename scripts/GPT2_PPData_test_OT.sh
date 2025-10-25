model_name=GPT2
batch_size=24
d_model=16
d_ff=32

comment='GPT2-PPData-OT-Only'

# Checkpoint path - using the multivariate model but only visualizing OT
checkpoint_path="./checkpoints/short_term_forecast_PPData_90_6_GPT2_PPData_ftM_sl90_ll6_pl6_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_Powerplantdata_0-GPT2-PPData/checkpoint"

# Configuration for predictions
# IMPORTANT: Your model was trained with pred_len=6
# You must use the same pred_len for testing, or retrain a new model
pred_len=6       # MUST match your training configuration
label_len=6      # MUST match your training configuration

# If you want to predict more steps (e.g., 30, 60), you need to:
# 1. First train a new model with that pred_len
# 2. Then update checkpoint_path to point to the new model
# 3. Then change pred_len here to match

# Visualization settings
plot_samples=10           # Number of individual prediction windows to show
continuous_steps=200      # Number of continuous time steps to visualize (e.g., 200 seconds)

python run_test_ppdata_ot.py \
  --task_name short_term_forecast \
  --root_path ./dataset/ \
  --data_path PPData.csv \
  --model_id PPData_90_${pred_len}_OT \
  --model $model_name \
  --data PPData \
  --features M \
  --target OT \
  --freq 's' \
  --seq_len 90 \
  --label_len $label_len \
  --pred_len $pred_len \
  --factor 3 \
  --enc_in 43 \
  --dec_in 43 \
  --c_out 43 \
  --llm_model LLAMA \
  --llm_dim 4096 \
  --des 'Power plant data - OT prediction' \
  --llm_layers 6 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --model_comment $comment \
  --checkpoint_path $checkpoint_path \
  --plot_samples $plot_samples \
  --continuous_steps $continuous_steps \
  --use_gpu
