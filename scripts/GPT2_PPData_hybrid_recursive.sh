model_name=GPT2
batch_size=1  # Must be 1 for recursive prediction
d_model=16
d_ff=32

comment='GPT2-PPData-Hybrid-Recursive-OT'

# Checkpoint path - using the multivariate model
checkpoint_path="./checkpoints/short_term_forecast_PPData_90_6_GPT2_PPData_ftM_sl90_ll6_pl6_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_Powerplantdata_0-GPT2-PPData/checkpoint"

# Configuration for predictions
# IMPORTANT: Your model was trained with pred_len=6
# The script will use pred_len=6 but only take the first step (step 0) for recursion
pred_len=6       # MUST match your training configuration
label_len=6      # MUST match your training configuration

# Recursive prediction settings
recursive_steps=200      # Number of recursive prediction steps to perform

# NOTE: Hybrid Recursive Prediction Strategy:
# - Input [0-89]: Use all ground truth features → Predict step 90
# - Take only the FIRST prediction (index 0) from the pred_len=6 output
# - Update ONLY the OT column with this prediction
# - Keep all other 42 features as ground truth
# - Input [1-90]: Mixed data (OT=predicted, others=ground truth) → Predict step 91
# - Repeat for recursive_steps iterations

python run_hybrid_recursive.py \
  --task_name short_term_forecast \
  --root_path ./dataset/ \
  --data_path PPData.csv \
  --model_id PPData_90_${pred_len}_hybrid_recursive \
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
  --des 'Power plant data - Hybrid recursive OT prediction' \
  --llm_layers 6 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --model_comment $comment \
  --checkpoint_path $checkpoint_path \
  --recursive_steps $recursive_steps \
  --use_gpu
