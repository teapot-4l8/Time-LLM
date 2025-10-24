model_name=GPT2
batch_size=24
d_model=16
d_ff=32

comment='GPT2-PPData'

# Checkpoint path - update this to your trained model checkpoint
# The checkpoint should be in a folder like:
# ./checkpoints/short_term_forecast_PPData_90_6_GPT2_PPData_ftM_sl90_ll6_pl6_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_Power plant data_0-GPT2-PPData/checkpoint
checkpoint_path="./checkpoints/short_term_forecast_PPData_90_6_GPT2_PPData_ftM_sl90_ll6_pl6_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_Powerplantdata_0-GPT2-PPData/checkpoint"

python run_test_ppdata_simple.py \
  --task_name short_term_forecast \
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
  --factor 3 \
  --enc_in 43 \
  --dec_in 43 \
  --c_out 43 \
  --llm_model LLAMA \
  --llm_dim 4096 \
  --des 'Power plant data' \
  --llm_layers 6 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --model_comment $comment \
  --checkpoint_path $checkpoint_path \
  --plot_samples 5 \
  --plot_features 3 \
  --use_gpu
