model_name=GPT2
train_epochs=10
learning_rate=0.001

batch_size=24
d_model=16
d_ff=32

comment='GPT2-PPData'

accelerate launch --mixed_precision bf16  run_main.py \
  --task_name short_term_forecast \
  --is_training 1 \
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
  --llm_dim 4096 \
  --des 'Powerplantdata' \
  --itr 1 \
  --llm_layers 6 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --model_comment $comment
