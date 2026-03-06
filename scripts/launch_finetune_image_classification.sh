#!/bin/bash

# number of gpus per task
n_workers=1
counter=20

exp_args_list=(
#  "--r 4 --tune_layernorm"
#  "--r 16 --use_mlora"
#  "--r 16 --use_mlora --use_normal_init"
#  "--r 16 --use_mlora --use_exp"
#  "--r 16 --use_mlora --use_exp --use_weight_norm"
  "--r 4 --use_mlora --lr_multiplier 4. --lora_alpha 1. --tune_layernorm"
  "--r 4 --use_mlora --mlora_init_mode normal --lr_multiplier 16. --lora_alpha 1. --tune_layernorm"
#  "--r 4 --use_mlora --mlora_init_mode uniform --lr_multiplier 16. --lora_alpha 1. --tune_layernorm"
#  "--r 32 --use_mlora --use_exp --fix_b"
)

common_args="--base_model vit"

for exp_args in "${exp_args_list[@]}"; do

  exp_name="mlora-vit-$counter"
  runai delete job $exp_name

  command="python apps/image_classification.py ${common_args} ${exp_args}"
  echo "$exp_name | $command"

  bash scripts/launch_runai.sh $exp_name $n_workers "$command"

  ((counter++))

done