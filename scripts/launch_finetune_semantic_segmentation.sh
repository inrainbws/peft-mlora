#!/bin/bash

# number of gpus per task
n_workers=1
counter=1

exp_args_list=(
#  "--r 16"
#  "--r 16 --use_mlora"
#  "--r 16 --use_mlora --use_exp"
#  "--r 16 --use_mlora --use_exp --use_weight_norm"
#  "--r 32 --use_mlora --use_exp --fix_b"
"--r 16 --use_mlora --use_normal_init --lr_multiplier 1."
"--r 16 --use_mlora --use_normal_init --use_weight_norm --lr_multiplier 1."
"--r 16 --use_mlora --use_normal_init --lr_multiplier 10."
"--r 16 --use_mlora --use_normal_init --use_weight_norm --lr_multiplier 10."
)

common_args="--base_model upernet"

for exp_args in "${exp_args_list[@]}"; do

  exp_name="mlora-segmentation-upernet-$counter"
  runai delete job $exp_name

  command="python apps/semantic_segmentation.py ${common_args} ${exp_args}"
  echo "$exp_name | $command"

  bash scripts/launch_runai.sh $exp_name $n_workers "$command"

  ((counter++))

done