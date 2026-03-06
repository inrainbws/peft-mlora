#!/bin/bash

# number of gpus per task
n_workers=1
counter=1

exp_args_list=(
  "--art_style Neo-Impressionism --output_dir output/lora-dreambooth-neo-impressionism --use_lora --lora_r 8"
  "--art_style Constructivism --output_dir output/lora-dreambooth-constructivism --use_lora --lora_r 8"
  "--art_style Neo-Expressionism --output_dir output/lora-dreambooth-neo-expressionism --use_lora --lora_r 8"
  "--art_style Post-Minimalism --output_dir output/lora-dreambooth-post-minimalism --use_lora --lora_r 8"
  "--art_style Neo-Impressionism --output_dir output/mlora-dreambooth-neo-impressionism --use_lora --use_mlora --lora_r 8"
  "--art_style Constructivism --output_dir output/mlora-dreambooth-constructivism --use_lora --use_mlora --lora_r 8"
  "--art_style Neo-Expressionism --output_dir output/mlora-dreambooth-neo-expressionism --use_lora --use_mlora --lora_r 8"
  "--art_style Post-Minimalism --output_dir output/mlora-dreambooth-post-minimalism --use_lora --use_mlora --lora_r 8"
)

common_args="--num_train_images 1000 --max_train_steps 5000 --evaluation_images 256 --resolution 512 --report_to wandb"

for exp_args in "${exp_args_list[@]}"; do

  exp_name="dreambooth-$counter"
  runai delete job $exp_name

  command="accelerate launch apps/dreambooth_wikiart.py ${common_args} ${exp_args}"
  echo "$exp_name | $command"

  bash scripts/launch_runai.sh $exp_name $n_workers "$command"

  ((counter++))

done