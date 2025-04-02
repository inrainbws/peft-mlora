#!/bin/bash
# cluster_enable_conda_and_run_command.sh

CONDA_ENV_NAME="dx"

source $HOME/.bashrc
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

echo "enabled conda env $CONDA_ENV_NAME"

command=$1
echo $command
$command