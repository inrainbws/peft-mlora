#!/bin/bash
# Usage
# bash runai_one.sh job_name num_gpu "command"
# Examples:
#	`bash runai_one.sh name-hello-1 1 "python hello.py"`
#	- creates a job names `name-hello-1`
#	- uses 1 GPU
#	- enters MY_WORK_DIR directory (set below) and runs `python hello.py`
#
#	`bash runai_one.sh name-hello-2 0.5 "python hello_half.py"`
#	- creates a job names `name-hello-2`
#	- receives half of a GPUs memory, 2 such jobs can fit on one GPU!
#	- enters MY_WORK_DIR directory (set below) and runs `python hello_half.py`

arg_job_name=$1
arg_gpu=$2
command=$3

CLUSTER_USER=zyang # find this by running `id -un` on iccvlabsrv
CLUSTER_USER_ID=269124 # find this by running `id -u` on iccvlabsrv
CLUSTER_GROUP_NAME=cvlab # find this by running `id -gn` on iccvlabsrv
CLUSTER_GROUP_ID=11166 # find this by running `id -g` on iccvlabsrv

MY_IMAGE="ic-registry.epfl.ch/cvlab/lis/lab-python-ml:cuda11"
NEW_HOME="/scratch/cvlab/home/$CLUSTER_USER"
MY_WORK_DIR="$NEW_HOME/codespace/multiplicative_lora/peft.pytorch.deploy"

RUNAI_PROJECT="cvlab-$CLUSTER_USER"

echo "Job [$arg_job_name] with $arg_gpu gpus"

runai submit $arg_job_name \
	-i $MY_IMAGE \
	-p $RUNAI_PROJECT \
	--node-pools default \
	--gpu $arg_gpu \
	--cpu 16 --memory 120G \
	--pvc cvlab-scratch:/scratch \
	--large-shm \
	-e CLUSTER_USER=$CLUSTER_USER \
	-e CLUSTER_USER_ID=$CLUSTER_USER_ID \
	-e CLUSTER_GROUP_NAME=$CLUSTER_GROUP_NAME \
	-e CLUSTER_GROUP_ID=$CLUSTER_GROUP_ID \
	-e TORCH_HOME="/scratch/cvlab/home/pytorch_model_zoo" \
	-e NEW_HOME=$NEW_HOME \
	--command -- bash $NEW_HOME/setup_and_run_command.sh "cd $MY_WORK_DIR && bash ./scripts/cluster_enable_conda_and_run_command.sh \"$3\" "

# check if succeeded
if [ $? -eq 0 ]; then
	sleep 1
	runai describe job $arg_job_name
fi