#!/bin/bash

nvidia-smi
export batch_size=32
export epochs=30
export retrain=5
export num_data_w=1
# export newload='enable'
export amp='disable'
export MASTER_ADDR=0.0.0.0
export MASTER_PORT=8882
export SLURM_PROCID=0
export append=$RANDOM
echo "append: $append"
touch output_$append.out
echo "output_$append.out"
date
time docker run --gpus device=all --env MASTER_ADDR --env MASTER_PORT --env SLURM_PROCID --env append --rm  -w /code -v /run/user/1001/2dnet:/projects/synergy_lab/garvit217/enhancement_data -v $PWD:/code nvcr.io/nvidia/pytorch:21.12-py3 python sparse_ddnet.py -n 1 -g 1 --batch $batch_size --epochs $epochs --retrain $retrain --amp $amp > output_$RANDOM.out 2>&1 &
