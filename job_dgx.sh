#!/bin/bash

nvidia-smi
export batch_size=8
export epochs=30
export retrain=5
export num_data_w=1

export MASTER_ADDR=0.0.0.0
export MASTER_PORT=8882
export SLURM_PROCID=0
export append=$RANDOM
echo "append: $append"
touch output_$append.out
echo "output_89.out"
date
time docker run --gpus  device=GPU-bbb786b9-5095-facf-71ba-63d386278e07 --env MASTER_ADDR --env MASTER_PORT --env SLURM_PROCID --env append --rm  -w /code -v /run/user/1001/2dnet:/projects/synergy_lab/garvit217/enhancement_data/ -v ~/Documents/2dnet/2dnet:/code nvcr.io/nvidia/pytorch:21.12-py3 python sparse_ddnet.py -n 1 -g 1 --batch $batch_size --epochs $epochs --retrain $retrain > output_$append.out &
