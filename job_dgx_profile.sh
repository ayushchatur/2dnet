#!/bin/bash

nvidia-smi
export batch_size=32
export epochs=30
export retrain=5

export MASTER_ADDR=0.0.0.0
export MASTER_PORT=8882
export SLURM_PROCID=0
export append=$RANDOM
export newload='enable'
export amp='disable'
touch output_${append}.out
echo "output_${append}.out"
date
time docker run --env MASTER_ADDR --env MASTER_PORT --env SLURM_PROCID --env append --rm --gpus 0,1 -w /code -v /run/user/1001/2dnet:/projects/synergy_lab/garvit217/enhancement_data/ -v $PWD:/code nvcr.io/nvidia/pytorch:21.12-py3 dlprof --output_path=data_loa --profile_name=data_load --dump_model_data=true --mode=pytorch --nsys_opts="-t osrt,cuda,nvtx,cudnn,cublas --cuda-memory-usage=true --gpuctxsw=true " -f true --reports=all --delay 120 --duration 60 python sparse_ddnet_pp.py -n 1 -g 1 --batch $batch_size --epochs $epochs --retrain $retrain --new_load $newload --amp $amp> output_$append.out 2>&1 &
